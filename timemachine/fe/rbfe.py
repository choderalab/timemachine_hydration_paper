import pickle
import warnings
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray
from openmm import app
from rdkit import Chem

from timemachine.constants import DEFAULT_POSITIONAL_RESTRAINT_K, DEFAULT_PRESSURE, DEFAULT_TEMP
from timemachine.fe import model_utils
from timemachine.fe.free_energy import (
    HostConfig,
    HREXParams,
    HREXPlots,
    HREXSimulationResult,
    InitialState,
    MDParams,
    SimulationResult,
    Trajectory,
    make_pair_bar_plots,
    run_sims_bisection,
    run_sims_hrex,
    run_sims_sequential,
)
from timemachine.fe.lambda_schedule import bisection_lambda_schedule
from timemachine.fe.plots import (
    plot_as_png_fxn,
    plot_hrex_replica_state_distribution_heatmap,
    plot_hrex_swap_acceptance_rates_convergence,
    plot_hrex_transition_matrix,
)
from timemachine.fe.single_topology import AtomMapFlags, SingleTopology
from timemachine.fe.system import VacuumSystem, convert_omm_system
from timemachine.fe.utils import bytes_to_id, get_mol_name, get_romol_conf
from timemachine.ff import Forcefield
from timemachine.lib import LangevinIntegrator, MonteCarloBarostat
from timemachine.md import builders, minimizer
from timemachine.md.barostat.utils import get_bond_list, get_group_indices
from timemachine.md.thermostat.utils import sample_velocities
from timemachine.potentials import BoundPotential, jax_utils

DEFAULT_NUM_WINDOWS = 48

# the constant is arbitrary, but see
# https://github.com/proteneer/timemachine/commit/e1f7328f01f427534d8744aab6027338e116ad09
MAX_SEED_VALUE = 10000

DEFAULT_MD_PARAMS = MDParams(n_frames=1000, n_eq_steps=10_000, steps_per_frame=400, seed=2023, hrex_params=None)

DEFAULT_HREX_PARAMS = replace(DEFAULT_MD_PARAMS, hrex_params=HREXParams(n_frames_bisection=100))


@dataclass
class Host:
    system: VacuumSystem
    physical_masses: List[float]
    conf: NDArray
    box: NDArray
    num_water_atoms: int
    omm_topology: app.topology.Topology


def setup_in_vacuum(st: SingleTopology, ligand_conf, lamb):
    """Prepare potentials, initial coords, large 10x10x10nm box, and HMR masses"""

    system = st.setup_intermediate_state(lamb)
    hmr_masses = np.array(st.combine_masses(use_hmr=True))

    potentials = system.get_U_fns()
    baro = None

    x0 = ligand_conf
    box0 = np.eye(3, dtype=np.float64) * 10  # make a large 10x10x10nm box

    return x0, box0, hmr_masses, potentials, baro


def setup_in_env(
    st: SingleTopology,
    host: Host,
    ligand_conf: NDArray,
    lamb: float,
    temperature: float,
    run_seed: int,
):
    """Prepare potentials, concatenate environment and ligand coords, apply HMR, and construct barostat"""
    barostat_interval = 25
    system = st.combine_with_host(host.system, lamb, host.num_water_atoms, st.ff, host.omm_topology)
    host_hmr_masses = model_utils.apply_hmr(host.physical_masses, host.system.bond.potential.idxs)
    hmr_masses = np.concatenate([host_hmr_masses, st.combine_masses(use_hmr=True)])

    potentials = system.get_U_fns()
    group_idxs = get_group_indices(get_bond_list(system.bond.potential), len(hmr_masses))
    baro = MonteCarloBarostat(
        len(hmr_masses), DEFAULT_PRESSURE, temperature, group_idxs, barostat_interval, run_seed + 1
    )

    x0 = np.concatenate([host.conf, ligand_conf])

    return x0, hmr_masses, potentials, baro


def assert_all_states_have_same_masses(initial_states: List[InitialState]):
    """
    hmr masses should be identical throughout the lambda schedule
    bond idxs should be the same at the two end-states, note that a possible corner
    case with bond breaking may seem to be problematic:

    0 1 2    0 1 2
    C-O-C -> C.H-C

    but this isn't an issue, since hydrogens will only ever be terminal atoms
    and core hydrogens that are mapped to heavy atoms will take the mass of the
    heavy atom (thereby not triggering the mass repartitioning to begin with).

    but it's reasonable to be skeptical, so we also assert consistency through the lambda
    schedule as an extra sanity check.
    """

    masses = np.array([s.integrator.masses for s in initial_states])
    deviation_among_windows = masses.std(0)
    np.testing.assert_array_almost_equal(deviation_among_windows, 0, err_msg="masses assumed constant w.r.t. lambda")


def setup_initial_state(
    st: SingleTopology,
    lamb: float,
    host: Optional[Host],
    temperature: float,
    seed: int,
) -> InitialState:
    conf_a = get_romol_conf(st.mol_a)
    conf_b = get_romol_conf(st.mol_b)

    ligand_conf = st.combine_confs(conf_a, conf_b, lamb)
    num_ligand_atoms = len(ligand_conf)
    # use a different seed to initialize every window,
    # but in a way that should be symmetric for
    # A -> B vs. B -> A edge definitions
    init_seed = int(seed + bytes_to_id(ligand_conf.tobytes())) % MAX_SEED_VALUE
    if host:
        x0, hmr_masses, potentials, baro = setup_in_env(st, host, ligand_conf, lamb, temperature, init_seed)
        box0 = host.box
        protein_idxs = np.arange(0, len(host.physical_masses) - host.num_water_atoms)
    else:
        x0, box0, hmr_masses, potentials, baro = setup_in_vacuum(st, ligand_conf, lamb)
        protein_idxs = np.array([], dtype=np.int32)

    # provide a different run_seed for every lambda window,
    # but in a way that should be symmetric for
    # A -> B vs. B -> A edge definitions
    run_seed = (
        int(seed + bytes_to_id(bytes().join([np.array(p.params).tobytes() for p in potentials]))) % MAX_SEED_VALUE
    )

    # initialize velocities
    v0 = sample_velocities(hmr_masses, temperature, init_seed)

    # determine ligand idxs

    num_total_atoms = len(x0)
    ligand_idxs = np.arange(num_total_atoms - num_ligand_atoms, num_total_atoms, dtype=np.int32)

    # initialize Langevin integrator
    dt = 2.5e-3
    friction = 1.0
    intg = LangevinIntegrator(temperature, dt, friction, hmr_masses, run_seed)

    # Determine the atoms that are in the 4d plane defined by all w_coords being 0.0
    # TBD: Do something more sophisticated depending on the actual parameters for when we vary w_coords independently.
    # Easily done for solvent and complex, little bit trickier in the vacuum case where there is a NonbondedPairlistPrecomputed
    if lamb == 0.0:
        interacting_atoms = ligand_idxs[st.c_flags != AtomMapFlags.MOL_B]
    elif lamb == 1.0:
        interacting_atoms = ligand_idxs[st.c_flags != AtomMapFlags.MOL_A]
    else:
        interacting_atoms = ligand_idxs[st.c_flags == AtomMapFlags.CORE]

    return InitialState(
        potentials, intg, baro, x0, v0, box0, lamb, ligand_idxs, protein_idxs, interacting_atoms=interacting_atoms
    )


def setup_optimized_host(st: SingleTopology, config: HostConfig) -> Host:
    """
    Optimize a SingleTopology host using pre_equilibrate_host

    Parameters
    ----------
    st: SingleTopology
        A single topology object

    config: HostConfig
        Host configuration

    Returns
    -------
    Host
        Minimized host state
    """
    system, masses = convert_omm_system(config.omm_system)
    conf, box = minimizer.pre_equilibrate_host([st.mol_a, st.mol_b], config, st.ff)
    return Host(system, masses, conf, box, config.num_water_atoms, config.omm_topology)


def setup_initial_states(
    st: SingleTopology,
    host: Optional[Host],
    temperature: float,
    lambda_schedule: Union[NDArray, Sequence[float]],
    seed: int,
    min_cutoff: Optional[float] = None,
) -> List[InitialState]:
    """
    Given a sequence of lambda values, return a list of initial states.

    The InitialState objects can be used to recover a bitwise-identical simulation for debugging.

    Assumes lambda schedule is a monotonically increasing sequence in the closed interval [0, 1].

    Parameters
    ----------
    st: SingleTopology
        A single topology object

    host: Host or None
        Pre-optimized host configuration, generated using `setup_optimized_host`. If None, return vacuum states.

    temperature: float
        Temperature to run the simulation at.

    lambda_schedule: list of float of length K
        Lambda schedule.

    seed: int
        Random number seed

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization. Typically only meaningful
        in the complex leg where the check may indicate that the ligand is no longer posed reliably.

    Returns
    -------
    list of InitialState
        Initial state for each value of lambda.
    """

    # check that the lambda schedule is monotonically increasing.
    assert np.all(np.diff(lambda_schedule) > 0)

    initial_states = [setup_initial_state(st, lamb, host, temperature, seed) for lamb in lambda_schedule]

    # minimize ligand and environment atoms within min_cutoff of the ligand
    # optimization introduces dependencies among states with lam < 0.5, and among states with lam >= 0.5
    optimized_x0s = optimize_coordinates(initial_states, min_cutoff=min_cutoff)

    # update initial states in-place
    for state, x0 in zip(initial_states, optimized_x0s):
        state.x0 = x0

    # perform any concluding sanity-checks
    assert_all_states_have_same_masses(initial_states)

    return initial_states


def setup_optimized_initial_state(
    st: SingleTopology,
    lamb: float,
    host: Optional[Host],
    optimized_initial_states: Sequence[InitialState],
    temperature: float,
    seed: int,
    k: Optional[float] = DEFAULT_POSITIONAL_RESTRAINT_K,
) -> InitialState:
    """Setup an InitialState for the specified lambda and optimize the coordinates given a list of pre-optimized IntialStates.
    If the specified lambda exists within the list of optimized_initial_states list, return the existing InitialState.
    The coordinates of the pre-optimized initial state with the closest value of lambda will be optimized to the new lambda value.

    Note
    ----
    When determining the nearest initial state, will only consider states on the same side of lambda=0.5 as the specified lambda value.
    This imitates the behavior of `optimize_coordinates` which minimizes from the endstate conformations towards lambda 0.5, resulting
    in a discontinuity in the conformation at lambda=0.5.

    Returns
    -------
    InitialState
        Optimized at the specified lambda value
    """

    # NOTE: The current approach for generating optimized conformations in `optimize_coordinates` creates a
    # discontinuity at lambda=0.5. Ensure that we pick a pre-optimized state on the same side of 0.5 as `lamb`
    states_subset = [s for s in optimized_initial_states if (s.lamb <= 0.5) == (lamb <= 0.5)]
    nearest_optimized = min(states_subset, key=lambda s: abs(lamb - s.lamb))

    if np.isclose(lamb, nearest_optimized.lamb):
        return nearest_optimized
    else:
        initial_state = setup_initial_state(st, lamb, host, temperature, seed)
        free_idxs = get_free_idxs(nearest_optimized)
        initial_state.x0 = optimize_coords_state(
            initial_state.potentials,
            nearest_optimized.x0,
            initial_state.box0,
            free_idxs,
            # assertion can lead to spurious errors when new state is close to an existing one
            assert_energy_decreased=False,
            k=k,
        )
        return initial_state


def optimize_coords_state(
    potentials: Sequence[BoundPotential],
    x0: NDArray,
    box: NDArray,
    free_idxs: List[int],
    assert_energy_decreased: bool,
    k: Optional[float],
) -> NDArray:
    val_and_grad_fn = minimizer.get_val_and_grad_fn(potentials, box)
    assert np.all(np.isfinite(x0)), "Initial coordinates contain nan or inf"

    x_opt = minimizer.local_minimize(
        x0, box, val_and_grad_fn, free_idxs, assert_energy_decreased=assert_energy_decreased, restraint_k=k
    )
    assert np.all(np.isfinite(x_opt)), "Minimization resulted in a nan"
    return x_opt


def get_free_idxs(initial_state: InitialState, cutoff: float = 0.5) -> List[int]:
    """Select particles within cutoff of ligand"""
    x = initial_state.x0
    x_lig = x[initial_state.ligand_idxs]
    box = initial_state.box0
    free_idxs = jax_utils.idxs_within_cutoff(x, x_lig, box, cutoff=cutoff).tolist()
    return free_idxs


def _optimize_coords_along_states(initial_states: List[InitialState], k: Optional[float]) -> List[NDArray]:
    # use the end-state to define the optimization settings
    end_state = initial_states[0]

    x_opt = end_state.x0

    x_traj = []
    for idx, initial_state in enumerate(initial_states):
        print(f"Optimizing initial state at λ={initial_state.lamb}")
        free_idxs = get_free_idxs(initial_state)
        try:
            x_opt = optimize_coords_state(
                initial_state.potentials, x_opt, initial_state.box0, free_idxs, assert_energy_decreased=idx == 0, k=k
            )
        except (AssertionError, minimizer.MinimizationError) as e:
            raise minimizer.MinimizationError(f"Failed to optimized state at λ={initial_state.lamb}") from e
        x_traj.append(x_opt)

    return x_traj


def optimize_coordinates(
    initial_states: List[InitialState],
    min_cutoff: Optional[float] = 0.7,
    k: Optional[float] = DEFAULT_POSITIONAL_RESTRAINT_K,
) -> List[NDArray]:
    """
    Optimize geometries of the initial states.

    Parameters
    ----------
    initial_states: list of InitialState

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    k: float, optional
        force constant for a positional harmonic restraint potential to apply to the initial positions.
        If None, minimize with no positional restraint. Refer to `timemachine.potentials.bonded.harmonic_positional_restraint`
        for implementation.

    Returns
    -------
    list of np.array
        Optimized coordinates

    """
    all_xs = []
    lambda_schedule = np.array([s.lamb for s in initial_states])

    # check for monotonic, any subsequence of a monotonic sequence is also monotonic.
    assert np.all(np.diff(lambda_schedule) > 0)

    lhs_initial_states = []
    rhs_initial_states = []

    for state in initial_states:
        if state.lamb < 0.5:
            lhs_initial_states.append(state)
        else:
            rhs_initial_states.append(state)

    # go from lambda 0 -> 0.5
    if len(lhs_initial_states) > 0:
        lhs_xs = _optimize_coords_along_states(lhs_initial_states, k)
        for xs in lhs_xs:
            all_xs.append(xs)

    # go from lambda 1 -> 0.5 and reverse the coordinate trajectory and lambda schedule
    if len(rhs_initial_states) > 0:
        rhs_xs = _optimize_coords_along_states(rhs_initial_states[::-1], k)[::-1]
        for xs in rhs_xs:
            all_xs.append(xs)

    # sanity check that no atom has moved more than `min_cutoff` nm away
    if min_cutoff is not None:
        for state, coords in zip(initial_states, all_xs):
            interacting_atoms = state.interacting_atoms
            # assert that interacting ligand atoms and protein atoms are not allowed to move more than min_cutoff
            if interacting_atoms is None:
                restricted_idxs = state.protein_idxs
            else:
                restricted_idxs = np.concatenate([interacting_atoms, state.protein_idxs])
            displacement_distances = jax_utils.distance_on_pairs(
                state.x0[restricted_idxs], coords[restricted_idxs], box=state.box0
            )
            max_moved = np.max(displacement_distances)
            moved_atoms = restricted_idxs[displacement_distances >= min_cutoff]
            assert (
                len(moved_atoms) == 0
            ), f"λ = {state.lamb} moved atoms {moved_atoms.tolist()} > {min_cutoff*10} Å from initial state during minimization. Largest displacement was {max_moved*10} Å"
    return all_xs


def estimate_relative_free_energy(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    md_params: MDParams = DEFAULT_MD_PARAMS,
    min_cutoff: Optional[float] = 0.7,
) -> SimulationResult:
    """
    Estimate relative free energy between mol_a and mol_b via independent simulations with a predetermined lambda
    schedule. Molecules should be aligned to each other and within the host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    prefix: str
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    min_cutoff: float, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes are of size n_windows.

    """
    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_min, lambda_max = lambda_interval or (0.0, 1.0)
    lambda_schedule = np.linspace(lambda_min, lambda_max, n_windows or DEFAULT_NUM_WINDOWS)

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_schedule, md_params.seed, min_cutoff=min_cutoff
    )

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix
    try:
        result, stored_trajectories = run_sims_sequential(initial_states, md_params, temperature)
        plots = make_pair_bar_plots(result, temperature, combined_prefix)
        return SimulationResult(result, plots, stored_trajectories, md_params, [])
    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((initial_states, md_params, err), fh)
        raise err


def estimate_relative_free_energy_bisection_or_hrex(*args, **kwargs) -> SimulationResult:
    """
    See `estimate_relative_free_energy_bisection` for parameters.

    Will call `estimate_relative_free_energy_bisection` or `estimate_relative_free_energy_bisection_hrex`
    as appropriate given md_params.

    """
    md_params = kwargs["md_params"]
    estimate_fxn = (
        estimate_relative_free_energy_bisection_hrex
        if md_params.hrex_params is not None
        else estimate_relative_free_energy_bisection
    )
    return estimate_fxn(*args, **kwargs)


def estimate_relative_free_energy_bisection(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    md_params: MDParams = DEFAULT_MD_PARAMS,
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    min_cutoff: Optional[float] = 0.7,
) -> SimulationResult:
    r"""Estimate relative free energy between mol_a and mol_b via independent simulations with a dynamic lambda schedule
    determined by successively bisecting the lambda interval between the pair of states with the greatest BAR
    :math:`\Delta G` error. Molecules should be aligned to each other and within the host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    prefix: str, optional
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Additionally controls the
        number of evenly-spaced lambda windows used for initial conformer optimization. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    min_overlap: float or None, optional
        If not None, terminate bisection early when the BAR overlap between all neighboring pairs of states exceeds this
        value. When given, the final number of windows may be less than or equal to n_windows.

    min_cutoff: float or None, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    SimulationResult
        Collected data from the simulation (see class for storage information). Returned frames and boxes are of size n_windows.
    """

    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_interval = lambda_interval or (0.0, 1.0)
    lambda_min, lambda_max = lambda_interval[0], lambda_interval[1]

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    lambda_grid = bisection_lambda_schedule(n_windows, lambda_interval=lambda_interval)

    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_grid, md_params.seed, min_cutoff=min_cutoff
    )

    make_optimized_initial_state = partial(
        setup_optimized_initial_state,
        single_topology,
        host=host,
        optimized_initial_states=initial_states,
        temperature=temperature,
        seed=md_params.seed,
    )

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix

    try:
        results, trajectories = run_sims_bisection(
            [lambda_min, lambda_max],
            make_optimized_initial_state,
            md_params,
            n_bisections=n_windows - 2,
            temperature=temperature,
            min_overlap=min_overlap,
        )

        final_result = results[-1]

        plots = make_pair_bar_plots(final_result, temperature, combined_prefix)

        assert len(trajectories) == len(results) + 1

        return SimulationResult(
            final_result,
            plots,
            trajectories,
            md_params,
            results,
        )

    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((md_params, err), fh)
        raise err


def estimate_relative_free_energy_bisection_hrex_impl(
    temperature: float,
    lambda_min: float,
    lambda_max: float,
    md_params: MDParams,
    n_windows: int,
    make_optimized_initial_state_fn: Callable[[float], InitialState],
    combined_prefix: str,
    min_overlap: Optional[float] = None,
) -> HREXSimulationResult:
    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    try:
        # First phase: bisection to determine lambda spacing
        assert md_params.hrex_params is not None, "hrex_params must be set to use HREX"
        md_params_bisection = replace(md_params, n_frames=md_params.hrex_params.n_frames_bisection)
        results, trajectories_by_state = run_sims_bisection(
            [lambda_min, lambda_max],
            make_optimized_initial_state_fn,
            md_params_bisection,
            n_bisections=n_windows - 2,
            temperature=temperature,
            min_overlap=min_overlap,
        )

        assert all(traj.final_velocities is not None for traj in trajectories_by_state)

        initial_states = results[-1].initial_states
        has_barostat_by_state = [initial_state.barostat is not None for initial_state in initial_states]
        assert all(has_barostat_by_state) or not any(has_barostat_by_state)

        # Second phase: sample initial states determined by bisection using HREX

        def get_mean_final_barostat_volume_scale_factor(trajectories_by_state: Iterable[Trajectory]) -> Optional[float]:
            scale_factors = [traj.final_barostat_volume_scale_factor for traj in trajectories_by_state]
            if any(x is not None for x in scale_factors):
                assert all(x is not None for x in scale_factors)
                sfs = cast(List[float], scale_factors)  # implied by assertion but required by mypy
                return float(np.mean(sfs))
            else:
                return None

        mean_final_barostat_volume_scale_factor = get_mean_final_barostat_volume_scale_factor(trajectories_by_state)
        assert (mean_final_barostat_volume_scale_factor is not None) == all(has_barostat_by_state)

        # Use equilibrated samples and the average of the final barostat volume scale factors from bisection phase to
        # initialize states for HREX
        initial_states_hrex = [
            replace(
                initial_state,
                x0=traj.frames[-1],
                v0=traj.final_velocities,  # type: ignore
                box0=traj.boxes[-1],
                barostat=(
                    replace(
                        initial_state.barostat,
                        adaptive_scaling_enabled=False,
                        initial_volume_scale_factor=mean_final_barostat_volume_scale_factor,
                    )
                    if initial_state.barostat
                    else None
                ),
            )
            for initial_state, traj in zip(initial_states, trajectories_by_state)
        ]

        pair_bar_result, trajectories_by_state, diagnostics = run_sims_hrex(
            initial_states_hrex,
            replace(md_params, n_eq_steps=0),  # using pre-equilibrated samples
        )

        plots = make_pair_bar_plots(pair_bar_result, temperature, combined_prefix)

        hrex_plots = HREXPlots(
            transition_matrix_png=plot_as_png_fxn(plot_hrex_transition_matrix, diagnostics.transition_matrix),
            swap_acceptance_rates_convergence_png=plot_as_png_fxn(
                plot_hrex_swap_acceptance_rates_convergence, diagnostics.cumulative_swap_acceptance_rates
            ),
            replica_state_distribution_heatmap_png=plot_as_png_fxn(
                plot_hrex_replica_state_distribution_heatmap, diagnostics.cumulative_replica_state_counts
            ),
        )
        return HREXSimulationResult(
            pair_bar_result, plots, trajectories_by_state, md_params, results, diagnostics, hrex_plots
        )

    except Exception as err:
        with open(f"failed_rbfe_result_{combined_prefix}.pkl", "wb") as fh:
            pickle.dump((md_params, err), fh)
        raise err


def estimate_relative_free_energy_bisection_hrex(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    ff: Forcefield,
    host_config: Optional[HostConfig],
    md_params: MDParams = DEFAULT_HREX_PARAMS,
    prefix: str = "",
    lambda_interval: Optional[Tuple[float, float]] = None,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    min_cutoff: Optional[float] = 0.7,
) -> HREXSimulationResult:
    """
    Estimate relative free energy between mol_a and mol_b using Hamiltonian Replica EXchange (HREX) sampling of a
    sequence of intermediate states determined by bisection. Molecules should be aligned to each other and within the
    host environment.

    Parameters
    ----------
    mol_a: Chem.Mol
        initial molecule

    mol_b: Chem.Mol
        target molecule

    core: list of 2-tuples
        atom_mapping of atoms in mol_a into atoms in mol_b

    ff: Forcefield
        Forcefield to be used for the system

    host_config: HostConfig or None
        Configuration for the host system. If None, then the vacuum leg is run.

    md_params: MDParams, optional
        Parameters for the equilibration and production MD. Defaults to :py:const:`timemachine.fe.rbfe.DEFAULT_MD_PARAMS`

    prefix: str, optional
        A prefix to append to figures

    lambda_interval: (float, float) or None, optional
        Minimum and maximum value of lambda for the transformation; typically (0, 1), but sometimes useful to choose
        other values for testing.

    n_windows: int or None, optional
        Number of windows used for interpolating the lambda schedule with additional windows. Defaults to
        `DEFAULT_NUM_WINDOWS` windows.

    min_overlap: float or None, optional
        If not None, terminate bisection early when the BAR overlap between all neighboring pairs of states exceeds this
        value. When given, the final number of windows may be less than or equal to n_windows.
    min_cutoff: float or None, optional
        Throw error if any atom moves more than this distance (nm) after minimization

    Returns
    -------
    HREXSimulationResult
        Collected data from the simulation (see class for storage information).

    """

    if n_windows is None:
        n_windows = DEFAULT_NUM_WINDOWS
    assert n_windows >= 2

    single_topology = SingleTopology(mol_a, mol_b, core, ff)

    lambda_interval = lambda_interval or (0.0, 1.0)
    lambda_min, lambda_max = lambda_interval[0], lambda_interval[1]

    temperature = DEFAULT_TEMP

    host = setup_optimized_host(single_topology, host_config) if host_config else None

    lambda_grid = bisection_lambda_schedule(n_windows, lambda_interval=lambda_interval)
    initial_states = setup_initial_states(
        single_topology, host, temperature, lambda_grid, md_params.seed, min_cutoff=min_cutoff
    )

    make_optimized_initial_state_fn = partial(
        setup_optimized_initial_state,
        single_topology,
        host=host,
        optimized_initial_states=initial_states,
        temperature=temperature,
        seed=md_params.seed,
    )

    # TODO: rename prefix to postfix, or move to beginning of combined_prefix?
    combined_prefix = get_mol_name(mol_a) + "_" + get_mol_name(mol_b) + "_" + prefix

    return estimate_relative_free_energy_bisection_hrex_impl(
        temperature,
        lambda_min,
        lambda_max,
        md_params,
        n_windows,
        make_optimized_initial_state_fn,
        combined_prefix,
        min_overlap,
    )


def run_vacuum(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    _,
    md_params: MDParams = DEFAULT_HREX_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    min_cutoff: Optional[float] = None,
):
    if md_params is not None and md_params.local_steps > 0:
        md_params = replace(md_params, local_steps=0)
        warnings.warn("Vacuum simulations don't support local steps, will use all global steps")
    if md_params is not None and md_params.water_sampling_params is not None:
        md_params = replace(md_params, water_sampling_params=None)
        warnings.warn("Vacuum simulations don't support water sampling, disabling")
    # min_cutoff defaults to None since there is no environment to prevent conformational changes in the ligand
    return estimate_relative_free_energy_bisection_or_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        md_params=md_params,
        host_config=None,
        prefix="vacuum",
        n_windows=n_windows,
        min_overlap=min_overlap,
        min_cutoff=min_cutoff,
    )


def run_solvent(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    _,
    md_params: MDParams = DEFAULT_HREX_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    min_cutoff: Optional[float] = None,
):
    if md_params is not None and md_params.water_sampling_params is not None:
        md_params = replace(md_params, water_sampling_params=None)
        warnings.warn("Solvent simulations don't benefit from water sampling, disabling")
    box_width = 4.0
    solvent_sys, solvent_conf, solvent_box, solvent_top = builders.build_water_system(
        box_width, forcefield.water_ff, mols=[mol_a, mol_b]
    )
    solvent_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    solvent_host_config = HostConfig(solvent_sys, solvent_conf, solvent_box, solvent_conf.shape[0], solvent_top)
    # min_cutoff defaults to None since the original poses tend to come from posing in a complex and
    # in solvent the molecules may adopt significantly different poses
    solvent_res = estimate_relative_free_energy_bisection_or_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        solvent_host_config,
        md_params=md_params,
        prefix="solvent",
        n_windows=n_windows,
        min_overlap=min_overlap,
        min_cutoff=min_cutoff,
    )
    return solvent_res, solvent_top, solvent_host_config


def run_complex(
    mol_a: Chem.rdchem.Mol,
    mol_b: Chem.rdchem.Mol,
    core: NDArray,
    forcefield: Forcefield,
    protein: Union[app.PDBFile, str],
    md_params: MDParams = DEFAULT_HREX_PARAMS,
    n_windows: Optional[int] = None,
    min_overlap: Optional[float] = None,
    min_cutoff: Optional[float] = 0.7,
):
    complex_sys, complex_conf, complex_box, complex_top, nwa = builders.build_protein_system(
        protein, forcefield.protein_ff, forcefield.water_ff, mols=[mol_a, mol_b]
    )
    complex_box += np.diag([0.1, 0.1, 0.1])  # remove any possible clashes, deboggle later
    complex_host_config = HostConfig(complex_sys, complex_conf, complex_box, nwa, complex_top)
    complex_res = estimate_relative_free_energy_bisection_or_hrex(
        mol_a,
        mol_b,
        core,
        forcefield,
        complex_host_config,
        prefix="complex",
        md_params=md_params,
        n_windows=n_windows,
        min_overlap=min_overlap,
        min_cutoff=min_cutoff,
    )
    return complex_res, complex_top, complex_host_config
