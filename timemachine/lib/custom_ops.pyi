from typing import List, Optional

import numpy

class BoundPotential:
    def __init__(self, potential: Potential, params: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def execute(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64]) -> tuple: ...
    def execute_fixed(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.uint64]: ...
    def get_potential(self) -> Potential: ...
    def size(self) -> int: ...

class CentroidRestraint_f32(Potential):
    def __init__(self, group_a_idxs: numpy.typing.NDArray[numpy.int32], group_b_idxs: numpy.typing.NDArray[numpy.int32], kb: float, b0: float) -> None: ...

class CentroidRestraint_f64(Potential):
    def __init__(self, group_a_idxs: numpy.typing.NDArray[numpy.int32], group_b_idxs: numpy.typing.NDArray[numpy.int32], kb: float, b0: float) -> None: ...

class ChiralAtomRestraint_f32(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralAtomRestraint_f64(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralBondRestraint_f32(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32], signs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class ChiralBondRestraint_f64(Potential):
    def __init__(self, idxs: numpy.typing.NDArray[numpy.int32], signs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class Context:
    def __init__(self, x0: numpy.typing.NDArray[numpy.float64], v0: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], integrator: Integrator, bps: List[BoundPotential], barostat: Optional[MonteCarloBarostat] = ...) -> None: ...
    def finalize(self) -> None: ...
    def get_box(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def get_v_t(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def get_x_t(self) -> numpy.typing.NDArray[numpy.float64]: ...
    def initialize(self) -> None: ...
    def multiple_steps(self, n_steps: int, store_x_interval: int = ...) -> tuple: ...
    def multiple_steps_U(self, n_steps: int, store_u_interval: int, store_x_interval: int) -> tuple: ...
    def multiple_steps_local(self, n_steps: int, local_idxs: numpy.typing.NDArray[numpy.int32], burn_in: int = ..., store_x_interval: int = ..., radius: float = ..., k: float = ..., seed: int = ...) -> tuple: ...
    def multiple_steps_local_selection(self, n_steps: int, reference_idx: int, selection_idxs: numpy.typing.NDArray[numpy.int32], burn_in: int = ..., store_x_interval: int = ..., radius: float = ..., k: float = ...) -> tuple: ...
    def set_box(self, box: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def set_v_t(self, velocities: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def set_x_t(self, coords: numpy.typing.NDArray[numpy.float64]) -> None: ...
    def setup_local_md(self, temperature: float, freeze_reference: bool) -> None: ...
    def step(self) -> None: ...

class FanoutSummedPotential(Potential):
    def __init__(self, potentials: List[Potential], parallel: bool = ...) -> None: ...
    def get_potentials(self) -> List[Potential]: ...

class FlatBottomBond_f32(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class FlatBottomBond_f64(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicAngleStable_f32(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicAngleStable_f64(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicAngle_f32(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicAngle_f64(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicBond_f32(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class HarmonicBond_f64(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class Integrator:
    def __init__(self, *args, **kwargs) -> None: ...

class LangevinIntegrator(Integrator):
    def __init__(self, masses: numpy.typing.NDArray[numpy.float64], temperature: float, dt: float, friction: float, seed: int) -> None: ...

class LogFlatBottomBond_f32(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32], beta: float) -> None: ...

class LogFlatBottomBond_f64(Potential):
    def __init__(self, bond_idxs: numpy.typing.NDArray[numpy.int32], beta: float) -> None: ...

class MonteCarloBarostat:
    def __init__(self, N: int, pressure: float, temperature: float, group_idxs: List[List[int]], frequency: int, bps, seed: int) -> None: ...
    def get_interval(self) -> int: ...
    def set_interval(self, interval: int) -> None: ...
    def set_pressure(self, pressure: float) -> None: ...

class Neighborlist_f32:
    def __init__(self, N: int) -> None: ...
    def compute_block_bounds(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], block_size: int) -> tuple: ...
    def get_nblist(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], cutoff: float) -> List[List[int]]: ...
    def reset_row_idxs(self) -> None: ...
    def resize(self, size: int) -> None: ...
    def set_row_idxs(self, idxs: numpy.typing.NDArray[numpy.uint32]) -> None: ...

class Neighborlist_f64:
    def __init__(self, N: int) -> None: ...
    def compute_block_bounds(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], block_size: int) -> tuple: ...
    def get_nblist(self, coords: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], cutoff: float) -> List[List[int]]: ...
    def reset_row_idxs(self) -> None: ...
    def resize(self, size: int) -> None: ...
    def set_row_idxs(self, idxs: numpy.typing.NDArray[numpy.uint32]) -> None: ...

class NonbondedAllPairs_f32(Potential):
    def __init__(self, num_atoms: int, beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ..., disable_hilbert_sort: bool = ..., nblist_padding: float = ...) -> None: ...
    def get_atom_idxs(self) -> List[int]: ...
    def get_num_atom_idxs(self) -> int: ...
    def set_atom_idxs(self, atom_idxs: List[int]) -> None: ...

class NonbondedAllPairs_f64(Potential):
    def __init__(self, num_atoms: int, beta: float, cutoff: float, atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ..., disable_hilbert_sort: bool = ..., nblist_padding: float = ...) -> None: ...
    def get_atom_idxs(self) -> List[int]: ...
    def get_num_atom_idxs(self) -> int: ...
    def set_atom_idxs(self, atom_idxs: List[int]) -> None: ...

class NonbondedInteractionGroup_f32(Potential):
    def __init__(self, num_atoms: int, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, col_atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ..., disable_hilbert_sort: bool = ..., nblist_padding: float = ...) -> None: ...
    def set_atom_idxs(self, row_atom_idxs: List[int], col_atom_idxs: List[int]) -> None: ...

class NonbondedInteractionGroup_f64(Potential):
    def __init__(self, num_atoms: int, row_atom_idxs_i: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float, col_atom_idxs_i: Optional[numpy.typing.NDArray[numpy.int32]] = ..., disable_hilbert_sort: bool = ..., nblist_padding: float = ...) -> None: ...
    def set_atom_idxs(self, row_atom_idxs: List[int], col_atom_idxs: List[int]) -> None: ...

class NonbondedPairListPrecomputed_f32(Potential):
    def __init__(self, pair_idxs: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairListPrecomputed_f64(Potential):
    def __init__(self, pair_idxs: numpy.typing.NDArray[numpy.int32], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f32_negated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class NonbondedPairList_f64_negated(Potential):
    def __init__(self, pair_idxs_i: numpy.typing.NDArray[numpy.int32], scales_i: numpy.typing.NDArray[numpy.float64], beta: float, cutoff: float) -> None: ...

class PeriodicTorsion_f32(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class PeriodicTorsion_f64(Potential):
    def __init__(self, angle_idxs: numpy.typing.NDArray[numpy.int32]) -> None: ...

class Potential:
    def __init__(self, *args, **kwargs) -> None: ...
    def execute(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64]) -> tuple: ...
    def execute_du_dx(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]: ...
    def execute_selective(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], box: numpy.typing.NDArray[numpy.float64], compute_du_dx: bool, compute_du_dp: bool, compute_u: bool) -> tuple: ...
    def execute_selective_batch(self, coords: numpy.typing.NDArray[numpy.float64], params: numpy.typing.NDArray[numpy.float64], boxes: numpy.typing.NDArray[numpy.float64], compute_du_dx: bool, compute_du_dp: bool, compute_u: bool) -> tuple: ...

class SummedPotential(Potential):
    def __init__(self, potentials: List[Potential], params_sizes: List[int], parallel: bool = ...) -> None: ...
    def get_potentials(self) -> List[Potential]: ...

class VelocityVerletIntegrator(Integrator):
    def __init__(self, dt: float, cbs: numpy.typing.NDArray[numpy.float64]) -> None: ...

def cuda_device_reset() -> None: ...
def rmsd_align(x1: numpy.typing.NDArray[numpy.float64], x2: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]: ...
