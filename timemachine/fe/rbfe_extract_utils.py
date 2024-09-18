"""utilities to extract and process rbfe data for refitting"""
"""utilities to refit electrostatic parameters"""
import pickle
import argparse
import os
import glob
import torch
import jax
import numpy as np
import typing
import functools
import itertools
from tqdm import tqdm, trange
import scipy
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from flax import linen as nn
import espaloma as esp
from openff.toolkit.topology import Molecule

# specific tm stuff
from timemachine.constants import BOLTZ, DEFAULT_TEMP, KCAL_TO_KJ, ONE_4PI_EPS0
from timemachine.fe.single_topology import SingleTopology
from timemachine.fe.free_energy import SimulationResult
from timemachine.fe.mle import infer_node_vals_and_errs
from timemachine.ff import Forcefield
from timemachine.potentials import NonbondedPairListPrecomputed
from timemachine.ff import make_mol_omm_sys
from timemachine.fe.refitting import load_pkl_data, es_ss_qs_hs_from_mol_graph, perturb_charges


# constants
kBT = BOLTZ * DEFAULT_TEMP
BETA = 1. / kBT
EMBED_DIM = 512 # this is default for esp model


# extract experimental data utils
def read_sddata(filename, name_sdtag=None):
    """Read molecular data from an SD file."""
    read_molecules = Molecule.from_file(filename, allow_undefined_stereo=True)
    if name_sdtag:
        molecules = {molecule.properties[name_sdtag]: molecule.properties for molecule in read_molecules}
    else:
        molecules = {molecule.name: molecule.properties for molecule in read_molecules}
    return molecules


def get_experimental_estimates(
    sdfile: str, pic50_sdtag: str="pChEMBL Value",
    pic50_error_sdtag: str="pChEMBL Error"):
    """Compute experimental estimates of binding free energies and errors from an SD file. from ivan pulido."""
    ligands_data = read_sddata(sdfile)
    results = {}
    for name, metadata in ligands_data.items():
        # Compute experimental affinity in kcal/mol
        ligand_name = name
        dg_exp = - np.log(10) * float(metadata[pic50_sdtag]) * kBT / KCAL_TO_KJ
        try:
            ddg_exp = - np.log(10) * float(metadata[pic50_error_sdtag]) * kBT / KCAL_TO_KJ
        except KeyError:
            warnings.warn(f"No SDtag '{pic50_error_sdtag}' for pIC50 error found, fixing to default value of 0.3 kcal/mol")
            ddg_exp = 0.3 # estimate
        results[name] = {"dg_exp": dg_exp, "ddg_exp": ddg_exp}

    return results


# standard plotting util
def plot_standard_with_errors(
    calc_dGs: jax.Array, exp_dGs: jax.Array, calc_dG_err: jax.Array, exp_dG_err: jax.Array, 
    xlabel = 'calc dG [kcal/mol]',
    ylabel = 'exp dG [kcal/mol]',
    title = '',
    do_linregress: bool=True):
    all_concat = np.concatenate([calc_dGs, exp_dGs])
    minval, maxval = np.min(all_concat), np.max(all_concat)
    x_ref = np.linspace(minval - 1, maxval + 1, 1000)
    y_ref = x_ref
    y_ref_down = y_ref - 1.
    y_ref_up = y_ref + 1.
    
    rmse = np.sqrt(np.mean(np.square(calc_dGs - exp_dGs))) 
    plt.errorbar(calc_dGs, exp_dGs, xerr = calc_dG_err, yerr = exp_dG_err, ls='none', marker='o', label=f"RMSE: {rmse}")
    plt.plot(x_ref, y_ref, color='k')
    plt.fill_between(x_ref, y_ref_down, y_ref_up, alpha=0.25, label=f"+/- 1 kcal/mol")

    if do_linregress:
        res = scipy.stats.linregress(calc_dGs, exp_dGs)
        fit_y = res.slope * x_ref + res.intercept
        r2 = res.rvalue**2
        plt.plot(x_ref, fit_y, label=f"best fit line: R^2 = {r2}")
        
    plt.xlim(minval - 1, maxval + 1)
    plt.ylim(minval - 1, maxval + 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()


# handler classes
class NonbondedPairlistPrecomputedHandler:
    """handler that will do the following:
    1. take two edge `mol`s with a `core` and a `forcefield` that will construct an 
        `intermediate` state at lambdas 0,1
    2. will make internal consistency test to ensure that the `params[:,0]` are consistent 
        with charges of the `mols` and evaluate the `rescale mask` by dividing by `q_ij`
    """
    def __init__(
        self, 
        mol_a, 
        mol_b, 
        core, 
        forcefield, 
        es_ss_a: jax.Array, # [2, n_atoms] 
        es_ss_b: jax.Array, # [2, n_atoms]
    ):
        """see first 4 inputs to 
        https://github.com/proteneer/timemachine/\
        blob/c63bc931cc57e89f9e18d4174bf212ab4a92209b/timemachine/fe/single_topology.py#L1053;
        they are the same as `timemachine.fe.single_topology.SingleTopology`;
        `es_ss_{a/b}` are the electronegativity and hardness mol_{a/b} atoms."""
        self.st = SingleTopology(mol_a, mol_b, core, forcefield)
        self.es_ss_a = es_ss_a
        self.es_ss_b = es_ss_b
        
        self.orig_qs = [self.st.ff.q_handle.parameterize(mol).astype(jnp.float32) for mol in [self.st.mol_a, self.st.mol_b]]
        self.total_charge_a = jnp.sum(self.orig_qs[0] / jnp.sqrt(ONE_4PI_EPS0))
        self.total_charge_b = jnp.sum(self.orig_qs[1] / jnp.sqrt(ONE_4PI_EPS0))
        self.num_hybrid_atoms = self.st.get_num_atoms()

    def test_es_ss_to_orig_charges(self, rtol: float=1e-3):
        """internal consistency test to ensure that the original charges of mols match the 
        embedding parameterization"""
        for mol, params in zip([self.st.mol_a, self.st.mol_b],[self.es_ss_a, self.es_ss_b]):
            rdkit_charges = np.array([float(atom.GetProp('PartialCharge')) for atom in mol.GetAtoms()])
            orig_charges_from_ff = self.st.ff.q_handle.parameterize(mol)
            orig_charges_from_params, _ = perturb_charges(
                particle_elecs = params[0], 
                particle_hards = params[1],
                particle_elec_perts = params[0] * 0.,
                particle_hard_perts = params[1] * 0.,
                total_charge = rdkit_charges.sum())
            try:
                assert np.allclose(orig_charges_from_ff, orig_charges_from_params * np.sqrt(ONE_4PI_EPS0), rtol=rtol)  
            except Exception as e:
                print(f"Raised charge consistency error: {e}")
                abs_resids = np.abs(orig_charges_from_ff - orig_charges_from_params * np.sqrt(ONE_4PI_EPS0))
                print(f"max residual: {np.max(abs_resids)}")


class RbfeDataHandler:
    """
    a handler that will extract/store ligand data, construct/store appropraite potentials
    to query for a rbfe free energy refitter.
    All reported free energies are in KCAL/MOL because that's canonical.
    """
    data_attr_template = [
        'mol_a', 'mol_b', 'edge_metadata', 'core', 
        'solvent_res', 'solvent_top', 'complex_res', 'complex_top',
        'complex_ligand_frames', 'complex_boxes',
        'solvent_ligand_frames', 'solvent_boxes',
        'complex_prefactors', 'solvent_prefactors'
    ]

    ligand_potential_handlers = {'NonbondedPairListPrecomputed': NonbondedPairlistPrecomputedHandler}
    
    def __init__(
        self,
        ligand_sdf_filepath: str = SDF_PATH,
        # query appropriate data 
        datapath_dir: str = DATA_PATH,# directory path of `success*.pkl` objects from timemachine  
        ligand_potential_names: typing.List[str] = ['NonbondedPairListPrecomputed'], # (intra)ligand potentials; default is just elec/steric
        esp_model: typing.Any = esp.get_model("latest"),
        **unused_kwargs):
        
        self.datapath_dir = datapath_dir
        self.ligand_potential_names = ligand_potential_names
        self.data_files = self.query_datafiles()
        self.data_filepaths = [os.path.join(self.datapath_dir, file) for file in self.data_files]
        self.n_edges = len(self.data_filepaths)
        self.esp_model = esp_model

        # attrs to be populated by `extract_phase_state_data`
        self.edge_mols = [] # entries are typing.Tuple[Chem.Mol, Chem.Mol]
        self.edge_cores = [] # typing.List[np.typing.NDArray]
        self.edge_mol_names = [] # entries are typing.List[str, str]
        self.hs = [] # entries of embeddings by mol pair

        # these entries are of shape [n_edges, n_endstates (2), n_frames, n_buffered_atoms, ...]
        self.frames = {'solvent': [], 'complex': []}
        self.boxes = {'solvent': [], 'complex': []}
        self.prefactors = {'solvent': [], 'complex': []}

        # get exptl free energies
        self.abs_dgs = get_experimental_estimates(ligand_sdf_filepath)

        # NonbondedPairListPrecomputed handlers
        self.nbplpc_handlers = []
        
        self.rel_dgs = {} # takes typing.Tuple[str, str]: {'dg_exp': float, 'ddg_exp': float} (and relative terms)

        # load data pickles sequentially
        self.extract_phase_state_data(self.data_filepaths)

        # prune the `abs_dgs` to only include data that was calculated
        self.prune_abs_dgs()
    

    def prune_abs_dgs(self):
        """remove mol names in `abs_dgs` that are not in `edge_mol_names`"""
        flat_edge_mol_names = set(list(itertools.chain.from_iterable(self.edge_mol_names)))
        out_dict = {}
        for key, val in self.abs_dgs.items():
            if key in flat_edge_mol_names:
                out_dict[key] = val
            else:
                pass
        self.abs_dgs = out_dict

    def query_datafiles(self, **unused_kwargs):
        """return the successful files"""
        success_files = collect_success_filepaths(self.datapath_dir)
        return success_files

    def build_exp_rel_dgs(self):
        """query the `exp_abs_dgs` dict and extract a relative dG rbfe"""
        out_dict = {}
        for (mol_a, mol_b) in edge_mol_names:
            mol_a_dg, mol_a_ddg = self.exp_abs_dgs[mol_a]['dg_exp'], self.exp_abs_dgs[mol_a]['ddg_exp']
            mol_b_dg, mol_b_ddg = self.exp_abs_dgs[mol_b]['dg_exp'], self.exp_abs_dgs[mol_b]['ddg_exp']
            rel_dg_exp = mol_b_dg - mol_a_dg
            rel_ddg_exp = np.sqrt(mol_a_ddg**2 + mol_b_ddg**2)
            out_dict[(mol_a, mol_b)] = {'dg_exp': rel_dg_exp, 'ddg_exp': rel_ddg_exp}
        return out_dict         
        
    def build_data_struct_from_pkl(self, filepath: str, **unused_kwargs) -> typing.Dict[str, typing.Any]:
        """load a pickle, and given its data template, return a dict of {'attr': typing.Any}
        which will be used to set/update attrs"""
        data = load_pkl_data(filepath)
        assert len(data) == len(self.data_attr_template)
        data_dict = {self.data_attr_template[idx]: data[idx] for idx in range(len(data))}
        return data_dict

    def pull_res_and_replica_idx_by_state_by_iter(self, data_dict, phase):
        """extract the `SimulationResult` and its `replica_idx_by_state_by_iter`"""
        res = data_dict[f'{phase}_res']
        replica_idx_by_state_by_iter = jnp.array(
            res.hrex_diagnostics.replica_idx_by_state_by_iter) # [n_frames, n_windows]
        return res, replica_idx_by_state_by_iter

    def pull_frames_boxes_prefactors_by_state(
        self, 
        data_dict: typing.Dict[str, typing.Any],
        phase: str) -> typing.Tuple[jax.Array, jax.Array]:
        res, _ = self.pull_res_and_replica_idx_by_state_by_iter(data_dict, phase)
        all_frames = data_dict[f'{phase}_ligand_frames'] # list of shape [n_windows, n_frames, n_atoms, 3] 
        all_boxes = data_dict[f'{phase}_boxes'] # list of shape [n_windows, n_frames, 3, 3] 
        all_prefactors = data_dict[f'{phase}_prefactors'] # [n_windows, n_frames, n_atoms]

        # concatenate lists into a single array
        concat_all_frames = jnp.concatenate(
            [np.concatenate([frame[np.newaxis, ...] for frame in _frames])[np.newaxis, ...] for _frames in all_frames])
        concat_all_boxes = jnp.concatenate(
            [np.concatenate([frame[np.newaxis, ...] for frame in _frames])[np.newaxis, ...] for _frames in all_boxes])

        # for prefactors, last 2 dims are already an array
        concat_all_prefactors = jnp.concatenate([frame[jnp.newaxis, ...] for frame in all_prefactors])


        return (
            jnp.vstack([concat_all_frames[0][jnp.newaxis, ...], concat_all_frames[-1][jnp.newaxis, ...]]),
            jnp.vstack([concat_all_boxes[0][jnp.newaxis, ...], concat_all_boxes[-1][jnp.newaxis, ...]]),
            jnp.vstack([concat_all_prefactors[0][jnp.newaxis, ...], concat_all_prefactors[-1][jnp.newaxis, ...]]),
        )


    def pull_num_particles(self, res: SimulationResult) -> int:
        """return the number of particles of a result"""
        x0 = res.final_result.initial_states[0].x0
        return x0.shape[0]

    def pull_ligand_idxs(self, res: SimulationResult) -> np.typing.NDArray:
        """return the ligand indices of a result"""
        return res.final_result.initial_states[0].ligand_idxs

    def build_NonbondedPairListPrecomputed_potential_params_fn(
        self,
        data_dict: typing.Dict[str, typing.Any], 
        phase: str,
        build_handler_on_phase: str = 'solvent'
    ):
        """given a `data_dict`, query it by phase and handle the `NonbondedPairListPrecomputed`
        for internal potential builders and parameterization/energy consistency checks.
        Specifically, run a test to make sure energies match between my 
            ad hoc potential/param construction and the existing potential/params; 
            this is the most crucial test for the validity of the intramolecular ligand charge reparameterization.
        """
        mol_a, mol_b = data_dict['mol_a'], data_dict['mol_b']
        core = data_dict['core']

        # build mol graphs
        es_ss_list, hs_list = [], []
        for mol in [mol_a, mol_b]:
            # build ff/mol_graph
            charged_mol, omm_sys, tm_ff, molecule_graph = make_mol_omm_sys(
                mol, charge_spec = 'nn', esp_model=self.esp_model)
            es, ss, qs, hs = es_ss_qs_hs_from_mol_graph(molecule_graph) # gather elec, hardness, qs, embeds
            es_ss = jnp.concatenate([es[jnp.newaxis, ...], 
                                     ss[jnp.newaxis, ...],
                                    ]) # concatenate elecs, hardnesses, hs to 1 arr
            es_ss_list.append(es_ss)
            hs_list.append(hs)

        if phase == build_handler_on_phase: # the handler is not yet built
            handler = NonbondedPairlistPrecomputedHandler(mol_a, mol_b, core, tm_ff, es_ss_list[0], es_ss_list[1])
            handler.test_es_ss_to_orig_charges() # make sure the computed charges from es and ss match original charges for each mol
        else: # the handler is already built; just query that
            handler = self.nbplpc_handlers[-1]

        if phase == build_handler_on_phase: # only add once for each edge
            self.nbplpc_handlers.append(handler)
            self.hs.append(hs_list)


    def calc_rel_dg_ddg(self, data_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, float]:
        """compute the calculated dg, ddg (and by phase) by querying the `SimulationResult`s of a `data_dict`"""
        solvent_res, complex_res = data_dict['solvent_res'], data_dict['complex_res']
        solvent_dg = sum(solvent_res.final_result.dGs)
        solvent_ddg = np.linalg.norm(solvent_res.final_result.dG_errs)
        complex_dg = sum(complex_res.final_result.dGs)
        complex_ddg = np.linalg.norm(complex_res.final_result.dG_errs)
        return {'dg_calc': (complex_dg - solvent_dg) / KCAL_TO_KJ, 
                'ddg_calc': np.linalg.norm(np.array([solvent_ddg, complex_ddg])) / KCAL_TO_KJ,
                'dg_calc_solvent': solvent_dg / KCAL_TO_KJ,
                'dg_calc_complex': complex_dg / KCAL_TO_KJ}

    def exp_rel_dg_ddg(self, mol_a_name: str, mol_b_name: str) -> typing.Dict[str, float]:
        mol_a_dg, mol_a_ddg = self.abs_dgs[mol_a_name]['dg_exp'], self.abs_dgs[mol_a_name]['ddg_exp']
        mol_b_dg, mol_b_ddg = self.abs_dgs[mol_b_name]['dg_exp'], self.abs_dgs[mol_b_name]['ddg_exp']
        dg_exp = mol_b_dg - mol_a_dg
        ddg_exp = np.linalg.norm(np.array([mol_a_ddg, mol_b_ddg]))
        return {'dg_exp': dg_exp, 'ddg_exp': ddg_exp}

    def extract_phase_state_data(
        self, 
        data_paths: typing.List[str],
        **kwargs):
        """iterate over the `data_paths`, construct the following data with corresponding attrs:
        1. `edge_mols`: tuple of mols, 
        2. `cores`: core for each edge
        3. `{phase}_{frames/boxes/prefactors}`: build/reshape frame/box/prefactor data for each phase and endstate
        """
        for data_path in tqdm(data_paths):
            data_dict = self.build_data_struct_from_pkl(data_path)

            # pull mol edges/cores
            self.edge_mols.append([data_dict['mol_a'], data_dict['mol_b']]) # 1
            lig_names = [lig.GetProp('_Name') for lig in self.edge_mols[-1]]
            self.edge_mol_names.append(lig_names)
            self.edge_cores.append(data_dict['core']) # 2

            # pull calculated/exptl dgs, ddgs by querying `SimulationResult`, add to `rel_dgs`
            calc_dgs = self.calc_rel_dg_ddg(data_dict)
            exp_dgs = self.exp_rel_dg_ddg(*tuple(lig_names))
            exp_dgs.update(calc_dgs)
            self.rel_dgs[tuple(lig_names)] = exp_dgs

            for phase in self.frames.keys(): # iterate through phase data
                # pull frames/boxes/prefactors after reshaping; this implicitly iterates over endstate
                frames, boxes, prefactors = self.pull_frames_boxes_prefactors_by_state(
                    data_dict, phase)
                self.frames[phase].append(frames)
                self.boxes[phase].append(boxes)
                self.prefactors[phase].append(prefactors)

                # now build internal `NonbondedPairListPrecomputed` potentials, params, param fns, and run internal consistency
                self.build_NonbondedPairListPrecomputed_potential_params_fn(data_dict, phase)


# handler extraction/postprocessing utils
def get_data_from_abs_rel_dgs(abs_dg_dict, rel_dg_dict):
    """retrieve edge idxs, diffs, stddevs, node idxs, vals, stddevs"""
    # first, retrieve the list of mol names
    mol_names_list = list(abs_dg_dict.keys())

    # then iterate over the rel dg dict to make idxs pairs, edge diffs, edge_stddevs, etc
    edge_idxs, edge_diffs, edge_stddevs = [], [], []
    for mol_name_pair, data_dict in rel_dg_dict.items():
        mol_name_1, mol_name_2 = mol_name_pair
        try:
            mol_idx1, mol_idx2 = mol_names_list.index(mol_name_1), mol_names_list.index(mol_name_2)
            edge_idxs.append([mol_idx1, mol_idx2])
            edge_diffs.append(data_dict['dg_calc'])
            edge_stddevs.append(data_dict['ddg_calc'])
        except Exception as e:
            print(e)

    # now iterate for absolutes
    ref_node_idxs = np.arange(len(mol_names_list))
    ref_node_vals = np.array([abs_dg_dict_val['dg_exp'] for abs_dg_dict_val in abs_dg_dict.values()])
    ref_node_stddevs = np.array([abs_dg_dict_val['ddg_exp'] for abs_dg_dict_val in abs_dg_dict.values()])
    return (
        np.array(edge_idxs), np.array(edge_diffs), np.array(edge_stddevs),
        ref_node_idxs, ref_node_vals, ref_node_stddevs
    )  


def matcher_allclose(list_of_arrs):
    arr0 = list_of_arrs[0]
    matches = [np.allclose(arr0, arr) for arr in list_of_arrs[1:]]
    assert all(matches)
    

def pull_prefactors_tmcharges_es_ss_hs(abs_dg_mol_names, edge_mol_names, prefactors_dict, nbplpc_handlers_list, edge_hs_list):
    N = len(abs_dg_mol_names)
    solv_prefactors_coll, comp_prefactors_coll = [], []
    tm_charges_coll, es_coll, ss_coll, hs_coll = [], [], [], []
    for mol_name_idx, mol_name in enumerate(abs_dg_mol_names):
        _solv_prefactors_coll, _comp_prefactors_coll = [], []
        _tm_charges_coll, _es_coll, _ss_coll, _hs_coll = [], [], [], []
        for edge_mol_name_pair_idx, edge_mol_name_pair in enumerate(edge_mol_names):
            matches = [mol_name == _edge_mol_name for _edge_mol_name in edge_mol_name_pair]
            if True not in matches:
                continue
            endstate = matches.index(True)
            st = nbplpc_handlers_list[edge_mol_name_pair_idx].st
            to_c_idxs = st.a_to_c if endstate == 0 else st.b_to_c
            
            # retrieve prefactors
            solvent_prefactors = prefactors_dict['solvent'][edge_mol_name_pair_idx][endstate][:,to_c_idxs]
            complex_prefactors = prefactors_dict['complex'][edge_mol_name_pair_idx][endstate][:,to_c_idxs]

            # retrieve tm charges, es, ss, hs
            nbplpc_handler = nbplpc_handlers_list[edge_mol_name_pair_idx]
            tm_charges = nbplpc_handler.orig_qs[endstate] # already in tm style
            es = nbplpc_handler.es_ss_a[0] if endstate == 0 else  nbplpc_handler.es_ss_b[0]
            ss = nbplpc_handler.es_ss_a[1] if endstate == 0 else  nbplpc_handler.es_ss_b[1]
            hs = edge_hs_list[edge_mol_name_pair_idx][endstate]

            # collect
            _solv_prefactors_coll.append(solvent_prefactors)
            _comp_prefactors_coll.append(complex_prefactors)
            _tm_charges_coll.append(tm_charges)
            _es_coll.append(es)
            _ss_coll.append(ss)
            _hs_coll.append(hs)
            
        # add prefactors
        solv_prefactors_coll.append(_solv_prefactors_coll)
        comp_prefactors_coll.append(_comp_prefactors_coll)
        # assert singular matches
        matcher_allclose(_tm_charges_coll)
        matcher_allclose(_es_coll)
        matcher_allclose(_ss_coll)
        matcher_allclose(_hs_coll)

        # then append
        tm_charges_coll.append(tm_charges)
        es_coll.append(es)
        ss_coll.append(ss)
        hs_coll.append(hs)

    # just stack the prefactors 
    solv_prefactors_coll = [np.vstack(lst) for lst in solv_prefactors_coll]
    comp_prefactors_coll = [np.vstack(lst) for lst in comp_prefactors_coll]

    return solv_prefactors_coll, comp_prefactors_coll, tm_charges_coll, es_coll, ss_coll, hs_coll 