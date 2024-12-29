#!/usr/bin/env python
import argparse
import sys
import typing
import dataclasses
import pickle

import numpy as np
from jax import numpy as jnp
from rdkit import Chem
from openmm import unit

from timemachine.ff.handlers.bonded import annotate_mol_sys_torsions
from timemachine.ff import make_mol_omm_sys
from timemachine.datasets.utils import fetch_freesolv
from timemachine.fe import absolute_hydration as ah
from timemachine.fe.utils import get_mol_name, get_romol_conf
from timemachine.fe.rbfe import cleanup
from timemachine.fe import refitting as ref

import torch
import espaloma as esp
from openff.toolkit.topology import Molecule

esp_model = esp.get_model("latest")
agg_freesolv_filepath = '/data1/choderaj/rufad/tm/agg_freesolv_data.pkl'
num_pcs, retrieve_by_descent, train_fraction, use_ml = 100, False, 0.75, False # hardcode this as our template optimized params to use.
use_ESS = True
refit_data_pkl = f"/data1/choderaj/rufad/tm/{num_pcs}_{retrieve_by_descent}_{train_fraction}_{use_ml}_{use_ESS}.t5.freesolv.pkl"
    
def read_from_args():
    parser = argparse.ArgumentParser(
        description="Estimate ahfe."
    )
    parser.add_argument("--idx", type=int, help="index of freesolv ligand to run ahfe", required=True)
    parser.add_argument("--seed", type=int, help="Random number seed", required=True)

    args = parser.parse_args() # parse args

    mol = fetch_freesolv()[args.idx]

    # charge and annotate
    charge_spec = 'nn'

    # default smirnoff 2.1.0 for torsions/sterics
    charged_mol, omm_sys, tm_ff, molecule_graph = make_mol_omm_sys(mol, charge_spec = charge_spec, esp_model=esp_model)

    # uses original improper torsions...
    annotate_mol_sys_torsions(charged_mol, omm_sys, molecule_graph, tm_ff)

    # update protein_ff; strictly this is unnecessary
    ff_protein_list = ["amber/ff14SB", "amber/phosaa10"]
    forcefield = dataclasses.replace(tm_ff, protein_ff = ff_protein_list)
    
    md_params = dataclasses.replace(ah.DEFAULT_AHFE_MD_PARAMS, n_frames=5000, seed = args.seed)
    
    # after loading the appropriate experiment pickle, we can query the auxiliary data to extract the tm ligand charges
    # without having to call the appropriate functions to regenerate the charges.
    
    # NOTE: the ligand indices do not correspond to those of `fetch_freesolv()` since some ligands failed.
    #       instead, the mol name will be queried and correspond to the appropriate index.
    
    # first, load the f"agg_freesolv_data.pkl"
    agg_freesolv_data = ref.load_pkl_data(agg_freesolv_filepath)
    (out_names, arr_exp_dGs, arr_calc_dGs, prefactors, ligand_charges, es, ss, hs) = agg_freesolv_data
    num_mols = len(out_names)

    # retrieve the mol name, mol_idx (w.r.t. mol name), and the number of atoms since `tm_charges` is padded
    mol_name = get_mol_name(mol)
    mol_idx = out_names.index(mol_name)
    mol_num_atoms = mol.GetNumAtoms()

    # load train/test data from appropriate experiment.
    data = ref.load_pkl_data(refit_data_pkl)
    [(res, num_callbacks), params, train_loss_auxs, test_loss_auxs, validate_loss_auxs, train_idxs, test_idxs, validate_idxs, cache] = data

    # train
    (tr_ESS, tr_delta_us, tr_orig_calc_dg, 
     tr_reweighted_solv_dg, tr_reweighted_solv_ddg, 
     tr_exp_dg, tr_ligand_tm_charges, tr_orig_es_ss, tr_mod_es_ss) = train_loss_auxs
    
    # test
    (te_ESS, te_delta_us, te_orig_calc_dg, 
     te_reweighted_solv_dg, te_reweighted_solv_ddg, 
     te_exp_dg, te_ligand_tm_charges, te_orig_es_ss, te_mod_es_ss) = test_loss_auxs

    # validate
    (vl_ESS, vl_delta_us, vl_orig_calc_dg, 
     vl_reweighted_solv_dg, vl_reweighted_solv_ddg, 
     vl_exp_dg, vl_ligand_tm_charges, vl_orig_es_ss, vl_mod_es_ss) = validate_loss_auxs

    # try to query idx from train, then test, then validate
    _arr_idx = None
    for set_name, idxs, charge_params in zip(
        ['train', 'test', 'validate'], 
        [train_idxs, test_idxs, validate_idxs],
        [tr_ligand_tm_charges, te_ligand_tm_charges, vl_ligand_tm_charges]):
        arr_idxs = np.where(idxs == mol_idx)[0]
        if len(arr_idxs) != 0:
            _arr_idx = arr_idxs[0]
            tm_charges = charge_params[_arr_idx][:mol_num_atoms] # because these are padded
            break
        else:
            continue # not the right set

    # detect and omit possible mismatch of padding.
    assert not np.isclose(tm_charges[-1], 0)
    assert np.isclose(charge_params[_arr_idx][mol_num_atoms], 0)

    if _arr_idx is None: # this is a problem if it happens.
        raise Exception(f"not found anywhere...")
    
    solvent_res, solvent_top, solvent_host_config = ah.run_solvent(
        charged_mol, forcefield, None, md_params, n_windows=48, ligand_tm_ixn_charges = tm_charges)
    
    print(f"computing prefactors...")
    prefactors0 = ah.ah_coulomb_prefactors_from_traj_state(solvent_res.final_result.initial_states[0], solvent_res.trajectories[0]) # coupled
    prefactors1 = ah.ah_coulomb_prefactors_from_traj_state(solvent_res.final_result.initial_states[-1], solvent_res.trajectories[-1]) # decoupled
    combined_prefix = get_mol_name(mol)
    print(f"cleaning up...")
    lig_frames, boxes = cleanup(solvent_res, 'ligand_only', False)
    print('saving...')
    with open(f"success_ahfe_result_{combined_prefix}_ixn.t5.pkl", "wb") as fh:
        pickle.dump((solvent_res, prefactors0, prefactors1), fh)

if __name__ == "__main__":
    read_from_args()