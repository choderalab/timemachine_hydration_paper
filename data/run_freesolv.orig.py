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

# for prefactor computation
from timemachine.potentials.nonbonded import coulomb_prefactors_on_traj, DEFAULT_CHUNK_SIZE
from timemachine.fe.free_energy import (
    HostConfig,
    HREXParams,
    HREXPlots,
    InitialState,
    MDParams,
    SimulationResult,
    Trajectory,
    make_pair_bar_plots,
    run_sims_bisection,
    run_sims_hrex,
    run_sims_sequential,
)

# esp stuff
import torch
import espaloma as esp

# define or load a molecule of interest via the Open Force Field toolkit
from openff.toolkit.topology import Molecule

esp_model = esp.get_model("latest")

def ah_coulomb_prefactors_from_traj_state(initial_state: InitialState, trajectory: Trajectory, chunk_size=DEFAULT_CHUNK_SIZE):
    """this is similar to `coulomb_prefactors_from_traj_state`, except extracts the appropriate nbf from 
    bps generated from `AbsoluteFreeEnergy`"""
    ligand_idxs = initial_state.ligand_idxs
    summed_bp = initial_state.potentials[-1] # by inspection, last force is a summed force of 3 nonbonded potentials
    water_nb_potential = summed_bp.potential.potentials[0]
    params = summed_bp.potential.params_init[0]
    charges = params[:,0]
    ligand_charges = charges[ligand_idxs]
    assert np.allclose(ligand_charges, 0.), f"ligand charges: {ligand_charges}" # all ligand idx charges should be zero
    env_idxs = np.array([i for i in range(params.shape[0]) if i not in ligand_idxs], dtype = ligand_idxs.dtype)
    beta, cutoff = water_nb_potential.beta, water_nb_potential.cutoff
    frames_as_arr = np.concatenate([frame[np.newaxis, ...] for frame in trajectory.frames], axis=0)
    boxes_as_arr = np.concatenate([box[np.newaxis, ...] for box in trajectory.boxes], axis=0)
    prefactors = coulomb_prefactors_on_traj(traj = frames_as_arr, 
                                        boxes = boxes_as_arr, 
                                        charges = charges, 
                                        ligand_indices = ligand_idxs, 
                                        env_indices = env_idxs, 
                                        beta=beta, 
                                        cutoff=cutoff, 
                                        chunk_size=chunk_size)
    return prefactors 
    
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
    solvent_res, solvent_top, solvent_host_config = ah.run_solvent(charged_mol, forcefield, None, md_params, n_windows=48)
    print(f"computing prefactors...")
    prefactors0 = ah_coulomb_prefactors_from_traj_state(solvent_res.final_result.initial_states[0], solvent_res.trajectories[0]) # coupled
    prefactors1 = ah_coulomb_prefactors_from_traj_state(solvent_res.final_result.initial_states[-1], solvent_res.trajectories[-1]) # decoupled
    combined_prefix = get_mol_name(mol)
    print(f"cleaning up...")
    lig_frames, boxes = cleanup(solvent_res, 'ligand_only', False)
    print('saving...')
    with open(f"success_ahfe_result_{combined_prefix}.pkl", "wb") as fh:
        pickle.dump((solvent_res, prefactors0, prefactors1), fh)

if __name__ == "__main__":
    read_from_args()