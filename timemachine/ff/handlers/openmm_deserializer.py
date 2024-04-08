from collections import defaultdict
from typing import DefaultDict, List, Tuple, Iterable

import numpy as np
import openmm as omm
from openmm import unit

from timemachine import constants, potentials

ORDERED_FORCES = ["HarmonicBond", "HarmonicAngle", "PeriodicTorsion", "Nonbonded"]


def value(quantity):
    return quantity.value_in_unit_system(unit.md_unit_system)

def deserialize_nonbonded_force(force, N):
    num_atoms = force.getNumParticles()

    charge_params_ = []
    lj_params_ = []

    for a_idx in range(num_atoms):
        charge, sig, eps = force.getParticleParameters(a_idx)
        charge = value(charge) * np.sqrt(constants.ONE_4PI_EPS0)

        sig = value(sig)
        eps = value(eps)

        # increment eps by 1e-3 if we have eps==0 to avoid a singularity in parameter derivatives
        # override default amber types

        # this doesn't work for water!
        # if eps == 0:
        # print("Warning: overriding eps by 1e-3 to avoid a singularity")
        # eps += 1e-3

        # charge_params.append(charge_idx)
        charge_params_.append(charge)
        lj_params_.append((sig, eps))

    charge_params = np.array(charge_params_, dtype=np.float64)

    # print("Protein net charge:", np.sum(np.array(global_params)[charge_param_idxs]))
    lj_params = np.array(lj_params_, dtype=np.float64)

    # 1 here means we fully remove the interaction
    # 1-2, 1-3
    # scale_full = insert_parameters(1.0, 20)

    # 1-4, remove half of the interaction
    # scale_half = insert_parameters(0.5, 21)

    exclusion_idxs_ = []
    scale_factors_ = []

    all_sig = lj_params[:, 0]
    all_eps = lj_params[:, 1]

    # validate exclusions/exceptions to make sure they make sense
    for a_idx in range(force.getNumExceptions()):
        # tbd charge scale factors
        src, dst, new_cp, new_sig, new_eps = force.getExceptionParameters(a_idx)
        new_sig = value(new_sig)
        new_eps = value(new_eps)

        src_sig = all_sig[src]
        dst_sig = all_sig[dst]

        src_eps = all_eps[src]
        dst_eps = all_eps[dst]
        expected_sig = (src_sig + dst_sig) / 2
        expected_eps = np.sqrt(src_eps * dst_eps)

        exclusion_idxs_.append([src, dst])

        # sanity check this (expected_eps can be zero), redo this thing

        # the lj_scale factor measures how much we *remove*
        if expected_eps == 0:
            if new_eps == 0:
                lj_scale_factor = 1
            else:
                raise RuntimeError("Divide by zero in epsilon calculation")
        else:
            lj_scale_factor = 1 - new_eps / expected_eps

        scale_factors_.append(lj_scale_factor)

        # tbd fix charge_scale_factors using new_cp
        if new_eps != 0:
            np.testing.assert_almost_equal(expected_sig, new_sig)

    exclusion_idxs = np.array(exclusion_idxs_, dtype=np.int32)

    # cutoff = 1000.0

    nb_params = np.concatenate(
        [
            np.expand_dims(charge_params, axis=1),
            lj_params,
            np.zeros((N, 1)),  # 4D coordinates
        ],
        axis=1,
    )

    # optimizations
    nb_params[:, 1] = nb_params[:, 1] / 2
    nb_params[:, 2] = np.sqrt(nb_params[:, 2])

    beta = 2.0  # erfc correction

    # use the same scale factors for electrostatics and lj
    scale_factors = np.stack([scale_factors_, scale_factors_], axis=1)

    return nb_params, exclusion_idxs, beta, scale_factors

def idxs_params_from_hb(forces: Iterable[omm.HarmonicBondForce], 
                        **unused_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    bond_idxs, bond_params = [], []
    for force in forces:
        for b_idx in range(force.getNumBonds()):
            src_idx, dst_idx, length, k = force.getBondParameters(b_idx)
            length = value(length)
            k = value(k)
            bond_idxs.append(np.array([src_idx, dst_idx], dtype=np.int32))
            bond_params.append(np.array([k, length], dtype=np.float64))
    
    if len(bond_idxs) == 0:
        bond_idxs = np.zeros((0, 2), dtype=np.int32)
        bond_params = np.zeros((0, 2), dtype=np.float64)

    return np.array(bond_idxs, dtype=np.int32), np.array(bond_params, dtype=np.float64)

def idxs_params_from_ha(forces: Iterable[omm.HarmonicAngleForce],
                       **unused_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    angle_idxs, angle_params = [], []
    for force in forces:
        for a_idx in range(force.getNumAngles()):
            src_idx, mid_idx, dst_idx, angle, k = force.getAngleParameters(a_idx)
            angle = value(angle)
            k = value(k)
            angle_idxs.append(np.array([src_idx, mid_idx, dst_idx], dtype=np.int32))
            angle_params.append(np.array([k, angle], dtype=np.float64))
    
    if len(angle_idxs) == 0:
        angle_idxs = np.zeros((0, 3), dtype=np.int32)
        angle_params = np.zeros((0, 2), dtype=np.float64)

    return np.array(angle_idxs, dtype=np.int32), np.array(angle_params, dtype=np.float64)

def idxs_params_from_t(
        forces: Iterable[omm.PeriodicTorsionForce],
        **unused_kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """return the proper(improper) torsion indices and parameters"""
    torsion_idxs, torsion_params = [], []
    for force in forces:
        for t_idx in range(force.getNumTorsions()):
            a_idx, b_idx, c_idx, d_idx, period, phase, k = force.getTorsionParameters(t_idx)
            phase = value(phase)
            k = value(k)
            torsion_idxs_ = np.array([a_idx, b_idx, c_idx, d_idx], dtype=np.int32)
            torsion_idxs_ = torsion_idxs_ if a_idx < d_idx else torsion_idxs_[::-1]
            torsion_params_ = np.array([k, phase, period], dtype=np.float64)
            torsion_idxs.append(torsion_idxs_)
            torsion_params.append(torsion_params_)
    
    if len(torsion_idxs) == 0:
        torsion_idxs = np.zeros((0, 4), dtype=np.int32)
        torsion_params = np.zeros((0, 3), dtype=np.float64)
                
    return np.array(torsion_idxs, dtype=np.int32), np.array(torsion_params, dtype=np.float64)
                          

def deserialize_system(
        system: omm.System, cutoff: float) -> Tuple[List[potentials.BoundPotential], List[float]]:
    """
    Deserialize an OpenMM XML file

    Parameters
    ----------
    system: omm.System
        A system object to be deserialized
    cutoff: float
        Nonbonded cutoff, in nm

    Returns
    -------
    list of lib.Potential, masses

    Note: We add a small epsilon (1e-3) to all zero eps values to prevent
    a singularity from occurring in the lennard jones derivatives

    """

    masses = []

    for p in range(system.getNumParticles()):
        masses.append(value(system.getParticleMass(p)))

    N = len(masses)

    # this should not be a dict since we may have more than one instance of a given
    # force.

    bps_dict: DefaultDict[str, List[potentials.BoundPotential]] = defaultdict(list)

    bond_forces = [f for f in system.getForces() if f.__class__.__name__ == 'HarmonicBondForce']
    bond_idxs, bond_params = idxs_params_from_hb(bond_forces)
    bps_dict["HarmonicBond"].append(potentials.HarmonicBond(bond_idxs).bind(bond_params))

    angle_forces = [f for f in system.getForces() if f.__class__.__name__ == 'HarmonicAngleForce']
    angle_idxs, angle_params = idxs_params_from_ha(angle_forces)
    bps_dict["HarmonicAngle"].append(potentials.HarmonicAngle(angle_idxs).bind(angle_params))

    torsion_forces = [f for f in system.getForces() if f.__class__.__name__ == 'PeriodicTorsionForce']
    torsion_idxs, torsion_params = idxs_params_from_t(torsion_forces)
    bps_dict["PeriodicTorsion"].append(potentials.PeriodicTorsion(torsion_idxs).bind(torsion_params))

    for force in system.getForces():
        if isinstance(force, omm.NonbondedForce):
            nb_params, exclusion_idxs, beta, scale_factors = deserialize_nonbonded_force(force, N)
            bps_dict["Nonbonded"].append(
                potentials.Nonbonded(N, exclusion_idxs, scale_factors, beta, cutoff).bind(nb_params)
            )
    
            # nrg_fns.append(('Exclusions', (exclusion_idxs, scale_factors, es_scale_factors)))

    # ugh, ... various parts of our code assume the bps are in a certain order
    # so put them back in that order here
    bps = []
    for k in ORDERED_FORCES:
        if bps_dict.get(k):
            bps.extend(bps_dict[k])

    return bps, masses