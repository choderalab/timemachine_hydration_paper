"""utilities to refit electrostatic parameters"""
import pickle
import os
import torch
import jax
import numpy as np
import typing
import functools
import itertools
import scipy
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from flax import linen as nn
import espaloma as esp
from openff.toolkit.topology import Molecule

# specific tm stuff
from timemachine.fe.reweighting import one_sided_exp
from timemachine.md.smc import effective_sample_size
from timemachine.fe.utils import get_mol_name
from timemachine.constants import BOLTZ, DEFAULT_TEMP, KCAL_TO_KJ, ONE_4PI_EPS0

# constants
kBT = BOLTZ * DEFAULT_TEMP
BETA = 1. / kBT
DEFAULT_NN_KEY = jax.random.PRNGKey(2024)
EMBED_DIM = 512 # this is default for esp model
ELECTRONEGATIVITY_PAD = 0.
HARDNESS_PAD = 1e8
EMBED_PAD = -1.



# querying utilities
def load_pkl_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def query_name_dG_dG_err(mol):
    """query name, dG, dG_err from Chem.Mol (freesolv)"""
    dG = float(mol.GetProp('dG'))
    dG_err = float(mol.GetProp('dG_err'))
    name = get_mol_name(mol)
    return name, dG, dG_err


def ligand_charges_from_solvent_res(solvent_res):
    """retrieve the ligand charges from a ahfe `SimulationResult`"""
    ligand_idxs = solvent_res.final_result.initial_states[-1].ligand_idxs
    ligand_nb_params = solvent_res.final_result.initial_states[-1].potentials[-1].potential.params_init[1][ligand_idxs]
    charges = ligand_nb_params[:,0]
    ws = ligand_nb_params[:,-1]
    assert np.allclose(ws, 0.), f"all ws at final endstate should be 0."
    return charges 


def es_ss_qs_hs_from_mol_graph( 
    mol_graph: esp.graphs.graph.Graph):
    """return the electronegativities, hardnesses, charges (elem units), and embeddings
    from an esp Graph"""
    kws = ['e', 's', 'q', 'h']
    outs = []
    for key in kws:
        _data = mol_graph.nodes['n1'].data[key].detach().cpu().numpy().astype(np.float32)
        if key in ['e', 's', 'q']:
            _data = _data.flatten()
        outs.append(np.array(_data))
    return outs


# atom type embedding and projection
def embedding_pca(
    all_hs: jax.Array, # [n_mols, embedding_dim]
) -> typing.Tuple[jax.Array, jax.Array]:
    """do jax PCA on embeddings; returns a 1d array of eigenvals 
    and corresponding col eigenvecs
    """
    m, n = all_hs.shape[::-1]
    X = all_hs.T
    X_bar = np.mean(X, axis=1)
    XlessX_bar = X - X_bar[:,np.newaxis]
    C = jnp.dot(XlessX_bar, XlessX_bar.T) / (n - 1)
    eigenvalues, eigenvectors = np.linalg.eig(C)
    return eigenvalues, eigenvectors


# MLP constructors
def custom_normal_initializer(mean: float = 0.0, stddev: float = 0.05):
    def init(key, shape, dtype=jnp.float64) -> jnp.ndarray:
        return jax.random.normal(key, shape, dtype) * stddev + mean
    return init


class MLP(nn.Module):
    """simple multilayer perceptron"""
    features: int=2 # hidden layer features
    num_layers: int=1 # number of layers
    output_dimension: int=2 # output dimension (e, s)
    nonlinearity: typing.Callable[float, float] = jnp.tanh
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # NOTE: may want to extract `Dense` unused_kwargs out in future.
        for layer in range(self.num_layers):
            x = nn.Dense(
                features = self.features, 
                dtype=jnp.float64, 
                param_dtype=jnp.float64,
                kernel_init=custom_normal_initializer(0., 1e-6))(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            features = self.output_dimension, 
            dtype=jnp.float64, 
            param_dtype=jnp.float64,
            kernel_init=custom_normal_initializer(0., 1e-6)
        )(x)
        return x


# exp reweighting/loss utils
def exp_and_error(delta_us):
    """compute `one_sided_exp` and estimator error"""
    f_mean = one_sided_exp(delta_us) # f_mean = -ln<x>
    T = len(delta_us)
    max_arg = jnp.max(-delta_us)
    x = jnp.exp(-delta_us - max_arg)
    Ex = x.mean()
    dx = jnp.std(x) / jnp.sqrt(T)
    df = dx / Ex
    return f_mean, df

def exp_reweighting_estimator(
    delta_f_base: float,
    delta_us_at_lam0: jax.Array,
    delta_us_at_lamneg1: jax.Array,
    err_delta_f_base: float,
    ) -> float:
    """from a base (reduced) free energy and delta us at lambda 0 (and -1), 
    compute a _perturbed_ Zwanzig-based free energy"""
    df_lam0, df_lam0_err = exp_and_error(delta_us_at_lam0)
    df_lamneg1, df_lamneg1_err = exp_and_error(delta_us_at_lamneg1)
    df = delta_f_base - df_lam0 + df_lamneg1
    ddf = jnp.sqrt(err_delta_f_base**2 + df_lam0_err**2 + df_lamneg1_err**2)
    return df, ddf

def ESS_from_delta_us(delta_us: jax.Array) -> float:
    """compute an effective sample size from delta_us (reduced change in potential energy)"""
    log_weights = -delta_us
    return effective_sample_size(log_weights) / len(log_weights) 


def abs_dg_reweighting_zwanzig(
    pert_us: jax.Array,
    orig_us: jax.Array,
    solv_base_dg: float,
    solv_base_ddg: float,
) -> typing.Tuple[float, float]:
    """compute absolute dg given `base` free energy (reduced) and reduced us; 
    lambda0 perturbation is 0; only perturbation at lambda1"""
    solv_delta_us_lambda1 = pert_us - orig_us
    ESS = ESS_from_delta_us(solv_delta_us_lambda1)
    solvent_reweighting_estimator, solvent_reweighting_error = exp_reweighting_estimator(
        delta_f_base = solv_base_dg,
        delta_us_at_lam0 = 0. * solv_delta_us_lambda1, # lambda 0 is not perturbed
        delta_us_at_lamneg1 = solv_delta_us_lambda1,
        err_delta_f_base = solv_base_ddg)
    return solvent_reweighting_estimator, (ESS, solvent_reweighting_error, solv_delta_us_lambda1)


def dg_l1_loss(r):
    return jnp.linalg.norm(r)

    
def dg_l2_loss(r): 
    return r**2
    

def dg_pseudo_huber_loss(r, delta=1.):
    return delta**2 * ( jnp.sqrt(1 + (r/delta)**2) - 1 )


def abs_dg_zwanzig_loss(pert_us, orig_us, solv_base_dg, solv_base_ddg, exp_dg, loss_fn):
    reweighted_solv_dg, (ESS, reweighted_solv_ddg, delta_us) = abs_dg_reweighting_zwanzig(pert_us, orig_us, solv_base_dg, solv_base_ddg)
    loss = loss_fn(reweighted_solv_dg - exp_dg)
    return loss, (ESS, delta_us, reweighted_solv_dg, reweighted_solv_ddg)


# electrostatic parameterization/energy eval utils
def elec_hardness_parameterizer(
    h: jax.Array, # [m] dim embedding
    params: typing.Union[typing.Dict, jax.Array], # if array: [2,n] params for electronegativity, hardness
    # partial the following parameters to jit compile...
    eigenvecs: typing.Union[jax.Array, None], # [m,n] eigenvectors
    e_scale: float, 
    s_scale: float,
    ml_model: typing.Union[MLP, None],
    ) -> jax.Array:
    """given an atom embedding, params and eigenvecs, 
    return a 2-array of hardness and electronegativity (typically perturbations)"""
    assert h.shape == (512,)
    assert eigenvecs.shape[0] == 512
    projections = jnp.transpose(eigenvecs) @ h[..., jnp.newaxis] if eigenvecs is not None else h # [n,m]x[m,1] -> [n,1]
    if ml_model:    
        assert type(params) == dict
        flat_projs = projections.flatten()
        out = ml_model.apply(params, flat_projs)
    else:
        e = jnp.dot(params[0,:], projections.flatten()) # jnp.dot((n,), (n,)) -> (1,)
        s = jnp.dot(params[1,:], projections.flatten()) # jnp.dot((n,), (n,)) -> (1,)
        out = jnp.array([e, s])
    out = jnp.array([e_scale, s_scale]) * out # scale the output
    return out

    
def singular_U_ligand_env(hybrid_unmasked_qs, prefactors):
    unpadded_hybrid_unmasked_qs = hybrid_unmasked_qs[:len(prefactors)]
    return jnp.dot(unpadded_hybrid_unmasked_qs, prefactors)


def U_ligand_env(
    hybrid_unmasked_qs: jax.Array, # [padded_n_hybrid_atoms], maybe padded,
    batch_prefactors: jax.Array, # [n_frames, n_hybrid_atoms], unmasked)
) -> jax.Array:
    return jax.vmap(singular_U_ligand_env, in_axes=(None,0))(hybrid_unmasked_qs, batch_prefactors)


def perturb_charges(
    particle_elecs: jax.Array, 
    particle_hards: jax.Array,
    particle_elec_perts: jax.Array,
    particle_hard_perts: jax.Array,
    total_charge: int,
    eps: float = 1e-6
    ) -> typing.Tuple[jax.Array]:
    """given some particle electronegativities and hardnesses
    along with perturbations thereof, compute new particle charges (in elementary units)"""
    e = particle_elecs + particle_elec_perts
    s = particle_hards + particle_hard_perts
    s_inv = 1. / (s + eps)
    sum_s_inv = jnp.sum(s_inv)

    # not sure where extra dimension is coming from right now, so will squeeze
    out = jnp.squeeze(s_inv * (-e + (total_charge + jnp.sum(e * s_inv)) / sum_s_inv)) 
    orig_es_ss = jnp.vstack([particle_elecs[jnp.newaxis,...], particle_hards[jnp.newaxis, ...]])
    mod_es_ss = jnp.vstack([e[jnp.newaxis,...], s[jnp.newaxis, ...]])
    return out, (orig_es_ss, mod_es_ss)


def compute_charges(
    hs: jax.Array, # [n_atoms_padded,dim_embed]
    params: typing.Union[typing.Dict, jax.Array], # if array: [2,n] params for electronegativity, hardness
    orig_params: jax.Array,  # [n_atoms, 2 (electronegativity/hardness)
    total_charge: int,
    eps: float,
    eigenvecs: jax.Array, # [n_eigvecs, dim_embed],
    e_scale: float,
    s_scale: float,
    ml_model: MLP) -> typing.Tuple[jax.Array, jax.Array]:
    """compute unperturbed and perturbed charges (in tm units)"""
    # open data
    orig_es, orig_ss = orig_params[:,0], orig_params[:,1]

    # retrieve mask (by matching orig ss large val)
    mask = jnp.where(jnp.isclose(orig_ss, HARDNESS_PAD), 0., 1.)
    
    # compute perturbations
    es_perts = jax.vmap(elec_hardness_parameterizer, in_axes=(0,None,None,None,None,None))(
        hs, params, jnp.transpose(eigenvecs), e_scale, s_scale, ml_model)
    e_perts, s_perts = es_perts[:,0] * mask, es_perts[:,1] * mask
    pert_charges, (orig_es_ss, mod_es_ss) = perturb_charges(orig_es, orig_ss, e_perts, s_perts, total_charge, eps)
    return pert_charges * jnp.sqrt(ONE_4PI_EPS0), (orig_es_ss, mod_es_ss)


def retrieve_es_ss_statistics(handlers):
    """return the standard deviations of """
    all_es, all_ss = [], []
    for handler in handlers:
        es_ss_a, es_ss_b = handler.es_ss_a, handler.es_ss_b
        all_es.append(np.concatenate([es_ss_a[0], es_ss_b[0]]))
        all_ss.append(np.concatenate([es_ss_a[1], es_ss_b[1]]))
    es = np.concatenate(all_es).flatten()
    ss = np.concatenate(all_ss).flatten()
    mean_es, mean_ss = np.mean(es), np.mean(ss)
    std_es, std_ss = np.std(es), np.std(ss)
    return std_es, std_ss


# padding/nESS/etc. penalty utils.
def abs_nESSs_penalty(nESS, nESS_frac_threshold: float, nESS_coeff: float):
    """add a penalty that penalizes loss of nESS below some threshold with a flat-bottomed quadratic term"""
    nESS_diff = nESS - nESS_frac_threshold
    return jax.lax.select(nESS_diff < 0., nESS_coeff * nESS_diff**2, nESS_diff * 0.)


def elec_hard_pert_penalty(orig_es_ss: jax.Array, mod_es_ss: jax.Array, e_coeff: float, s_coeff: float):
    """penalize deviations of the electronegativity and hardnesses from original params with a quadratic term"""
    diffs = orig_es_ss - mod_es_ss
    square_diffs = diffs**2
    return e_coeff * jnp.mean(square_diffs[0,:]) + s_coeff * jnp.mean(square_diffs[1,:])

def dg_loss_aux(exp_dg, orig_calc_dg, orig_calc_ddg, orig_us, prefactors, es, ss, hs, 
            total_charge, pc_vecs, params, eps, e_scale, s_scale, model, loss_fn):
    orig_params = jnp.concatenate([es[..., jnp.newaxis], ss[..., jnp.newaxis]], axis=-1)
    ligand_tm_charges, (orig_es_ss, mod_es_ss) = compute_charges(
        hs, params, orig_params, total_charge, eps, pc_vecs, e_scale, s_scale, model)
    pert_us = U_ligand_env(ligand_tm_charges, prefactors) * BETA
    loss, (ESS, delta_us, reweighted_solv_dg, reweighted_solv_ddg) = abs_dg_zwanzig_loss(
        pert_us, orig_us, orig_calc_dg, orig_calc_ddg, exp_dg, loss_fn)
    return loss, (ESS, delta_us, reweighted_solv_dg, reweighted_solv_ddg, ligand_tm_charges, orig_es_ss, mod_es_ss)

def create_pads(
    es, ss, hs, prefactors, ligand_charges):
    """return es, ss, prefactors, tm_ligand_charges as list of different shaped arrs
    with appropriate pads"""
    max_num_atoms = max([len(e) for e in es])
    pad_es = [np.pad(e, ((0,max_num_atoms - len(e)),), mode='constant', constant_values=ELECTRONEGATIVITY_PAD) for e in es]
    pad_ss = [np.pad(s, ((0,max_num_atoms - len(s)),), mode='constant', constant_values=HARDNESS_PAD) for s in ss]
    pad_hs = [np.pad(h, ((0,max_num_atoms - len(h)),(0,0)), mode='constant', constant_values=EMBED_PAD) for h in hs]
    pad_ligand_charges = [np.pad(q, ((0,max_num_atoms - len(q)),), mode='constant', constant_values=0.) for q in ligand_charges]
    pad_prefactors = [np.pad(p, ((0,0), (0,max_num_atoms - p.shape[1])), mode='constant', constant_values=0.) for p in prefactors]
    return jnp.array(pad_es), jnp.array(pad_ss), jnp.array(pad_hs), jnp.array(pad_prefactors), jnp.array(pad_ligand_charges)


# ahfe loss fn
def get_ahfe_joint_loss( 
    tm_ligand_charges, # tm
    hs, 
    es, 
    ss,
    num_pcs,
    mlp_init_params: typing.Union[typing.Tuple[int], None], # if mlp, (num_features, num_layers)
    ):
    """given experimental dgs, calculated dgs, prefactors (at lambda 1), ligand charges (in tm),
    electronegativities, hardnesses, atom embeddings, and the number of principal components to use, 
    get a joint loss function"""
    if mlp_init_params:
        model = MLP(*mlp_init_params) # features, layers, (output dim defaults as 2)
        model_params = model.init(DEFAULT_NN_KEY, jnp.zeros(num_pcs))
    else:
        model = None
        model_params = jnp.ones((2, num_pcs)) * 1e-6 # make init params small to as to start ~0 perturbation.

    pc_vals, pc_vecs = embedding_pca(jnp.vstack(hs)) # compute pcs
    pc_vecs = jnp.transpose(pc_vecs)[:num_pcs]
    ligand_charges = [mol_tm_charges / jnp.sqrt(ONE_4PI_EPS0) for mol_tm_charges in tm_ligand_charges] # convert tm ligand charges back
    # compute the std of the es_ss for all particles to get `e_scale`, `s_scale` for `compute_charges`
    e_scale, s_scale = np.std(np.concatenate(es).flatten()), np.std(np.concatenate(ss).flatten())
    
    pad_es, pad_ss, pad_hs, pad_prefactors, pad_ligand_charges = create_pads(es, ss, hs, prefactors, ligand_charges)
    orig_us = jax.vmap(U_ligand_env, in_axes=(0,0))(
        pad_ligand_charges * np.sqrt(ONE_4PI_EPS0), pad_prefactors) * BETA # back to tm_ligand charges, and reduce potential energy
    dg_loss_fn = functools.partial(
        dg_loss_aux, model = model, loss_fn = dg_pseudo_huber_loss) # partial dg loss fn

    def total_loss_fn(
        exp_dg, orig_calc_dg, orig_calc_ddg, orig_us, prefactors, es, ss, 
        hs, pad_ligand_charges, pc_vecs, params, eps, e_scale, s_scale, nESS_frac_threshold, nESS_coeff, dg_loss_weight):
        total_charge = pad_ligand_charges.sum()
        dg_loss, (ESS, delta_us, reweighted_solv_dg, reweighted_solv_ddg, ligand_tm_charges, orig_es_ss, mod_es_ss) = dg_loss_fn(
            exp_dg, orig_calc_dg, orig_calc_ddg, orig_us, prefactors, es, ss, hs, 
            total_charge, pc_vecs, params, eps, e_scale, s_scale)
        nESS_loss = abs_nESSs_penalty(ESS, nESS_frac_threshold, nESS_coeff)
        return dg_loss*dg_loss_weight, nESS_loss, (ESS, delta_us, orig_calc_dg, reweighted_solv_dg, reweighted_solv_ddg, exp_dg, ligand_tm_charges, orig_es_ss, mod_es_ss)
    return pc_vals, pc_vecs, pad_es, pad_ss, pad_hs, pad_prefactors, pad_ligand_charges, e_scale, s_scale, orig_us, total_loss_fn, model_params


# train/test retrieval utils
def indices_in_ascending_order(arr):
    # Pair each element with its index
    indexed_arr = list(enumerate(arr))
    # Sort the pairs based on the values
    sorted_indexed_arr = sorted(indexed_arr, key=lambda x: x[1])
    # Extract the indices from the sorted pairs
    sorted_indices = [index for index, value in sorted_indexed_arr]
    return sorted_indices
    
def retrieve_idxs_by_dg_neg(dGs, fraction_descent):
    list_dGs = dGs.tolist()
    len_abs = len(list_dGs)
    up_to_idx = int(np.floor(fraction_descent * len_abs))
    descending_indices = indices_in_ascending_order(list_dGs)[::-1]
    indices_to_pull = set(descending_indices[:up_to_idx])
    rest_of_indices = set(descending_indices[up_to_idx:])
    return jnp.array(list(indices_to_pull)), jnp.array(list(rest_of_indices))

def generate_sequence(key, min_val: int, max_val: int, num: int):
    integers = jnp.arange(min_val, max_val + 1)
    sequence = jax.random.choice(key, integers, shape=(num,), replace=False)
    return sequence


# parameter reshaping utils
def flatten_list_of_arrs(arr_list):
    return jnp.concatenate([arr.flatten() for arr in arr_list])

def reshape_from_template(flat_array, template_list):
    reshaped_arrays = []
    current_index = 0
    
    for template in template_list:
        # Get the shape and number of elements of the current template array
        shape = jnp.array(list(template.shape))
        num_elements = jnp.prod(shape)
        
        # Extract the appropriate slice from the flat array
        array_slice = flat_array[current_index:current_index + num_elements]
        
        # Reshape the slice to the original shape of the template
        reshaped_array = array_slice.reshape(shape)
        
        # Append the reshaped array to the list
        reshaped_arrays.append(reshaped_array)
        
        # Update the current index to the next position
        current_index += num_elements
        
    return reshaped_arrays


# training wrapper
class Wrapper:
    def __init__(
        self,
        exp_dgs: jax.Array,
        orig_calc_dgs: jax.Array,
        orig_calc_ddgs: jax.Array, 
        tm_ligand_charges: jax.Array, # tm
        hs: jax.Array, 
        es: jax.Array,
        ss: jax.Array,
        num_pcs: jax.Array,
        mlp_init_params: typing.Union[typing.Tuple[int], None], # use (2,1) as default (2 features, 1 layer)
        retrieve_by_descent: bool,
        retrieval_seed: float,
        train_fraction: float,
        nESS_frac_threshold: float,
        nESS_coeff: float,
        nESS_on_test: bool = False,
        ):
        self.exp_dgs = exp_dgs
        self.orig_calc_dgs = orig_calc_dgs
        self.orig_calc_ddgs = orig_calc_ddgs
        self.tm_ligand_charges = tm_ligand_charges
        self.hs = hs
        self.num_pcs = num_pcs
        self.retrieve_by_descent = retrieve_by_descent
        self.retrieval_seed = retrieval_seed
        self.train_fraction = train_fraction
        self.test_fraction = 1. - self.train_fraction
        self.nESS_frac_threshold = nESS_frac_threshold
        self.nESS_coeff = nESS_coeff
        self.nESS_on_test = nESS_on_test
        (
            self.pc_vals, 
            self.pc_vecs, 
            self.pad_es, 
            self.pad_ss, 
            self.pad_hs,
            self.pad_prefactors, 
            self.pad_ligand_charges, 
            self.e_scale, 
            self.s_scale, 
            self.orig_us, 
            self.loss_fn, 
            self.model_params
        ) = get_ahfe_joint_loss(self.tm_ligand_charges, self.hs, es, ss, self.num_pcs, mlp_init_params)

        # handle flattening.
        if mlp_init_params:
            self.params_as_dict = True
            self.params_list, self.treedef = jax.tree.flatten(self.model_params)
            self.flat_params = self.dict_to_flat(self.model_params)
        else:
            self.params_as_dict = False
            self.params_list, self.treedef = None, None
            self.flat_params = self.model_params.flatten()
        

        (self.train_idxs, self.test_idxs) = self.split_train_test()
        (self.train_loss_fn, self.test_losses_fn) = self.get_loss_fn()
        self.cache = {}
        

    def split_train_test(self):
        if self.retrieve_by_descent:
            train_idxs, test_idxs = retrieve_idxs_by_dg_neg(self.exp_dgs, self.train_fraction)
        else:
            num = int(self.train_fraction * len(self.exp_dgs)) 
            train_idxs = generate_sequence(self.retrieval_seed, 0, len(self.exp_dgs)-1, num)
            all_idxs = jnp.arange(len(self.exp_dgs))
            test_idxs = jnp.array([q for q in all_idxs if q not in train_idxs])
        return train_idxs, test_idxs    

    def get_loss_fn(self):
        # exp_dg, orig_calc_dg, orig_us, prefactors, es, ss, 
        # hs, pad_ligand_charges, pc_vecs, params, eps, e_scale, s_scale, nESS_frac_threshold, nESS_coeff, dg_loss_weight)
        vloss_fn = jax.vmap(self.loss_fn, in_axes=(0,0,0,0,0,0,0,0,0,None,None,None,None,None,None,None,0))
        def _loss_fn(params, dg_loss_weight, idxs):
            dg_losses, nESS_losses, auxs = vloss_fn(
                self.exp_dgs[idxs], self.orig_calc_dgs[idxs], self.orig_calc_ddgs[idxs], self.orig_us[idxs], self.pad_prefactors[idxs],
                self.pad_es[idxs], self.pad_ss[idxs], self.pad_hs[idxs], self.pad_ligand_charges[idxs],
                self.pc_vecs, params, 1e-8, self.e_scale, self.s_scale, self.nESS_frac_threshold, self.nESS_coeff, dg_loss_weight)
            return jnp.mean(dg_losses) + jnp.sum(nESS_losses), auxs
        if self.nESS_on_test: # need to run twice 
            def train_loss(params):
                train_loss_vals, train_loss_auxs = _loss_fn(
                    params, jnp.ones(self.train_idxs.shape, dtype=jnp.float64), self.train_idxs)
                test_loss_vals, test_loss_auxs = _loss_fn(
                    params, jnp.zeros(self.test_idxs.shape, dtype=jnp.float64), self.test_idxs
                )
                return jnp.mean(train_loss_vals) + jnp.mean(test_loss_vals), train_loss_auxs
        else:
            def train_loss(params):
                train_loss_vals, train_loss_auxs = _loss_fn(
                    params, jnp.ones(self.train_idxs.shape, dtype=jnp.float64), self.train_idxs)
                return jnp.mean(train_loss_vals), train_loss_auxs
                
        def test_losses(params):
            loss_vals, auxs = _loss_fn(
                params, jnp.ones(self.test_idxs.shape, dtype=jnp.float64), self.test_idxs)
            return loss_vals, auxs

        return jax.jit(train_loss), jax.jit(test_losses)

    def flat_to_dict(self, flat_params):
        list_params = reshape_from_template(flat_params, self.params_list)
        return jax.tree.unflatten(self.treedef, list_params)

    def dict_to_flat(self, dict_params):
        params_list, _ = jax.tree.flatten(dict_params)
        return flatten_list_of_arrs(params_list)

    def __call__(self, flat_params, *args):
        if self.params_as_dict:
            params = self.flat_to_dict(flat_params)
        else:
            params = flat_params.reshape(*self.model_params.shape)
            
        (val, aux_data), grad = jax.value_and_grad(self.train_loss_fn, has_aux=True)(params)

        if self.params_as_dict:
            self.cache['grad'] = self.dict_to_flat(grad)
        else:
            self.cache['grad'] = grad.flatten()
        print(val)
        return val
        
    def jac(self, x, *args):
        return self.cache.pop('grad')
