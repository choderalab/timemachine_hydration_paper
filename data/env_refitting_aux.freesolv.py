import jax
jax.config.update("jax_enable_x64", True)
import pickle
import argparse
import os
import numpy as np
import typing
import functools
import itertools
import scipy
import tqdm
from jax import numpy as jnp
from matplotlib import pyplot as plt

from timemachine.fe.refitting import Wrapper, BETA, load_pkl_data, embedding_pca, ESS_from_delta_us, abs_dg_reweighting_zwanzig, EMBED_DIM
from timemachine.constants import KCAL_TO_KJ

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="freesolv_run")
    # Add arguments
    parser.add_argument("idx", type=int, help="index of combination to use")
    args = parser.parse_args()
    print(args.idx)

    # hardcoded/not experimented
    nESS_frac_threshold = 0.15
    nESS_break_threshold = 0.1
    nESS_on_test = True
    retrieve_by_descent = False
    use_ml = False
    
    # build list of experiments; there are 144 experiments in this regime.
    params_list = []
    for num_pcs in [10, 25, 50, 75, 100, 250, 500, EMBED_DIM]: # 8*6 = 48 experiments
        for train_fraction in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9]:
            params_list.append([num_pcs, retrieve_by_descent, train_fraction, use_ml])
    print(len(params_list))

    # # Add arguments
    parser.add_argument("idx", type=int, help="index of combination to use")

    # Parse the arguments
    (num_pcs, retrieve_by_descent, train_fraction, use_ml) = params_list[args.idx]
    print(f"num_pcs: {num_pcs}; train_fraction: {train_fraction}")
    mlp_init_params = (2, 1) if use_ml else None
    
    data = load_pkl_data('agg_freesolv_data.pkl')
    (out_names, arr_exp_dGs, arr_calc_dGs, prefactors, tm_ligand_charges, es, ss, hs) = data
    # convert to reduced units since corrections are also reduced
    arr_exp_dGs = jnp.array(arr_exp_dGs) * KCAL_TO_KJ * BETA
    arr_calc_dGs = jnp.array(arr_calc_dGs) * KCAL_TO_KJ * BETA
    test_fraction = (1. - train_fraction) * 0.75
    
    wrapper = Wrapper(
            exp_dgs = arr_exp_dGs[:,0],
            orig_calc_dgs = arr_calc_dGs[:,0],
            orig_calc_ddgs = arr_calc_dGs[:,1],
            tm_ligand_charges = tm_ligand_charges, # tm
            hs = hs, 
            es = es,
            ss = ss,
            prefactors = prefactors,
            num_pcs = num_pcs,
            retrieve_by_descent = retrieve_by_descent,
            retrieval_seed = jax.random.PRNGKey(48),
            train_fraction = train_fraction,
            test_fraction = test_fraction, 
            nESS_frac_threshold = nESS_frac_threshold,
            nESS_coeff = 100.,
            nESS_on_test = nESS_on_test,
            mlp_init_params = mlp_init_params,
            use_pca = num_pcs != EMBED_DIM,
    )

    wrapper._prev_params = wrapper.model_params
    wrapper._callback_counter = 0
    # need to make a modification of validate callback if we want to STOP optimization if any ESS 
    def callback(flat_params, *args):
        _ = wrapper.validate_callback(flat_params, *args) # early stopping if validate falls outside of 95% ci
        params = flat_params.reshape(*wrapper.model_params.shape)
        train_loss_vals, train_loss_auxs = wrapper.train_loss_fn(params)
        test_loss_vals, test_loss_auxs = wrapper.test_losses_fn(params)
        validate_loss_vals, _, validate_loss_auxs = wrapper.validate_losses_fn(params)
        train_nESSs, validate_nESSs, test_nESSs = train_loss_auxs[0], test_loss_auxs[0], validate_loss_auxs[0]
        try:
            assert not np.any(train_nESSs < nESS_break_threshold)
            assert not np.any(validate_nESSs < nESS_break_threshold)
            assert not np.any(test_nESSs < nESS_break_threshold)
        except Exception as e:
            print(e)
            raise ValueError(f"nESS threshold surpassed")
        wrapper._prev_params = params
        wrapper._callback_counter += 1


    # set maxiter to allow optimization to complete (not always the case for large `num_pcs`)
    try: # if callback fails, we still save `validate_min_mean_loss_params` to protect optimal params
        res = scipy.optimize.minimize(
            wrapper, wrapper.flat_params, method = 'BFGS', 
            jac = wrapper.jac, callback=callback, options={'maxiter': 1000})
        print(f"optimization complete without break")
    except Exception as e:
        res = None
        print(f"optimization failed with break: {e}")
        
    params = wrapper.cache['validate_min_mean_loss_params']
        
    # then compute res from train/test
    train_loss_vals, train_loss_auxs = wrapper.train_loss_fn(params)
    test_loss_vals, test_loss_auxs = wrapper.test_losses_fn(params)
    validate_loss_vals, _, validate_loss_auxs = wrapper.validate_losses_fn(params)
    my_list = [(res, wrapper._callback_counter), params, train_loss_auxs, test_loss_auxs, validate_loss_auxs, 
               wrapper.train_idxs, wrapper.test_idxs, wrapper.validate_idxs, wrapper.cache]

    # num_pcs, retrieve_by_descent, train_fraction, use_ml
    with open(
        #f"{num_pcs}_{retrieve_by_descent}_{train_fraction}_{use_ml}.t2.freesolv.pkl", 
        # f"{num_pcs}_{retrieve_by_descent}_{train_fraction}_{use_ml}.t4.freesolv.pkl", # for consistency 
        f"{num_pcs}_{retrieve_by_descent}_{train_fraction}_{use_ml}_{nESS_on_test}.t5.freesolv.pkl", # for consistency 
        'wb') as file:
        pickle.dump(my_list, file)
