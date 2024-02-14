#!/usr/bin/env python
# run.py

import os
import pickle
import numpy as np
import multiprocessing
import time

### THIS MUST OCCUR BEFORE JAX IS IMPORTED ###
# for parallelization for multiple chains. Must be done before jax import:
# Blackjax tutorial: https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
import jax 
import jax.numpy as jnp
from jax import random as jax_random
from jax.random import PRNGKey
print("Number of CPU cores:", jax.local_device_count()) # often will not allow multi-threading in Jupyter notebooks


from src.utils import adj2vec, degrees_from_upper_tri

from src.models import dpg_bnn, dpg_mimo_bnn, dpg_mimo_stochastic_head, pds_bnn
#from src.data import load_data
from src.metrics import compute_metrics

from src.config import num_samples_to_generate, w_init_scale, lam_init_scale, altered_prior

# take the model configs from iid_generalization experiment
from experiments.iid_generalization.config import dpg_hyperparameters, pds_hyperparameters, dpg_mimo_hyperparameters, dpg_mimo_e_hyperparameters, experiment_settings
from src import NUM_BINS

from src import RESULTS_PATH
PATH2COVARIATE_SHIFT_RESULTS = RESULTS_PATH + 'covariate_shift/'

def load_test_data():
    data_dict = pickle.load(open(experiment_settings['data_path'], "rb"))

    # data is pairs of adjacency matrices and euclidean distance matrices
    adjacencies = data_dict['adjacencies'].astype(np.float32)
    adjacencies = adj2vec(adjacencies)
    num_edges = adjacencies.shape[-1]

    print(f'Signals (keys) in data_dict: {data_dict.keys()}')
    data_sets = {}
    for signals_key in data_dict:
        if any(term in signals_key for term in ['adjacencies', 'graph_params']): # these don't rever to varying numbers of signal data
            continue

        euclidean_distance_matrices = adj2vec(data_dict[signals_key])

        # concatenate for easier processing
        data = np.concatenate([euclidean_distance_matrices, adjacencies], axis=1)

        # predetermine train/val/test split
        num_train, num_val, num_test \
            = num_samples_to_generate['train'], num_samples_to_generate['val'], num_samples_to_generate['test']
        train, val, test = data[:num_train], data[num_train:num_train + num_val], data[num_train + num_val:]
        data =  {"train": (train[:, :num_edges], train[:, num_edges:]), 
                "val": (val[:, :num_edges], val[:, num_edges:]),
                "test": (test[:, :num_edges], test[:, num_edges:])}

        # unpack data
        x_train, y_train = jnp.array(data['train'][0]), jnp.array(data['train'][1])
        x_train, y_train = x_train[:experiment_settings['num_train_samples']], y_train[:experiment_settings['num_train_samples']]
        
        x_test, y_test = jnp.array(data['test'][0]), jnp.array(data['test'][1])
        x_test, y_test = x_test[:experiment_settings['num_test_samples']], y_test[:experiment_settings['num_test_samples']]

        data_sets[signals_key] = (x_test, y_test) # y_test is the same for all signals
    
    return data_sets

def display_metrics(metrics_dict, precision=8):
    calibration_dict = metrics_dict['calibration_dict']
    precision_format = f'.{precision}f'
    std_precision_format = f'.{precision}f' if precision != 3 else '.3f'  # Adjusting for different precision needs

    print(f'Test Error: {1 - metrics_dict["accuracies"].mean():{precision_format}} \pm {metrics_dict["accuracies"].std():{std_precision_format}}')
    print(f'Test NLL: {-1 * metrics_dict["log_likelihoods"].mean():.3f} \pm {metrics_dict["log_likelihoods"].std():.3f}')
    print(f'Test BS: {metrics_dict["brier_scores"].mean():{precision_format}} \pm {metrics_dict["brier_scores"].std():{std_precision_format}}')
    print(f'Test ECE:{calibration_dict["ece"]:{precision_format}}')

def run_dpg(model_class, params, data, samples):
    
    num_signals_to_metrics_dict = {}

    """ Run the test data through the model and compute metrics for each dataset """
    for num_signals, (x_test, y_test) in data.items():
        print(f"\n** Running DPG model for {num_signals} signals **")
        """ Test """
        num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
        w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))
        S = jnp.array(degrees_from_upper_tri(n))
        edge_logits = model_class.forward_pass_vmap()(
            samples['theta'],
            samples['delta'],
            samples['b'],
            x_test,
            w_test,
            lam_test,
            params['depth'],
            S)
        metrics_dict = compute_metrics(edge_logits, y_test, NUM_BINS)
        
        num_signals_to_metrics_dict[num_signals] = metrics_dict
        display_metrics(metrics_dict)
    return num_signals_to_metrics_dict

def run_pds(model_class, params, data, samples):
    
    num_signals_to_metrics_dict = {}

    """ Run the test data through the model and compute metrics for each dataset """
    for num_signals, (x_test, y_test) in data.items():
        print(f"\n** Running PDS model for {num_signals} signals **")
        """ Test """
        num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
        w_test, v_test = jnp.zeros((num_test_samples, num_edges)), jnp.zeros((num_test_samples, n))
        S = jnp.array(degrees_from_upper_tri(n))
        edge_logits = model_class.forward_pass_vmap()(
            samples['alpha'],
            samples['beta'],
            samples['gamma'],
            samples['b'],
            x_test,
            w_test,
            v_test,
            params['depth'],
            S)
        metrics_dict = compute_metrics(edge_logits, y_test, NUM_BINS)
        
        num_signals_to_metrics_dict[num_signals] = metrics_dict
        display_metrics(metrics_dict)
    return num_signals_to_metrics_dict

def run_dpg_mimo(model_class, params, data, samples):
    
    num_signals_to_metrics_dict = {}
    
    """ Test """
    for num_signals, (x_test, y_test) in data.items():
        print(f"\n** Running DPG model for {num_signals} signals **")
        num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
        S = jnp.array(degrees_from_upper_tri(n))
        w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))
        w_test, lam_test = \
            jnp.repeat(jnp.expand_dims(w_test, axis=-1), params['num_channels'], axis=-1), \
            jnp.repeat(jnp.expand_dims(lam_test, axis=-1), params['num_channels'], axis=-1)
        edge_logits = dpg_mimo_bnn.forward_pass(samples['theta'], 
                                                samples['delta'], 
                                                samples['b'], 
                                                x_test, 
                                                w_test, 
                                                lam_test,
                                                params['depth'],
                                                S)
        metrics_dict = compute_metrics(jnp.expand_dims(edge_logits, axis=0), y_test, NUM_BINS)
        num_signals_to_metrics_dict[num_signals] = metrics_dict
        display_metrics(metrics_dict)

    return num_signals_to_metrics_dict

def run_dpg_mimo_e(model_class, params, data, mimo_base_samples, samples):
    num_signals_to_metrics_dict = {}
    mimo_base_params = params['mimo_base'] 
    
    """ Test """
    for num_signals, (x_test, y_test) in data.items():
        print(f"\n\n\n** Running DPG model for {num_signals} signals **")
        num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
        S = jnp.array(degrees_from_upper_tri(n))
        w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))
        w_test, lam_test = \
            jnp.repeat(jnp.expand_dims(w_test, axis=-1), mimo_base_params['num_channels'], axis=-1), \
            jnp.repeat(jnp.expand_dims(lam_test, axis=-1), mimo_base_params['num_channels'], axis=-1)

        w, lam = dpg_mimo_bnn.forward_pass(mimo_base_samples['theta'], 
                                           mimo_base_samples['delta'], 
                                           mimo_base_samples['b'],
                                           x_test, 
                                           w_test, 
                                           lam_test,
                                           mimo_base_params['depth'],
                                           S,
                                           return_raw_output=True)
        
        partial_stoch_logits = model_class.forward_pass_vmap()(samples['theta'],
                                                               samples['delta'],
                                                               samples['b'],
                                                               x_test,
                                                               w,
                                                               lam,
                                                               params['stochastic_head_depth'],
                                                               S,
                                                               params['num_stochastic_channels']
                                                               )
        
        metrics_dict = compute_metrics(partial_stoch_logits, y_test, NUM_BINS) 
        num_signals_to_metrics_dict[num_signals] = metrics_dict
        display_metrics(metrics_dict)

    return num_signals_to_metrics_dict


def main():
    data = load_test_data()
    
    # Models to evaluate
    models = {
        'DPG': (dpg_bnn, dpg_hyperparameters, f"dpg_D={dpg_hyperparameters['depth']}.pkl"),
        'PDS': (pds_bnn, pds_hyperparameters, f"pds_D={dpg_hyperparameters['depth']}.pkl"),
        'DPG-MIMO': (dpg_mimo_bnn, dpg_mimo_hyperparameters, f"dpg_mimo_D={dpg_mimo_hyperparameters['depth']}_C={dpg_mimo_hyperparameters['num_channels']}.pkl"),
        'DPG-MIMO-E': (dpg_mimo_stochastic_head, dpg_mimo_e_hyperparameters, f"dpg_mimo_e_D={dpg_mimo_e_hyperparameters['mimo_base']['depth']}_C={dpg_mimo_e_hyperparameters['mimo_base']['num_channels']}_stoch_head_C={dpg_mimo_e_hyperparameters['num_stochastic_channels']}_D={dpg_mimo_e_hyperparameters['stochastic_head_depth']}.pkl")
    }

    model_metrics_dicts = {}

    # Run experiments for each model
    for name, (model, params, results_file) in models.items():
        results_path = os.path.join(experiment_settings['results_path'], results_file)
        print(f"\n**** {name} ****")
        if not os.path.exists(results_path):
            print("Results don't exist yet. Cannot evaluate.")
        else:
            # Load results = samples and metrics
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                samples, metrics_dict = results['samples'], results['metrics']

            if name == 'DPG':
                model_metrics_dicts[name] = run_dpg(model, params, data, samples)
            elif name == 'PDS':
                model_metrics_dicts[name] = run_pds(model, params, data, samples)
            elif name == 'DPG-MIMO':
                model_metrics_dicts[name] = run_dpg_mimo(model, params, data, samples)
            elif name == 'DPG-MIMO-E':
                # we must have already trained the base dpg_mimo model. Check that it's trained, and feed in the samples
                mimo_base_params = params['mimo_base']
                base_results_file = f"dpg_mimo_D={mimo_base_params['depth']}_C={mimo_base_params['num_channels']}.pkl"
                base_results_path = os.path.join(experiment_settings['results_path'], base_results_file)
                if not os.path.exists(base_results_path):
                    raise ValueError(f"Base dpg_mimo model not trained. Results not found at {base_results_path}")
                with open(base_results_path, 'rb') as f:
                    base_results = pickle.load(f)
                    base_samples = base_results['samples']
                model_metrics_dicts[name] = run_dpg_mimo_e(model, params, data, base_samples, samples)
            else:
                raise ValueError(f"Model {model} not recognized")
    
    # save the resulting dictionaries to a pickle file
    with open(os.path.join(PATH2COVARIATE_SHIFT_RESULTS, 'covariate_shift_metrics.pkl'), 'wb') as f:
        pickle.dump(model_metrics_dicts, f)


if __name__ == '__main__':
    main()
