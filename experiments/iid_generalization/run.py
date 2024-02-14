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

import numpyro
from numpyro.infer import MCMC, NUTS, autoguide, SVI, Trace_ELBO
from numpyro.infer import init_to_value

from src.utils import adj2vec, degrees_from_upper_tri

from src.models import dpg_bnn, dpg_mimo_bnn, dpg_mimo_stochastic_head, pds_bnn
#from src.data import load_data
from src.metrics import compute_metrics

from src.config import num_samples_to_generate, w_init_scale, lam_init_scale, altered_prior
from config import dpg_hyperparameters, pds_hyperparameters, dpg_mimo_hyperparameters, dpg_mimo_e_hyperparameters, experiment_settings
from src import NUM_BINS

num_chains, num_warmup_samples, num_samples = 4, 500, 1000 #8, 500, 500 #jax.local_device_count()



def load_data():
    data_dict = pickle.load(open(experiment_settings['data_path'], "rb"))

    # data is pairs of adjacency matrices and euclidean distance matrices
    adjacencies = data_dict['adjacencies'].astype(np.float32)
    dataset_key = 'expected' if experiment_settings['num_signals'] == float('inf') else str(experiment_settings['num_signals'])
    euclidean_distance_matrices = data_dict[dataset_key]

    # convert to vectors
    adjacencies = adj2vec(adjacencies)
    euclidean_distance_matrices = adj2vec(euclidean_distance_matrices)
    num_edges = adjacencies.shape[-1]

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

    return (x_train, y_train), (x_test, y_test)


def run_dpg(model_class, params, data):
    """ Train """
    x_train, y_train = data[0]
    num_train_samples = len(x_train)
    num_edges = x_train.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))
    S = jnp.array(degrees_from_upper_tri(n))
    model_args = {'x': x_train, 'y': y_train,
                'depth': params['depth'],
                'w_init': w_init_scale * jnp.ones((num_train_samples, num_edges)), 
                'lam_init': lam_init_scale * jnp.ones((num_train_samples, n)),
                'S': S,
                'prior_settings': params['prior']} #altered_prior} # priors for model parameters

    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = jax_random.PRNGKey(0)
    rng_key, rng_key_ = jax_random.split(rng_key)
    kernel = NUTS(model_class.model, forward_mode_differentiation=True)
    mcmc = MCMC(kernel,
                num_warmup=num_warmup_samples, num_samples=num_samples,
                progress_bar=True,
                num_chains=num_chains, chain_method='parallel')
    start_time = time.time()
    mcmc.run(rng_key_, **model_args)
    end_time = time.time()
    print(f"Time taken for inference using {num_chains} with {num_warmup_samples} warmup samples and {num_samples} samples: {end_time - start_time}")
    mcmc.print_summary()
    print(f"^^********** Finished  **********^^\n\n")
    samples = mcmc.get_samples()

    """ Test """
    x_test, y_test = data[1]
    num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
    w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))

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
    return samples, metrics_dict

def run_pds(model_class, params, data):


    """ Train """
    x_train, y_train = data[0]
    num_train_samples = len(x_train)
    num_edges = x_train.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))
    S = jnp.array(degrees_from_upper_tri(n))
    model_args = {'x': x_train, 'y': y_train,
                  'depth': params['depth'],
                  'w_init': jnp.zeros((num_train_samples, num_edges)), # initializations dont seem to impact iid perf
                  'v_init': jnp.zeros((num_train_samples, n)),
                  'S': S,
                  'prior_settings': params['prior']} # priors for model parameters

    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = jax_random.PRNGKey(0)
    rng_key, rng_key_ = jax_random.split(rng_key)
    kernel = NUTS(model_class.model, forward_mode_differentiation=True)
    #num_chains, num_warmup_samples, num_samples = 8, 500, 250 #jax.local_device_count()
    mcmc = MCMC(kernel,
                num_warmup=num_warmup_samples, num_samples=num_samples,
                progress_bar=True,
                num_chains=num_chains, chain_method='parallel')
    start_time = time.time()
    mcmc.run(rng_key_, **model_args)
    end_time = time.time()
    print(f"Time taken for inference using {num_chains} with {num_warmup_samples} warmup samples and {num_samples} samples: {end_time - start_time}")
    mcmc.print_summary()
    print(f"^^********** Finished  **********^^\n\n")
    samples = mcmc.get_samples()

    """ Test """
    x_test, y_test = data[1]
    num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
    w_test, v_test = jnp.zeros((num_test_samples, num_edges)), jnp.zeros((num_test_samples, n))

    edge_logits = model_class.forward_pass_vmap()(
        samples['alpha'],
        samples['beta'],
        samples['gamma'], # this is just 0.1 for every sample
        samples['b'],
        x_test,
        w_test,
        v_test,
        params['depth'],
        S)


    metrics_dict = compute_metrics(edge_logits, y_test, NUM_BINS)
    return samples, metrics_dict

def run_dpg_mimo(model_class, params, data):
    """ Train """
    x_train, y_train = data[0]
    num_train_samples = len(x_train)
    num_edges = x_train.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))
    S = jnp.array(degrees_from_upper_tri(n))
    w_init = w_init_scale * jnp.ones((num_train_samples, num_edges))
    lam_init = lam_init_scale * jnp.ones((num_train_samples, n))
    w_init, lam_init = \
        jnp.repeat(jnp.expand_dims(w_init, axis=-1), params['num_channels'], axis=-1), \
        jnp.repeat(jnp.expand_dims(lam_init, axis=-1), params['num_channels'], axis=-1)
    model_args = {'x': x_train, 'y': y_train,
                  'depth': params['depth'],
                  'w_init': w_init, 
                  'lam_init': lam_init,
                  'S': S,
                  'prior_settings': params['prior'],
                  'num_channels': params['num_channels']}

    def run_svi(model, guide_family, model_args, optim_args):
        if guide_family == "AutoDelta":
            guide = autoguide.AutoDelta(model)
        elif guide_family == "AutoDiagonalNormal":
            guide = autoguide.AutoDiagonalNormal(model)
        else:
            raise ValueError("Invalid guide family")

        optimizer = numpyro.optim.Adam(optim_args['learning_rate'])
        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_results = svi.run(PRNGKey(1), optim_args['maxiter'], **model_args)
        params = svi_results.params

        return params, guide
    
    point_estimates, map_guide = run_svi(model_class.model, "AutoDelta", model_args, params['optim_args'])
    point_estimates['theta'] = point_estimates['theta_auto_loc']
    point_estimates['delta'] = point_estimates['delta_auto_loc']
    point_estimates['b'] = point_estimates['b_auto_loc']
    del point_estimates['theta_auto_loc']; del point_estimates['delta_auto_loc']; del point_estimates['b_auto_loc']

    """ Test """
    x_test, y_test = data[1]
    num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
    w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))
    w_test, lam_test = \
        jnp.repeat(jnp.expand_dims(w_test, axis=-1), params['num_channels'], axis=-1), \
        jnp.repeat(jnp.expand_dims(lam_test, axis=-1), params['num_channels'], axis=-1)
    edge_logits = dpg_mimo_bnn.forward_pass(point_estimates['theta'], 
                                            point_estimates['delta'], 
                                            point_estimates['b'], 
                                            x_test, 
                                            w_test, 
                                            lam_test,
                                            params['depth'],
                                            S)
    metrics_dict = compute_metrics(jnp.expand_dims(edge_logits, axis=0), y_test, NUM_BINS)
    return point_estimates, metrics_dict

def run_dpg_mimo_e(model_class, params, data, mimo_base_samples):
    """ Train """
    x_train, y_train = data[0]
    num_train_samples = len(x_train)
    num_edges = x_train.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))
    S = jnp.array(degrees_from_upper_tri(n))
    w_init = w_init_scale * jnp.ones((num_train_samples, num_edges))
    lam_init = lam_init_scale * jnp.ones((num_train_samples, n))
    mimo_base_params = params['mimo_base']
    w_init, lam_init = \
        jnp.repeat(jnp.expand_dims(w_init, axis=-1), mimo_base_params['num_channels'], axis=-1), \
        jnp.repeat(jnp.expand_dims(lam_init, axis=-1), mimo_base_params['num_channels'], axis=-1)
    w, lam = dpg_mimo_bnn.forward_pass(mimo_base_samples['theta'], 
                                       mimo_base_samples['delta'], 
                                       mimo_base_samples['b'],
                                       x_train, 
                                       w_init, 
                                       lam_init,
                                       mimo_base_params['depth'],
                                       S,
                                       return_raw_output=True)
    
    # now run inference on the stochastic head using the static outputs from the mimo base model
    # stochastic head model args
    model_args = {'x': x_train, 'y': y_train,
                  'w_init': w, 
                  'lam_init': lam,
                  'S': S,
                  'depth': params['stochastic_head_depth'],
                  'num_stochastic_channels': params['num_stochastic_channels'],
                  'prior_settings': params['prior']}
    
    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = jax_random.PRNGKey(0)
    rng_key, rng_key_ = jax_random.split(rng_key)
    kernel = NUTS(model_class.model, forward_mode_differentiation=True)
    mcmc = MCMC(kernel,
                num_warmup=num_warmup_samples, num_samples=num_samples,
                progress_bar=True,
                num_chains=num_chains, chain_method='parallel')
    start_time = time.time()
    mcmc.run(rng_key_, **model_args)
    end_time = time.time()
    print(f"Time taken for inference using {num_chains} with {num_warmup_samples} warmup samples and {num_samples} samples: {end_time - start_time}")
    mcmc.print_summary()
    print(f"^^********** Finished  **********^^\n\n")
    samples = mcmc.get_samples() 


    """ Test """
    x_test, y_test = data[1]
    num_test_samples, n, num_edges = len(x_test), int(0.5*(np.sqrt(8 * x_test.shape[-1] + 1) + 1)), x_test.shape[-1]
    w_test, lam_test = w_init_scale * jnp.ones((num_test_samples, num_edges)), lam_init_scale * jnp.ones((num_test_samples, n))
    w_test, lam_test = \
        jnp.repeat(jnp.expand_dims(w_test, axis=-1), mimo_base_params['num_channels'], axis=-1), \
        jnp.repeat(jnp.expand_dims(lam_test, axis=-1), mimo_base_params['num_channels'], axis=-1)
    
    w_embed, lam_embed = dpg_mimo_bnn.forward_pass(mimo_base_samples['theta'], 
                                       mimo_base_samples['delta'], 
                                       mimo_base_samples['b'], 
                                       x_test, 
                                       w_test, 
                                       lam_test,
                                       params['stochastic_head_depth'],
                                       S,
                                       params['num_stochastic_channels'])
    
    # now pass through the stochastic head
    partial_stoch_logits = model_class.forward_pass_vmap()(samples['theta'],
                                                           samples['delta'],
                                                           samples['b'],
                                                           x_test,
                                                           w_embed,
                                                           lam_embed,
                                                           params['stochastic_head_depth'],
                                                           S,
                                                           params['num_stochastic_channels']
                                                           )

    metrics_dict = compute_metrics(partial_stoch_logits, y_test, NUM_BINS) #jnp.expand_dims(partial_stoch_logits, axis=0), y_test, NUM_BINS)
    return samples, metrics_dict


def display_metrics(metrics_dict):
    calibration_dict = metrics_dict['calibration_dict']
    print(f'Test Error: {1 - metrics_dict["accuracies"].mean():.5f} \pm {metrics_dict["accuracies"].std():.5f}')
    print(f'Test NLL: {-1 * metrics_dict["log_likelihoods"].mean():.3f} \pm {metrics_dict["log_likelihoods"].std():.3f}')
    print(f'Test BS: {metrics_dict["brier_scores"].mean():.5f} \pm {metrics_dict["brier_scores"].std():.5f}')
    print(f'Test ECE:{calibration_dict["ece"]:.5f}') 

def main():
    # Load the data
    data = load_data()
    
    # Models to evaluate
    models = {
        'DPG': (dpg_bnn, dpg_hyperparameters, f"dpg_D={dpg_hyperparameters['depth']}.pkl"),
        'PDS': (pds_bnn, pds_hyperparameters, f"pds_D={dpg_hyperparameters['depth']}.pkl"),
        'DPG-MIMO': (dpg_mimo_bnn, dpg_mimo_hyperparameters, f"dpg_mimo_D={dpg_mimo_hyperparameters['depth']}_C={dpg_mimo_hyperparameters['num_channels']}.pkl"),
        'DPG-MIMO-E': (dpg_mimo_stochastic_head, dpg_mimo_e_hyperparameters, f"dpg_mimo_e_D={dpg_mimo_e_hyperparameters['mimo_base']['depth']}_C={dpg_mimo_e_hyperparameters['mimo_base']['num_channels']}_stoch_head_C={dpg_mimo_e_hyperparameters['num_stochastic_channels']}_D={dpg_mimo_e_hyperparameters['stochastic_head_depth']}.pkl")
    }
    
    # Run experiments for each model
    for name, (model, params, results_file) in models.items():
        results_path = os.path.join(experiment_settings['results_path'], results_file)
        print(f"\n**** {name} ****")
        if os.path.exists(results_path):
            print(f"Results already exist at {results_path}. Skipping experiment.")
            # Load results
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
                #samples_key = 'samples' if name != 'dpg_mimo' else 'map point estimates'
                samples, metrics_dict = results['samples'], results['metrics']
        else:
            print(f"Results do not exist at {results_path}. Running experiment.")
            if name == 'DPG':
                samples, metrics_dict = run_dpg(model, params, data)
            elif name == 'PDS':
                samples, metrics_dict = run_pds(model, params, data)
            elif name == 'dpg_mimo':
                samples, metrics_dict = run_dpg_mimo(model, params, data)
            elif name == 'dpg_mimo_e':
                # we must have already trained the base dpg_mimo model. Check that it's trained, and feed in the samples
                mimo_base_params = params['mimo_base']
                base_results_file = f"dpg_mimo_D={mimo_base_params['depth']}_C={mimo_base_params['num_channels']}.pkl"
                base_results_path = os.path.join(experiment_settings['results_path'], base_results_file)
                if not os.path.exists(base_results_path):
                    raise ValueError(f"Base dpg_mimo model not trained. Results not found at {base_results_path}")
                with open(base_results_path, 'rb') as f:
                    base_results = pickle.load(f)
                    base_samples = base_results['samples']
                samples, metrics_dict = run_dpg_mimo_e(model, params, data, base_samples)
            else:
                raise ValueError(f"Model {model} not recognized")
        
        display_metrics(metrics_dict)
        # If experiment was run, load results and calculate metrics
        if not os.path.exists(results_path):
            print(f"Saving results to {results_path}")
            results = {'samples': samples, 'metrics': metrics_dict}
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)

if __name__ == '__main__':
    main()
