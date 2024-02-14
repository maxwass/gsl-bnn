# config.py

from src import SYNTHETIC_DATA_PATH, RESULTS_PATH
from src.config import altered_prior, pds_prior_settings
import jax.numpy as jnp

# Hyperparameters
dpg_hyperparameters = {
    'depth': 200,
    'prior': altered_prior
}

dpg_mimo_hyperparameters = {
    'depth': 200,
    'num_channels': 4,
    'optim_args': {'learning_rate': 0.01, 'maxiter': 50000},
    'prior': {
       'theta_loc': altered_prior['theta_loc'],
       'theta_scale': altered_prior['theta_scale'],
       'delta_loc': jnp.zeros(1),
       'delta_scale': 100*jnp.ones(1),
       'b_loc': jnp.zeros(1),
       'b_scale': 10*jnp.ones(1), 
    }
}


dpg_mimo_e_hyperparameters = {
    # DPG MIMO base hyperparameters
    'mimo_base': dpg_mimo_hyperparameters,
    # Stochastic Head hyperparameters
    'num_stochastic_channels': 2,
    'stochastic_head_depth': 20,
    'prior': { # for the stochastic head parameters
       'theta_loc': altered_prior['theta_loc'],
       'theta_scale': altered_prior['theta_scale'],
       'delta_loc': jnp.zeros(1),
       'delta_scale': 100*jnp.ones(1),
       'b_loc': jnp.zeros(1),
       'b_scale': 10*jnp.ones(1), 
    }
}

pds_hyperparameters = {
    'depth': 200,
    #'gamma': 0.1, # hard coded in the model defn
    'prior': pds_prior_settings
}

# General experiment settings
experiment_settings  = {
    # data
    'data_path': SYNTHETIC_DATA_PATH + 'RG_N=20_r=0.32_dim=2.pt',
    'num_signals': float('inf'),
    'num_train_samples': 50,
    'num_test_samples': 100,

    # inference settings
    'num_chains': 4,
    'num_warmup_samples': 500, # per chain
    'num_posterior_samples': 1000, # per chain

    # posterior samples / point estimates
    'results_path': RESULTS_PATH + 'iid_generalization/', #new_run/',

    # misc
    'random_seed': 0,
    # ... other settings
}
