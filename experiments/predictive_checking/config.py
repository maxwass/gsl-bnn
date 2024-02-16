# config.py

from src import SYNTHETIC_DATA_PATH, RESULTS_PATH
from src.config import original_prior, altered_prior
import jax.numpy as jnp

# Hyperparameters
dpg_hyperparameters = {
    'depth': 30,
}

# General experiment settings
experiment_settings  = {
    # data
    'data_path': SYNTHETIC_DATA_PATH + 'RG_N=20_r=0.32_dim=2.pt',
    'num_signals': float('inf'),
    'num_train_samples': 50,
    'num_val_samples': 5,
    'num_test_samples': 100,

    # prior predictive checking
    'num_prior_samples': 10000,

    # inference settings
    'num_chains': 4,
    'num_warmup_samples': 500, # per chain
    'num_posterior_samples': 1000, # per chain

    # posterior samples / point estimates
    'results_path': RESULTS_PATH + 'predictive_checking/',

    # misc
    'random_seed': 0,
    # ... other settings
}