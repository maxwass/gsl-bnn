# config.py

from src import  RESULTS_PATH
from src.config import altered_prior
import jax.numpy as jnp

# Hyperparameters
dpg_hyperparameters = {
    'depth': 200,
    'prior': altered_prior
}

# General experiment settings
experiment_settings  = {
    # data
    # posterior samples / point estimates
    'results_path': RESULTS_PATH + 'mnist/',
    
    # inference settings
    'num_chains': 4,
    'num_warmup_samples': 100, #500, # per chain
    'num_posterior_samples': 300, #1000, # per chain

    # misc
    'random_seed': 0,
    # ... other settings
}
