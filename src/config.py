import jax.numpy as jnp

"""
    For Synthetic Data Generation
"""
#num_samples_to_generate = {'train': 50, 'val': 50, 'test': 100}
num_samples_to_generate = {'train': 4000, 'val': 500, 'test': 500}


# initial values of w and lam
w_init_scale, lam_init_scale = 0.5, 17

def initial_optimization_variable_values(num_train_samples, num_channels, num_edges, n):
    w = 0.5*jnp.ones((num_train_samples, num_edges, num_channels))
    lam = 17*jnp.ones((num_train_samples, n, num_channels))
    return w, lam


# prior constructed using convergent iterations on some sample data
original_prior = {'theta_loc': jnp.log(10 ** (0)),
                  'theta_scale': jnp.sqrt(4),
                  'delta_loc': jnp.log(10 ** 2),
                  'delta_scale': jnp.sqrt(2),
                  'b_loc': jnp.log(10 ** 0),
                  'b_scale': jnp.sqrt(2)}

# adjusted prior AFTER prior predictive check correction 
altered_prior = {'theta_loc': jnp.log(10 ** (-.5)),
                 'theta_scale': jnp.sqrt(4),
                 'delta_loc': jnp.log(10 ** 2),
                 'delta_scale': jnp.sqrt(2),
                 'b_loc': jnp.log(10 ** 0),
                 'b_scale': jnp.sqrt(2)}

# pds prior settings
pds_prior_settings = {
    'alpha_loc': jnp.log(10**0),
    'alpha_scale': 10,
    'beta_loc': jnp.log(10**0),
    'beta_scale': 10,
    'sigma': 1e3
}
