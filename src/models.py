import jax.nn
import numpy as np
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam, TransformReparam
import numpyro.distributions.constraints as constraints
import jax.numpy as jnp
from typing import Dict

from src.iterates_and_unroll import \
    unroll_dpg, unroll_dpg_channel_vmap, \
    unroll_dpg_mimo, unroll_pds

from typing import Optional


"""
    Base Model
"""

class dpg_bnn:
    @staticmethod
    def forward_pass(
            theta: float,
            delta: float,
            b: float,
            x: jnp.ndarray,
            w_in: jnp.ndarray,
            lam_in: jnp.ndarray,
            depth: int,
            S: jnp.ndarray
            ):
        """
            theta: shape (1,)
            delta: shape (1,)
            b: shape (1,)
            x: shape (num_samples, num_edges)
            w_in: shape (num_samples, num_edges, num_channels) ## num_channels needed/used/cause problems????
            v_in: shape (num_samples, n, num_channels)
            depth: shape ()
            S: shape (n, num_edges)
        """

        w, lam = unroll_dpg(
            x=x,
            w_init=w_in,
            lam_init=lam_in,
            theta=theta,
            num_steps=depth,
            S=S)

        return delta * w - b
    @staticmethod
    def forward_pass_vmap():
        """
            Vmap the forward pass over the posterior samples
        """
        return vmap(dpg_bnn.forward_pass, in_axes=(0, 0, 0, None, None, None, None, None))

    @staticmethod
    def model(
            x: jnp.ndarray,
            y: jnp.ndarray,
            w_init: jnp.ndarray,
            lam_init: jnp.ndarray,
            S: jnp.ndarray,
            depth: int,
            prior_settings: Dict = None
            ):

        num_samples, num_edges = x.shape
        
        theta_loc, theta_scale = prior_settings['theta_loc'], prior_settings['theta_scale']
        theta = numpyro.sample('theta', dist.LogNormal(loc=theta_loc, scale=theta_scale))

        delta_loc, delta_scale = prior_settings['delta_loc'], prior_settings['delta_scale']
        delta = numpyro.sample('delta', dist.LogNormal(loc=delta_loc, scale=delta_scale))

        b_loc, b_scale = prior_settings['b_loc'], prior_settings['b_scale']
        b = numpyro.sample('b', dist.LogNormal(loc=b_loc, scale=b_scale))

        depth = numpyro.deterministic('depth', depth)

        w_final, _ = unroll_dpg(
            x=x,
            w_init=w_init,
            lam_init=lam_init,
            theta=theta,
            num_steps=depth,
            S=S)

        logits = delta * w_final - b

        with numpyro.plate('num_sample_graphs', num_samples, dim=-2):
            # dim = -2 <--> num_samples will be the first dimension of logits/y
            with numpyro.plate('num_edges', num_edges, dim=-1):
                # dim = -1 <--> num_edges will be the second dimension of logits/y
                return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

    @staticmethod
    def ablation_model(
            x: jnp.ndarray,
            y: jnp.ndarray,
            w_init: jnp.ndarray,
            lam_init: jnp.ndarray,
            S: jnp.ndarray,
            depth: int,
            prior_settings: Dict = None):

        """
            Remove Informative Priors.

        """
        num_samples, num_edges = x.shape
        theta_param = numpyro.sample('theta', dist.LogUniform(low=prior_settings['theta_low'],
                                                              high=prior_settings['theta_high']))
        theta = jnp.repeat(theta_param, depth)

        sigma = numpyro.deterministic('sigma', prior_settings['sigma'])
        delta = numpyro.sample('delta', dist.Normal(loc=0.0, scale=sigma))
        b = numpyro.sample('b', dist.Normal(loc=0.0, scale=sigma))

        depth = numpyro.deterministic('depth', depth)

        w_final, _ = unroll_dpg(
            x=x,
            w_init=w_init,
            lam_init=lam_init,
            theta=theta,
            num_steps=depth,
            S=S)

        logits = delta * w_final - b

        with numpyro.plate('num_sample_graphs', num_samples, dim=-2):
            # dim = -2 <--> num_samples will be the first dimension of logits/y
            with numpyro.plate('num_edges', num_edges, dim=-1):
                # dim = -1 <--> num_edges will be the second dimension of logits/y
                return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)

class pds_bnn:
    @staticmethod
    def forward_pass(
            alpha: float,
            beta: float,
            gamma: float,
            b: float,
            x: jnp.ndarray,
            w_in: jnp.ndarray,
            v_in: jnp.ndarray,
            depth: int,
            S: jnp.ndarray
            ):
        w, v = unroll_pds(x, w_in, v_in, alpha, beta, gamma, depth, S)
        return  w - b

    @staticmethod
    def forward_pass_vmap():
        """
            Vmap the forward pass over the posterior samples
        """
        return vmap(pds_bnn.forward_pass, in_axes=(0, 0, 0, 0, None, None, None, None, None))

    @staticmethod
    def model(
            x: jnp.ndarray,
            y: jnp.ndarray,
            w_init: jnp.ndarray,
            v_init: jnp.ndarray,
            S: jnp.ndarray,
            depth: int,
            prior_settings: dict = None):

        assert len(x.shape) == 2
        num_samples, num_edges = x.shape

        # w, lam should be of shape (num_samples, *, num_channels)
        alpha = numpyro.sample('alpha', dist.LogNormal(loc=prior_settings['alpha_loc'], scale=prior_settings['alpha_scale']))
        beta = numpyro.sample('beta', dist.LogNormal(loc=prior_settings['beta_loc'], scale=prior_settings['beta_scale']))
        gamma = numpyro.param('gamma_', init_value=0.1, constraints=constraints.positive)
        numpyro.deterministic('gamma', gamma)
        b = numpyro.sample('b', dist.Normal(loc=0, scale=prior_settings['sigma']))

        depth = numpyro.deterministic('depth', depth)

        w_final, _ = unroll_pds(
            x=x,
            w_init=w_init,
            v_init=v_init,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_steps=depth,
            S=S)
        logits = w_final - b

        with numpyro.plate('num_sample_graphs', num_samples, dim=-2):
            # dim = -2 <--> num_samples will be the first dimension of logits/y
            with numpyro.plate('num_edges', num_edges, dim=-1):
                # dim = -1 <--> num_edges will be the second dimension of logits/y
                return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
            

class dpg_mimo_bnn:
    @staticmethod
    def forward_pass(
            theta: jnp.ndarray,
            delta: jnp.ndarray,
            b: jnp.ndarray,
            x: jnp.ndarray,
            w_in: jnp.ndarray,
            lam_in: jnp.ndarray,
            depth: int,
            S: jnp.ndarray,
            return_raw_output: Optional[bool] = False
            ):
        """
            theta: shape (num_channels, num_channels)
            delta: shape (num_channels,)
            b: shape (1,)
            x: shape (num_samples, num_edges)
            w_in: shape (num_samples, num_edges, num_channels)
            lam_in: shape (num_samples, n, num_channels)
            D: shape (n, num_edges)
            depth: shape ()
        """
        assert x.ndim == 2 and w_in.ndim == lam_in.ndim == 3 and w_in.shape[-1] == lam_in.shape[-1]
        assert theta.ndim == 2
        num_samples, num_edges, num_channels = w_in.shape
        theta = theta.reshape((1, 1, num_channels, num_channels))
        x = x.reshape(num_samples, num_edges, 1, 1)
        w, lam = unroll_dpg_mimo(
            x=x,
            w_init=w_in,
            lam_init=lam_in,
            theta=theta,
            num_steps=depth,
            S=S)

        assert (delta.ndim == 1 and b.ndim == 1)
        delta = delta.reshape((1, 1, num_channels))
        logits = w * delta - b
        #return logits.mean(axis=-1)
        
        if return_raw_output:
            # this is for the stochastic head, which needs access to the raw w and lam
            return w, lam
            #return {'logits': logits.mean(axis=-1), 'w': w, 'lam': lam}
        else:
            return logits.mean(axis=-1)

    @staticmethod
    def model(
            x: jnp.ndarray,
            y: jnp.ndarray,
            w_init: jnp.ndarray,
            lam_init: jnp.ndarray,
            depth: int,
            S: jnp.ndarray,
            num_channels: int,
            prior_settings: dict = None):
        """
            delta is now normally distributed.
        """
        assert len(x.shape) == 2
        num_samples, num_edges = x.shape
        
        # prior is needed in stochastic or MAP
        theta = numpyro.sample('theta', dist.LogNormal(loc=prior_settings['theta_loc'], scale=prior_settings['theta_scale']
                                                       ).expand((num_channels, num_channels)))
        # this is now normal vs lognormal in original DPG
        delta = numpyro.sample('delta', dist.Normal(loc=prior_settings['delta_loc'], scale=prior_settings['delta_scale']
                                                    ).expand((num_channels,) ))
        b = numpyro.sample('b', dist.Normal(loc=prior_settings['b_loc'], scale=prior_settings['b_scale']
                                            ).expand((1,)))
        depth = numpyro.deterministic('depth', depth)

        logits = dpg_mimo_bnn.forward_pass(theta, delta, b, x, w_init, lam_init, depth, S)

        with numpyro.plate('num_sample_graphs', num_samples, dim=-2):
            # dim = -2 <--> num_samples will be the first dimension of logits/y
            with numpyro.plate('num_edges', num_edges, dim=-1):
                # dim = -1 <--> num_edges will be the second dimension of logits/y
                return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)
            

class dpg_mimo_stochastic_head:
    @staticmethod
    def forward_pass(
            theta: jnp.ndarray,
            delta: jnp.ndarray,
            b: jnp.ndarray,
            x: jnp.ndarray,
            w_in: jnp.ndarray,
            lam_in: jnp.ndarray,
            head_depth: int,
            S: jnp.ndarray,
            num_stochastic_channels: int,
            ):
        # We append a stochastic 'head' to the end of the network. The head is simply a set of parallel non-mimo dpg models 
        #  run on the first num_stochastic_channels channels. The remaining channels are not changed. We then combine the 
        # output of the stochastic head with the output of the original network using stochastic \delta vector and b.
        assert x.ndim == 2 and w_in.ndim == lam_in.ndim == 3 and w_in.shape[-1] == lam_in.shape[-1]
        assert theta.ndim == 1 and theta.shape[0] == num_stochastic_channels

        num_samples, num_edges, num_channels = w_in.shape
        theta = theta.reshape((num_stochastic_channels,))
        # only run on the random channels
        w, lam = unroll_dpg_channel_vmap(x,
                                         w_in[:, :, :num_stochastic_channels],
                                         lam_in[:, :, :num_stochastic_channels],
                                         theta, 
                                         head_depth, 
                                         S)

        # concatenate the head output with the non-random channels output (from base network)
        w = jnp.concatenate((w_in[:, :, num_stochastic_channels:], w), axis=-1)
        #lam = np.concatenate((lam_in[:, :, num_random_channels:], lam), axis=-1)

        assert delta.ndim == 1 and b.ndim == 1
        delta = delta.reshape((1, 1, num_channels))
        logits = w * delta - b
        assert logits.shape == (num_samples, num_edges, num_channels)
        return logits.mean(axis=-1)
    
    @staticmethod
    def forward_pass_vmap():
        """
            Vmap the forward pass over the posterior samples
        """
        return vmap(dpg_mimo_stochastic_head.forward_pass, in_axes=(0, 0, 0, None, None, None, None, None, None))

    @staticmethod
    def model(
            x: jnp.ndarray,
            y: jnp.ndarray,
            w_init: jnp.ndarray,
            lam_init: jnp.ndarray,
            S: jnp.ndarray,
            depth: int,
            num_stochastic_channels: int,
            prior_settings: dict = None,
    ):

        assert len(x.shape) == 2
        num_samples, num_edges = x.shape

        # w, lam should be of shape (num_samples, *, num_channels)
        num_channels = w_init.shape[-1]
        assert num_channels == lam_init.shape[-1]

        assert num_stochastic_channels <= num_channels, f"num_stochastic_channels {num_stochastic_channels} > " \
                                                    f"num_channels {num_channels}"
        theta_loc, theta_scale = prior_settings['theta_loc'], prior_settings['theta_scale']
        theta = numpyro.sample('theta', dist.LogNormal(loc=theta_loc, scale=theta_scale).expand((num_stochastic_channels,)))

        delta_loc, delta_scale = prior_settings['delta_loc'], prior_settings['delta_scale'] 
        delta = numpyro.sample('delta', dist.Normal(loc=delta_loc, scale=delta_scale).expand((num_channels,)))
        
        b_loc, b_scale = prior_settings['b_loc'], prior_settings['b_scale']
        b = numpyro.sample('b', dist.Normal(loc=b_loc, scale=b_scale).expand((1,)))

        depth = numpyro.deterministic('depth', depth)

        logits = dpg_mimo_stochastic_head.forward_pass(theta, delta, b, x, w_init, lam_init, depth, S, num_stochastic_channels)

        with numpyro.plate('num_sample_graphs', num_samples, dim=-2):
            # dim = -2 <--> num_samples will be the first dimension of logits/y
            with numpyro.plate('num_edges', num_edges, dim=-1):
                # dim = -1 <--> num_edges will be the second dimension of logits/y
                return numpyro.sample('obs', dist.Bernoulli(logits=logits), obs=y)