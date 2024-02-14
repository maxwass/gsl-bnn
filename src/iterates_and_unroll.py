from jax import jit, vmap
import jax.numpy as jnp
import jax
import numpy as np
from functools import partial


"""
    Inner 'step' functions
"""

def step_dpg_alpha_beta(
        x: jnp.ndarray,
        w_prev: jnp.ndarray,
        lam_prev: jnp.ndarray,
        alpha: float,
        beta: float,
        S: jnp.ndarray
):
    """
    The inner loop of the \alpha, \beta parametrized Dual Proximal Gradient Descent algorithm.

    Arguments:
        x is the input data
        *_prev are the outputs from the last iteration
        S is linear transform mapping vectorized adjacency into degree vector
        alpha and beta are the regularization parameters
    """
    num_edges = x.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))

    # inner = S*w_prev - (n-1)/beta * lam_prev
    dw = jnp.matmul(w_prev, S.T)
    inner = dw - ((n - 1) / beta) * lam_prev

    # lam = -s * inner + s * sqrt[ inner^2 + 4(n-1)alpha/beta*1]
    # NOTE: If using pytorch: **torch**.sqrt() has posed issue (on m1-mac). Use .pow(0.5) instead
    s = (beta / (2 * (n - 1)))
    lam = -s * inner + s * jnp.sqrt(jnp.square(inner) + 4 * (n - 1) * jnp.true_divide(alpha, beta))

    # w = 1/(2*beta) * relu( S^T lam - 2x )
    w_ = jnp.maximum(jnp.matmul(lam, S) - 2 * x, 1e-16) #8)
    w = 1 / (2 * beta) * w_
    return w, lam


def step_dpg(
        x: jnp.ndarray,
        w_prev: jnp.ndarray,
        lam_prev: jnp.ndarray,
        theta: float,
        S: jnp.ndarray
):
    """
    Performs the inner loop of the theta-delta parametrized Dual Proximal Gradient Descent (Algorithm 1).

    Args:
        x (jnp.ndarray): The input data with shape (num_samples, num_edges).
        w_prev (jnp.ndarray): The previous iteration's edge weight matrix with shape (num_samples, num_edges).
        lam_prev (jnp.ndarray): The previous iteration's dual vector with shape (num_samples, n).
        theta (float): The regularization parameter.
        S (jnp.ndarray): Linear transform matrix mapping vectorized adjacency into degree vector with shape (n, num_edges).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the updated edge weight matrix and dual vector.

    Notes:
        - The \delta parameter is not used in the inner step; it is applied to the output after D steps.
    """
    num_edges = x.shape[-1]
    n = int(0.5*(np.sqrt(8 * num_edges + 1) + 1))

    # u_k = S*w_{k-1} - (n-1) * lam_{k-1}
    u = jnp.matmul(w_prev, S.T) - (n - 1) * lam_prev

    # lam_k = -s * (u_k -  sqrt[ u_k^2 + 4(n-1)*1 ])
    # NOTE: If using pytorch: **torch**.sqrt() has posed issue (on m1-mac). Use .pow(0.5) instead
    s = 1 / (2 * (n - 1))
    lam = -s * (u - jnp.sqrt(jnp.square(u) + 4 * (n - 1)))

    w = jnp.maximum(0.5 * jnp.matmul(lam, S) - theta * x, 1e-8) #8)
    return w, lam


def step_dpg_mimo(
        x: jnp.ndarray,
        w_prev: jnp.ndarray,
        lam_prev: jnp.ndarray,
        theta: jnp.ndarray,
        S: jnp.ndarray
):
    """
        x.shape = (num_samples, num_edges, 1, 1)
        w_prev.shape = (num_samples, num_edges, num_channels_in)
        lam_prev.shape = (num_samples, n, num_channels_in)
        theta.shape = (1, 1, num_channels_out, num_channels_in)
        S.shape = (num_edges, n)
    """
    n = lam_prev.shape[1]
    num_samples, num_edges = x.shape[:2]
    num_channels_out, num_channels_in = theta.shape[2:]

    # u_k = S*w_{k-1} - (n-1) * lam_{k-1}, shape = (num_samples, n, num_channels_in)
    u = jnp.einsum('teC,Ne->tNC', w_prev, S) - (n - 1) * lam_prev

    # lam_k = -s * (u_k -  sqrt[ u_k^2 + 4(n-1)*1 ]), shape = (num_samples, n, num_channels_in)
    # NOTE: If using pytorch: **torch**.sqrt() has posed issue (on m1-mac). Use .pow(0.5) instead
    # ensure lam of shape (num_samples, num_channels, n)
    s = 1 / (2 * (n - 1))
    lam = -s * (u - jnp.sqrt(jnp.square(u) + 4 * (n - 1)))

    # apply S.T to lam to get shape (num_samples, num_edges,  num_channels_in)
    w_ = jnp.einsum('ijk,jl->ilk', .5 * lam, S)
    # add extra dimension to w_ to get shape (num_samples, num_edges, 1, num_channels_in)
    w_ = w_[..., None, :]
    # apply theta to w_ to get shape (num_samples, num_edges, num_channels_out, num_channels_in)
    w = jnp.maximum(w_ - theta * x, 1e-8)
    # reduce over the num_channels_in dimension to get shape (num_samples, num_edges, num_channels_out)
    w = jnp.mean(w, axis=-1)
    return w, lam


def step_pds(
        x: jnp.ndarray,
        w_prev: jnp.ndarray,
        v_prev: jnp.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        S: jnp.ndarray,
        eps: float = 1e-16
):
    """
    Performs the inner loop of the alpha-beta parametrized Proximal Dual Splitting (PDS - Algorithm 2).

    Args:
        x (jnp.ndarray): The input data with shape (num_samples, num_edges).
        w_prev (jnp.ndarray): The previous iteration's weight matrix with shape (num_samples, num_edges).
        v_prev (jnp.ndarray): The previous iteration's dual vector with shape (num_samples, n).
        alpha (float): A regularization parameter.
        beta (float): A regularization parameter.
        gamma (float): A step size parameter.
        eps is a small positive number to avoid disconnected graph
        S (jnp.ndarray): Linear transform matrix mapping vectorized adjacency into degree vector with shape (n, num_edges).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: A tuple containing the updated edge weight matrix and dual vector.

    Notes:
        - The \delta parameter is not used in the inner step; it is applied to the output after D steps.
    """
    y1 = w_prev - gamma * (2 * beta * w_prev + 2 * x + jnp.matmul(v_prev, S)) #v_prev.matmul(S))  # torch.matmul(v, S)) # == S.T v
    y2 = v_prev + gamma * jnp.matmul(w_prev, S.T) #w_prev.matmul(S.T)  # torch.matmul(w, S.T) # == S w

    p1 = jax.nn.relu(y1)  #

    # CLAMPING (placing floor below values) CRITICAL FOR UNROLLING PERFORMANCE, but not included (or needed?) in model based
    up = y2 ** 2 + 4 * gamma * alpha
    up = jnp.clip(up, a_min=eps) if eps is not None else up
    #up = torch.clamp(up, min=eps) if eps is not None else up

    p2 = (y2 - jnp.sqrt(up)) / 2

    q1 = p1 - gamma * (2 * beta * p1 + 2 * x + jnp.matmul(p2, S)) # p2.matmul(S))  # torch.matmul(p2, S)) # == S.T*p2
    q2 = p2 + gamma * jnp.matmul(p1, S.T)  # == S*p1

    w = w_prev - y1 + q1
    v = v_prev - y2 + q2

    return w, v



"""
    Unrollings: Multiple 'steps'
"""

@partial(jit, static_argnames=['num_steps'])
def unroll_dpg_alpha_beta(
        x: jnp.ndarray,
        w_init: jnp.ndarray,
        lam_init: jnp.ndarray,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        num_steps: int,
        S: jnp.ndarray,
):
    f = partial(step_dpg_alpha_beta, S=S, x=x)

    def body_fun(i, carry):
        w, lam = carry
        w, lam = f(w_prev=w, lam_prev=lam, alpha=alpha[i], beta=beta[i])
        return w, lam

    w, lam = jax.lax.fori_loop(0, num_steps, body_fun, (w_init, lam_init))
    return w, lam


@partial(jit, static_argnames=['num_steps'])
def unroll_dpg(
        x: jnp.ndarray,
        w_init: jnp.ndarray,
        lam_init: jnp.ndarray,
        theta: float, #jnp.ndarray,
        num_steps: int,
        S: jnp.ndarray
):
    """
    Unrolls the Dual Proximal Gradient descent algorithm (DPG - Algorithm 1) for a specified number of steps.

    Args:
        x (jnp.ndarray): Input data with shape (num_samples, num_edges).
        w_init (jnp.ndarray): Initial weight matrix with shape (num_samples, num_edges).
        lam_init (jnp.ndarray): Initial lambda vector with shape (num_samples, n).
        theta (float): Regularization parameter.
        num_steps (int): Number of steps to unroll the algorithm.
        S (jnp.ndarray): Linear transform matrix with shape (n, num_edges).

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The final edge weight matrix and dual vector after unrolling.

    Notes:
        - This function utilizes `jax.lax.fori_loop` for iteration, which benefits from both compiled and non-compiled runtime performance improvements and more reasonable JIT compile times.
        - Backward gradients are not supported due to the use of `fori_loop`. For operations requiring backward gradients, consider alternatives or adjustments that enable forward-mode differentiation.
        - To use this with NumPyro, set the `forward_mode_differentiation=True` flag in the inference kernel.
    """
    f = partial(step_dpg, x=x, theta=theta, S=S)

    def body_fun(i, carry):
        w, lam = carry
        w, lam = f(w_prev=w, lam_prev=lam)
        return w, lam

    w, lam = jax.lax.fori_loop(0, num_steps, body_fun, (w_init, lam_init))
    return w, lam


# now w_init, lam_init, and theta have multiple channels. vmap over each of their respective channel dimensions
unroll_dpg_channel_vmap = jax.vmap(unroll_dpg, in_axes=(None, -1, -1, 0, None, None), out_axes=(-1, -1))


@partial(jit, static_argnames=['num_steps'])
def unroll_dpg_mimo(
        x: jnp.ndarray,
        w_init: jnp.ndarray,
        lam_init: jnp.ndarray,
        theta: jnp.ndarray,
        num_steps: int,
        S: jnp.ndarray
):
    """
        Note: theta is now (1, num_channels_out, num_channels_in, 1). Different than direct unrolling.
    """
    """
    assert len(theta.shape) == 4 and len(theta.shape) == 4
    num_samples, _, _, num_edges = x.shape
    num_channels_in, num_channels_out = theta.shape[1:3]
    n = lam_init.shape[-1]
    assert delta.shape == (1, num_channels_out, 1), f'delta.shape = {delta.shape}, should be {(1, num_channels_out, 1)}'
    assert w_init.shape == (num_samples, num_channels_in,
                            num_edges), f'w_prev.shape = {w_init.shape}, should be {(num_samples, num_channels_in, num_edges)}'
    assert lam_init.shape == (num_samples, num_channels_in,
                              n), f'lam_prev.shape = {lam_init.shape}, should be {(num_samples, num_channels_in, n)}'
    assert theta.shape == (1, num_channels_out, num_channels_in,
                           1), f'theta.shape = {theta.shape}, should be {(1, num_channels_out, num_channels_in, 1)}'
    """

    f = partial(step_dpg_mimo, x=x, theta=theta, S=S)

    def body_fun(i, carry):
        w, lam = carry
        w, lam = f(w_prev=w, lam_prev=lam)
        return w, lam

    w, lam = jax.lax.fori_loop(0, num_steps, body_fun, (w_init, lam_init))
    # take mean over weighted sum of channels
    return w, lam

@partial(jit, static_argnames=['num_steps'])
def unroll_pds(
        x: jnp.ndarray,
        w_init: jnp.ndarray,
        v_init: jnp.ndarray,
        alpha: jnp.ndarray,
        beta: jnp.ndarray,
        gamma: jnp.ndarray,
        num_steps: int,
        S: jnp.ndarray,
):
    f = partial(step_pds, x=x, alpha=alpha, beta=beta, gamma=gamma, S=S)

    def body_fun(i, carry):
        w, v = carry
        w, v = f(w_prev=w, v_prev=v)
        return w, v

    w, v = jax.lax.fori_loop(0, num_steps, body_fun, (w_init, v_init))
    return w, v
