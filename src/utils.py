"""
    Utility functions useful for graph structure learning methods
"""

from __future__ import print_function, division
from jax import vmap
import jax.numpy as jnp
import numpy as np
from math import sqrt


def edge_density(a: jnp.ndarray):
    assert a.ndim in [2, 3], f"edge_density takes vectorized adjacencies (2D) or stack of n x n adjs (3D). Input shape {a.shape} has more then 3 dims."

    if a.ndim == 2:
        """
            a[i] are all edges for sample i.
        """
        bs, total_edges = a.shape
        assert bs != total_edges, f'Single matrix not allowed. Please add leading dimension'
        a_abs = jnp.abs(a) if a.dtype != jnp.bool_ else a
        return jnp.sum(a_abs > 0, axis=1) / total_edges
    else:
        """
            a[i] is an 'n x n' adjacency matrix. Extract the upper triangular portion of adjacency.
        """
        assert a.shape[-1] == a.shape[-2]

        a_vec = adj2vec(a)
        a_vec_abs = jnp.abs(a_vec) if a_vec.dtype != jnp.bool_ else a_vec
        bs, total_edges = a_vec.shape
        return jnp.sum(a_vec_abs > 0, axis=1) / total_edges


edge_density_vmap = vmap(edge_density, in_axes=(0,), out_axes=0)


def count_connected_components(a: jnp.ndarray):
    """
        Compute the number of connected components in each adjacency matrix in a.
    """

    if a.ndim == 2:
        # assume a is a batch of vectorized adjacency matrices
        # a.shape = [batch_size, n(n-1)/2]
        n = int(0.5 * (jnp.sqrt(8 * a.shape[-1] + 1) + 1))
        a_ = vec2adj(a, n=n)
    else:
        assert a.ndim == 3, f"count_connected_components takes vectorized adjacencies (2D) or stack of n x n adjs (3D). Input shape {a.shape} has more then 3 dims."
        n = a.shape[-1]
        a_ = a

    batch_size = a_.shape[0]
    diags = jnp.sum(a_, axis=-1)

    # Create an empty array of shape (batch_size, n, n)
    D = jnp.zeros((batch_size, n, n), dtype=diags.dtype)

    # Create an array of diagonal indices of shape (n,) using jnp.arange and pass it to jnp.diag_indices
    # to get the indices of the diagonal elements of a matrix of shape (n, n)
    diag_indices = jnp.diag_indices(n)

    # Use jnp.at to set the diagonal of each matrix in the batch to the corresponding row of the diags array
    D = D.at[:, diag_indices[0], diag_indices[1]].set(diags)
    laplacian = D - a_

    # compute eigenvalues of laplacians
    eigenvalues = jnp.linalg.eigvalsh(laplacian)

    # sort eigenvalues in each row
    eigenvalues_sort = jnp.sort(eigenvalues, axis=-1)

    # count the number of eigenvalues that are 0 in each. Connected graphs have exactly 1 eigenvalue that is 0. Need to
    # account for numerical error.
    return (jnp.abs(eigenvalues) < 1e-4).sum(axis=-1)


count_connected_components_vmap = vmap(count_connected_components, in_axes=(0,), out_axes=0)


def vec2adj(v: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Takes a batch of vectorized adjacency matrices and returns a batch of full symmetric adjacency matrices.

    Args:
        v: A JAX array of shape (batch_size, num_edges) containing a batch of vectorized adjacency matrices.
        n: An integer representing the number of nodes in the adjacency matrices.

    Returns:
        A JAX array of shape (batch_size, n, n) containing a batch of full symmetric adjacency matrices.
    """
    # Initialize the adjacency matrix
    adj = jnp.zeros((v.shape[0], n, n))

    # Compute the indices of the upper triangular part (not including diagonal)
    indices = jnp.triu_indices(n, k=1)

    # Fill in the upper triangular part (not including diagonal) with the values from v
    adj = adj.at[:, indices[0], indices[1]].add(v)

    # Construct a symmetric adjacency matrix by adding its transpose to itself
    symm = adj + adj.transpose((0, 2, 1))

    return symm


def adj2vec(A: jnp.ndarray, offset=1):
    # offset = 1 : by default dont include diagonal
    assert A.ndim == 3 and (
                A.shape[-1] == A.shape[-2]), f'adj2vec: input is not stack of matrices: input shape {A.shape}'
    batch_size, n = A.shape[:2]
    idxs = jnp.triu_indices(n=n, m=n, k=offset)  # ignore diagonal
    v = A[:, idxs[0], idxs[1]]
    return v


def degrees_from_upper_tri(n: int):
    """
        Returns a linear transformation D such that Dw = A1, where A is the adjacency matrix of an undirected graph
        without self loops and w is the upper triangular part of A (without diagonal).

        D is of shape 'n x (n(n-1)/2)'.

    """
    dtype = np.float32

    # total number of possible edges in undirected graph without self loops n(n-1)/2
    NumCol = int((n * (n - 1) / 2))
    I = np.zeros(NumCol, dtype=dtype)
    J = np.zeros(NumCol, dtype=dtype)
    k = 0
    for i in np.arange(n - 1):
        I[k:k + n - i - 1] = np.arange(i + 1, n)
        k = k + n - i - 1
    k = 0
    for i in np.arange(n - 1):
        J[k:k + n - i - 1] = i
        k = k + n - i - 1
    Jidx = np.concatenate((I, J), axis=0)
    Iidx = np.tile(np.arange(NumCol, dtype=dtype), 2)

    Jidx, Iidx = Jidx.astype(np.int32), Iidx.astype(np.int32)
    Dt = np.zeros((NumCol, n), dtype=dtype)
    Dt[Iidx, Jidx] = 1
    D = Dt.T
    return D


def num_edges2num_nodes(num_edges: int):
    assert num_edges > 2
    # https://www.wolframalpha.com/input/?i=solve+y%3D+%28x*%28x-1%29%2F2%29
    inv = sqrt(8 * num_edges + 1) + 1
    num_nodes = int(0.5 * inv)
    assert num_nodes * (num_nodes - 1) // 2 == num_edges
    return num_nodes


# define __main__ function to run tes
if __name__ == '__main__':
    # sample symmetric adjacency matrices with non-negative entries and no self-loops
    for i in range(10):
        n = 100
        num_samples = 100
        A = np.random.rand(num_samples, n, n)
        A = A * (A < 0.05)
        A = A * (np.eye(n) == 0)
        A = A + np.transpose(A, (0, 2, 1))

        """
            Test conversion between vectorized and adjacency matrices
        """
        v = adj2vec(A)
        A_ = vec2adj(v, n=n)
        assert np.allclose(A, A_)

        jnp_edge_density = edge_density(A)
        jnp_cc = count_connected_components(A)