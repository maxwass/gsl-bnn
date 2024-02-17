import jax.numpy as jnp
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import jit


from functools import partial
import time

from src.utils import adj2vec, vec2adj, degrees_from_upper_tri

from src.utils import edge_density
from src.synthetic_data_generator import sample_latent_graphs, \
    expected_euclidean_distance, euclidean_distance, \
    euclidean_distance_matrix_convergence, viz_euclidean_distance_matrix_convergence, \
    visualize_graphs, plot_laplacian_eigenvalues
from experiments.scaling.config import experiment_settings


"""
    Script to generate synthetic data. Sample graphs and sample signals smooth on those graphs.

    Construct multiple datasets with the same latent graphs, and varying amounts of noise in the estimatation of the
    euclidean  distance matrices (epistemic uncertainty). Noise is inversely proportional to the number of
    signals used to construct A_O.

    See synthetic_data_tutorial.py for a deeper walk through of each step.
"""

vizualize = False

"""
    Custom graph distributions
    - smaller for faster prototyping
    - 
"""

def sample_er_graphs(num_graphs: int, num_vertices: int, p: float, seed: int = 50): # n = 20, p = 0.15, 0.5
    """
        Function to sample ER graphs.
    """

    """
        ER graph parameters --> copy from other file!!
    """
    graph_distribution = 'ER'
    edge_density_floor, edge_density_ceiling = 0.0, .99, #0.05, 0.1 # remove edge density constraints?
    distribution_parameters = {'p': p}

    """
        sample the graphs and place into tensor
    """
    adjs = sample_latent_graphs(num_vertices=num_vertices,
                                num_graphs=num_graphs,
                                edge_density_floor=edge_density_floor,
                                edge_density_ceiling=edge_density_ceiling,
                                graph_distribution=graph_distribution,
                                graph_distribution_params=distribution_parameters,
                                seed=seed, debug=False)

    return adjs

"""
    Compute laplacians and normalize.
"""

def diag_embed(diags: np.ndarray):
    """
        diags: np.ndarray of shape (batch_size, n)

        returns: np.ndarray of shape (batch_size, n, n) where the diagonal elements are diags
    """
    mask = np.eye(diags.shape[-1], dtype=bool)
    diag = diags[:, :, np.newaxis]
    return diag * mask[np.newaxis, :, :]

def sample_er_graphs_and_euclidean_distance(num_graphs, num_vertices, random_seed, p=.25):
    adjacencies = sample_er_graphs(num_graphs, num_vertices, p, random_seed) #num_vertices=50, p=.25)

    D = diag_embed(np.sum(adjacencies, axis=1))
    laplacians = D - adjacencies.astype(np.float32)
    # normalize each laplacian slice by its max eigenvalue
    max_eigvals = np.linalg.eigvalsh(laplacians).max(axis=1)
    normalized_laplacians = laplacians / max_eigvals[:, np.newaxis, np.newaxis]

    if vizualize:
        visualize_graphs(adjacencies, num_graphs_viz=3)
        plot_laplacian_eigenvalues(normalized_laplacians, num_graphs_viz=3)
        fig, ax = plt.subplots(1, 1)
        plt.hist(edge_density(adjacencies), bins=50)
        plt.title("Edge density distribution")
        plt.show(block=False)

    # find the eigenvalues and eigenvectors of the laplacians. Note they are real symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacians)
    # we know the smallest eigenvalue is exactly 0, but numerical precision/error makes it ~1e-8
    eigenvalues[:, 0] = 0.0

    # apply the pseudo-inverse function to the eigenvalues
    eigenvalues_pinv = np.where(eigenvalues > 1e-6, 1 / eigenvalues, 0.0)
    # compute \sqrt{L^{\dagger}} = V \sqrt{\Lambda^{\dagger}} V^T
    normalized_laplacians_square_root_pinv = eigenvectors @ diag_embed(np.sqrt(eigenvalues_pinv)) @ eigenvectors.transpose(0, 2, 1)
    #print(f'Sanity Check: manual same as np.pinv: ', np.allclose(normalized_laplacians_square_root_pinv, normalized_laplacians_square_root_pinv))

    # sanity check: compare L^{\dagger} with pinv function is the same as \sqrt{L^{\dagger}} * \sqrt{L^{\dagger}}
    normalized_laplacians_pinv = normalized_laplacians_square_root_pinv @ normalized_laplacians_square_root_pinv
    reconstruction_error = ((normalized_laplacians_pinv - np.linalg.pinv(normalized_laplacians))**2).sum(axis=(-1, -2))
    #print("Maximum reconstruction error between L^-0.5 @ L^-0.5 and pinv(L): {:.6f}".format(reconstruction_error.max().item()))

    # now apply filter to (non-smooth) signals: x = \sqrt(L^{\dagger}} x_0

    expected_e = expected_euclidean_distance(normalized_laplacians_pinv)
    euclidean_distance_dict = {'expected': expected_e}
    return adjacencies, expected_e


# first attempt to load from datasets.pickle. If doesn't exist, then sample and save.
data_dir = experiment_settings['data_path']

try:
    graph_sizes, adjacencies, euclidean_distances = [], [], []
    with open(experiment_settings['data_path'], 'rb') as handle:
        print(f'loading datasets from {experiment_settings["data_path"]}.')
        datasets = pickle.load(handle)
        for n, d in datasets.items():
            print(f'num_vertices: {n}')
            graph_sizes.append(n)
            adjacencies.append(d['adjacencies'])
            euclidean_distances.append(d['euclidean_distances'])
except:
    print(f"Scaling datasets not found at {experiment_settings['data_path']}. Sampling and saving. Can take a while as we sample large graphs.")
    adjacencies, euclidean_distances = [], []
    datasets = {}
    for n in experiment_settings['graph_sizes']:
        if n not in datasets:
            print(f'sampling graph sizes {n}')
            a, e = sample_er_graphs_and_euclidean_distance(experiment_settings['num_graphs'], n, experiment_settings['random_seed'], experiment_settings['p'])
            adjacencies.append(a)
            euclidean_distances.append(e)
            datasets[n] = {'adjacencies': a, 'euclidean_distances': e}

    with open(os.path.join(data_dir, 'datasets.pkl'), 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)