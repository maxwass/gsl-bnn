"""
    Script to generate synthetic data. Sample graphs and sample signals smooth on those graphs.

    Construct multiple datasets with the same latent graphs, and varying amounts of noise in the estimatation of the
    euclidean  distance matrices (epistemic uncertainty). Noise is inversely proportional to the number of
    signals used to construct A_O.

    See synthetic_data_tutorial.py for a deeper walk through of each step.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import networkx as nx
import random

from src.utils import edge_density
from src.config import num_samples_to_generate


from src import SYNTHETIC_DATA_PATH



def euclidean_distance(x: np.ndarray):
    """
        x of shape 'num_graphs x num_variables x num_signals'

        returns: euclidean_distances of shape 'num_graphs x num_variables x num_variables'
    """
    assert x.ndim == 3

    num_graphs, num_vertices, num_signals = x.shape

    # rely on internal broadcasting to repeat the diagonal of G across rows and columns
    gram_matrix = x @ np.transpose(x, (0, 2, 1))
    gram_matrix_diag = np.diagonal(gram_matrix, axis1=-1, axis2=-2)[:, :, np.newaxis]
    euclidean_distances = (1/num_signals)*(gram_matrix_diag + np.transpose(gram_matrix_diag, (0, 2, 1)) - 2*gram_matrix)
    return euclidean_distances


def expected_euclidean_distance(covariance: np.ndarray):
    """
        X = [x_1, ..., x_P].
        When samples x_i are drawn iid from N(0, covariance), the expected euclidean distance matrix of X is:
        E[E] = 1diag(covariance)^T + diag(covariance)1^T - 2 * covariance

        covariance of shape 'num_graphs x num_vertices x num_vertices'

    """
    assert covariance.ndim == 3
    laplacians_diag = np.diagonal(covariance, axis1=-1, axis2=-2)[:, :, np.newaxis]  # .unsqueeze(-2)
    expected_euclidean_distances = laplacians_diag + np.transpose(laplacians_diag, (0, 2, 1)) - 2 * covariance
    return expected_euclidean_distances


def euclidean_distance_matrix_convergence(signal_subsets: List[int], euclidean_distance_dict: Dict[str, np.ndarray]):
    """
        Created many datasets, check that the error in approximating the expected E goes down as P increases
    """

    mean_errors, stdv_errors = [], []
    norms = np.sum((euclidean_distance_dict['expected'])**2, axis=(-1, -2))
    for p in signal_subsets:

        errors = ((euclidean_distance_dict[str(p)] - euclidean_distance_dict['expected'])**2).sum((-1, -2)) / norms
        mean_errors.append(errors.mean().item())
        stdv_errors.append(errors.std().item())
    # Plot the means with error bars
    fig, ax = plt.subplots()

    ax.errorbar(signal_subsets, mean_errors, yerr=stdv_errors, markersize=3)
    # Set the x-axis and y-axis scales to log
    ax.set_xscale('log')
    ax.set_yscale('log')
    # Set the plot title
    ax.set_title(r"Convergence of $\lim_{p \rightarrow \infty} \hat{E} = \mathbb{E}[E]$", fontsize=20)
    ax.set_xlabel("Number of Samples: P", fontsize=15)
    ax.set_ylabel(r'$\frac{ \| \mathbf{\mathbb{E}}[E] - \hat{E} \|_F}{ \| \mathbf{\mathbb{E}}[E] \|_F}}$', fontsize=15)

    plt.show(block=False)


def viz_euclidean_distance_matrix_convergence(signal_subsets: List[int],
                                              euclidean_distance_dict: Dict[str, np.ndarray],
                                              adjacencies: np.ndarray,
                                              idx: int):
    """
        Created many datasets, for a single graph, visualize the euclidean distance matrix as P increases
    """

    fig, axes = plt.subplots(ncols=len(signal_subsets)+2)

    max_val = np.max([np.max(euclidean_distance_dict[str(p)][idx]) for p in signal_subsets] + [np.max(euclidean_distance_dict['expected'][idx])])
    axes[-1].imshow(adjacencies[idx], cmap='binary')
    axes[-1].set_title("Adjacency")
    axes[-2].imshow(euclidean_distance_dict['expected'][idx], vmin=0, vmax=max_val)
    axes[-2].set_title("Expected E")
    for i, p in enumerate(signal_subsets):
        axes[i].imshow(euclidean_distance_dict[str(p)][idx], vmin=0, vmax=max_val)
        axes[i].set_title(f"P={p}")

    # remove all xtix and yticks and labels
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.show(block=False)


def visualize_graphs(adjs, num_graphs_viz: int):
    """
        Visualize the adjaceny matrices
    """
    assert adjs.ndim == 3
    _, num_vertices, _ = adjs.shape
    assert num_graphs_viz <= len(adjs), f"Attempting to viz {num_graphs_viz} which is more grpahs than exist in " \
                                        f"input {len(adjs)}"

    fig, axes = plt.subplots(ncols=num_graphs_viz)
    for i in range(num_graphs_viz):
        ax = axes[i]
        ax.imshow(adjs[i], cmap='binary')
        ax.set_xticks([])  # Remove the x-ticks
        ax.set_yticks([])  # Remove the y-ticks
        ax.set_xticklabels([])  # Remove the x-tick labels
        ax.set_yticklabels([])  # Remove the y-tick labels
    plt.tight_layout()
    plt.show(block=False)


def plot_laplacian_eigenvalues(laplacian, num_graphs_viz: int = 10):
    """
        Vizualize the eigenvalues of the Laplacians. Important for the smoothness of the signals generated.
    """
    # Compute the eigenvalues for the first 10 slices of the laplacian tensor
    eigenvalues = [np.linalg.eigvals(laplacian[i]) for i in range(num_graphs_viz)]
    #eigenvalues = [eigvals(slice) for slice in laplacian[:num_graphs_viz]]

    # Extract the real part of the eigenvalues
    eigenvalues = [np.array([e.real for e in slice]) for slice in eigenvalues]

    # Create a histogram plot of the eigenvalues
    plt.figure()
    for i, slice in enumerate(eigenvalues):
        plt.hist(slice, bins=50, alpha=0.3, label=f"Graph {i}")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.title("Distribution of Eigenvalues for Laplacian Matrices")
    plt.legend()
    plt.show(block=False)


"""

    Hyperparameters for the graph sampling.

"""

train, val, test = num_samples_to_generate['train'], num_samples_to_generate['val'], num_samples_to_generate['test'] #50, 50, 100
num_graphs = train + val + test
signal_subsets = [10, 20, 100, 1000, 10000]
seed = 50
vizualize = False


def sample_latent_graphs(num_vertices: int,
                         num_graphs: int,
                         edge_density_floor: float,
                         edge_density_ceiling: float,
                         graph_distribution: str,
                         graph_distribution_params: Dict,
                         seed: int,
                         debug: bool = True
                         ):
    r"""
    inputs:
        num_vertices: how many vertices each sampled graph should have
        num_graphs: how many graphs to sample
        edge_density_floor: lowest edge density allowable
        edge_density_ceiling: highest edge density allowable
        graph_distribution: which random graph distribution to sample from
        graph_distribution_params: Dictionary of parameters specific to random graph distribution
        seed: int random seed. Used directly before any sampling begins.
    """

    # sample binary graphs - store as bool to save space. Add edge weights later if needed.
    adjs = np.zeros((num_graphs, num_vertices, num_vertices), dtype=bool)
    edge_densities = np.zeros(num_graphs)

    # set seeds: numpy and python
    np.random.seed(seed)
    random.seed(seed)
    for i in range(num_graphs):
        G, attempts, edge_density = sample_single_graph(num_vertices=num_vertices,
                                                        edge_density_floor=edge_density_floor,
                                                        edge_density_ceiling=edge_density_ceiling,
                                                        graph_distribution=graph_distribution,
                                                        graph_distribution_params=graph_distribution_params,
                                                    max_attempts=20)
        # plot_graph(G, np.ones(N))
        adjs[i] = np.array(nx.to_numpy_matrix(G), dtype=bool)
        edge_densities[i] = edge_density
        if debug:
            print(f"{i}/{num_graphs} sampled, edge density {edge_density.item():.3f}")
    print(f"Successfully sampled {num_graphs}")
    return adjs


def sample_single_graph(
        num_vertices: int,
        edge_density_floor: float,
        edge_density_ceiling: float,
        graph_distribution: str,
        graph_distribution_params: Dict,
        max_attempts: int = 20):
    r"""
        Given a random 'graph_distribution' with specified 'graph_distribution_params', sample from the
            distribution at most 'max_attempts' times until we find a single graph that is both connected and within the
            edge density limits.

        inputs:
    """
    assert num_vertices > 2, f'number of nodes {num_vertices} must be >3 for sensical graph'
    assert 0 <= edge_density_floor <= edge_density_ceiling <= 1.0, f"Invalid edge density range " \
                                                                   f"[{edge_density_floor}, {edge_density_ceiling}]"
    assert graph_distribution in ['ER', 'RG', 'BA', 'SBM', 'sbm'], f"Unrecognized graph distrib {graph_distribution}"


    attempts = 1
    edge_densities = []
    while True:
        G = {}
        if graph_distribution == 'ER':
            G = nx.fast_gnp_random_graph(n=num_vertices,
                                         p=graph_distribution_params['p'])
        elif graph_distribution == 'RG':
            G = nx.random_geometric_graph(n=num_vertices,
                                          radius=graph_distribution_params['r'],
                                          dim=graph_distribution_params['dim'])
        elif graph_distribution == 'BA':
            # num_vertices = 68, m = 28 -> sparsity of ~1/2
            G = nx.barabasi_albert_graph(n=num_vertices,
                                         m=graph_distribution_params['m'])
        elif graph_distribution in ['SBM', 'sbm']:
            raise NotImplementedError("SBM not implemented yet")
            """
            num_communities, p_in, p_out = \
                graph_distribution_params['num_communities'], graph_distribution_params['p_in'], \
                graph_distribution_params['p_out']
            sizes, prob_matrix = sbm_constructor(num_vertices, num_communities, p_in, p_out)
            probs = []
            for i in range(num_communities):
                probs_from_i = [(p_in if i == j else p_out) for j in range(num_communities)]
                probs.append(probs_from_i)
            G = nx.stochastic_block_model(sizes=sizes,
                                          p=probs,
                                          nodelist=range(sum(sizes)),  # This should ensure consistent node labeling
                                          directed=False,
                                          selfloops=False)
            """

        connected = nx.is_connected(G)
        self_loops = any([G.has_edge(i, i) for i in range(num_vertices)])
        ed = edge_density(np.expand_dims(nx.to_numpy_matrix(G), axis=0))#.unsqueeze(dim=0)
        edge_densities.append(ed.item())
        correct_density = edge_density_floor <= ed <= edge_density_ceiling
        if connected and correct_density and (not self_loops):
            return G, attempts, ed
        elif graph_distribution == "BA": #pref_attach':
            raise ValueError(f'Preferential Attachment model was not able to meet sparsity requirements. Either m is too low, or it is not possible."')
        # print(f'attempt {attempts}: connected? {connected}, sparsity: {sparsity}')
        attempts += 1
        if attempts > max_attempts:
            # print(f'\tfailed after {attempts} attempts, with connected?({connected} & edge density of {ed}')
            raise ValueError(f"Graph Sampling: Sampled > {max_attempts} graphs without connectivity or edge density constraints satisfied. Ave edge density {sum(edge_densities) / len(edge_densities):.3f}. Range [{edge_density_floor:.3f}, {edge_density_ceiling:.3f}]. Adjust parameters.")


def sample_ba_graphs(seed: int = 50):
    """
        Function to sample BA graphs.
    """

    """
        BA graph parameters --> copy from other file!!
    """
    graph_distribution = 'BA'
    num_vertices, edge_density_floor, edge_density_ceiling = 20, 0.0, .99
    distribution_parameters = {'m': 1}
    egde_weights = False
    graph_params = {'graph_distribution': graph_distribution,
                    'edge_density_floor': edge_density_floor,
                    'edge_density_ceiling': edge_density_ceiling,
                    'distribution_parameters': distribution_parameters,
                    'edge_weights': egde_weights}

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

    """
        Optionally add edge weights
    """
    if egde_weights:
        print("Adding edge weights to graphs")
        raise NotImplementedError("Edge weights not yet implemented for RG graphs")

    """
        Unique filname for the data.
    """
    filename = f"{graph_distribution}_N={num_vertices}_m={distribution_parameters['m']}.pt"

    return adjs, filename, graph_params


def sample_er_graphs(num_vertices: int, p: float, seed: int = 50): # n = 20, p = 0.15, 0.5
    """
        Function to sample ER graphs.
    """

    """
        ER graph parameters --> copy from other file!!
    """
    graph_distribution = 'ER'
    num_vertices, edge_density_floor, edge_density_ceiling = num_vertices, 0.0, .99, #0.05, 0.1 # remove edge density constraints?
    distribution_parameters = {'p': p}
    egde_weights = False
    graph_params = {'graph_distribution': graph_distribution,
                    'edge_density_floor': edge_density_floor,
                    'edge_density_ceiling': edge_density_ceiling,
                    'distribution_parameters': distribution_parameters,
                    'edge_weights': egde_weights}

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

    """
        Optionally add edge weights
    """
    if egde_weights:
        print("Adding edge weights to graphs")
        raise NotImplementedError("Edge weights not yet implemented for RG graphs")

    """
        Unique filname for the data.
    """
    filename = f"{graph_distribution}_N={num_vertices}_p={distribution_parameters['p']}.pt"

    return adjs, filename, graph_params


def sample_rg_graphs(r: float, seed: int = 50): # r = 0.32, 0.5
    """
        Function to sample RG graphs.
    """

    """
        RG graph parameters --> copy from other file!!
    """
    graph_distribution = 'RG'
    num_vertices, edge_density_floor, edge_density_ceiling = 20, 0.0, .99
    distribution_parameters = {'r': r, 'dim': 2}
    egde_weights = False
    graph_params = {'graph_distribution': graph_distribution,
                    'edge_density_floor': edge_density_floor,
                    'edge_density_ceiling': edge_density_ceiling,
                    'distribution_parameters': distribution_parameters,
                    'edge_weights': egde_weights}

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

    """
        Optionally add edge weights
    """
    if egde_weights:
        print("Adding edge weights to graphs")
        raise NotImplementedError("Edge weights not yet implemented for RG graphs")

    """
        Unique filname for the data.
    """
    filename = f"{graph_distribution}_N={num_vertices}_r={distribution_parameters['r']}_dim={distribution_parameters['dim']}.pt"

    return adjs, filename, graph_params

"""

    Entry point to sample graphs and signals.
    Uncomment the desired graph distribution to sample from.

"""

#adjacencies, filename, graph_params = sample_ba_graphs()
#adjacencies, filename, graph_params = sample_er_graphs(num_vertices=20, p=.5)
adjacencies, filename, graph_params = sample_rg_graphs(r=.32)
#adjacencies, filename, graph_params = sample_rg_graphs(r=.5)


"""
    Check for the existence of the synthetic data directory, or the existence of the file.

"""

path2file = SYNTHETIC_DATA_PATH + filename

# check if the directory exists and if the file already exists
if not os.path.isdir(SYNTHETIC_DATA_PATH):
    print(f"Directory {SYNTHETIC_DATA_PATH} does not exist. Please create it.")
    raise ValueError(f"Directory {SYNTHETIC_DATA_PATH} does not exist. Please create it.")
elif os.path.isfile(path2file):
    raise ValueError(f"Filname {filename} already exists! Manually remove it or change filename.")


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


num_graphs, num_vertices = adjacencies.shape[0:2]

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
print(f'Sanity Check: manual same as np.pinv: ', np.allclose(normalized_laplacians_square_root_pinv, normalized_laplacians_square_root_pinv))

# sanity check: compare L^{\dagger} with pinv function is the same as \sqrt{L^{\dagger}} * \sqrt{L^{\dagger}}
normalized_laplacians_pinv = normalized_laplacians_square_root_pinv @ normalized_laplacians_square_root_pinv
reconstruction_error = ((normalized_laplacians_pinv - np.linalg.pinv(normalized_laplacians))**2).sum(axis=(-1, -2))
print("Maximum reconstruction error between L^-0.5 @ L^-0.5 and pinv(L): {:.6f}".format(reconstruction_error.max().item()))

# now apply filter to (non-smooth) signals: x = \sqrt(L^{\dagger}} x_0
np.random.seed(seed)
num_signals = signal_subsets[-1]
x_0_all = np.random.normal(size=(num_graphs, num_vertices, num_signals))

expected_e = expected_euclidean_distance(normalized_laplacians_pinv)
euclidean_distance_dict = {'expected': expected_e}

print(f"Generating datasets with {signal_subsets} signals...")
for p in signal_subsets:
    r"""
        Create a dataset for each number of sampled signals.
    """
    print(f"\t{p} signals")
    # only use the first p signals
    x_0 = x_0_all[:, :, :p]
    smooth_signals = normalized_laplacians_square_root_pinv @ x_0

    e_hat = euclidean_distance(smooth_signals)
    euclidean_distance_dict[str(p)] = e_hat
    """
    if vizualize:
        plot_signal_energy_distribution(x=smooth_signals, eigenvectors=eigenvectors, eigenvalues=eigenvalues,
                                        num_signals_to_plot=10 if p>10 else p)
        euclidean_distance_plot(e=e_hat, expected_e=expected_e, adjs=adjacencies, num_signals=p)
    """

if vizualize:
    euclidean_distance_matrix_convergence(signal_subsets=signal_subsets,
                                          euclidean_distance_dict=euclidean_distance_dict)
    viz_euclidean_distance_matrix_convergence(signal_subsets=[3, 10, 100, 1000, 10000],
                                              euclidean_distance_dict=euclidean_distance_dict,
                                              adjacencies=adjacencies, idx=0)


"""
    Save a dictionary consisting of the sampled graphs, the graph parameters, and the euclidean distance matrices.
"""
data = {'adjacencies': adjacencies, **euclidean_distance_dict, 'graph_params': graph_params}

"""
    Save the data to file
"""

print(f"Attemping to save data to {path2file}")
with open(path2file, 'wb') as f:
    pickle.dump(data, f)
print(f"Saved data to {path2file}")
# Load the dictionary from the file to ensure saved correctly
with open(path2file, 'rb') as f:
    loaded_data = pickle.load(f)


