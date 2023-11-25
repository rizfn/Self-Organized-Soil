import numpy as np
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from nutrient_utils import run_stochastic


def calculate_cluster_sizes(soil_lattice_data, target_site=1):
    """Calculate the cluster sizes for each timestep.

    Parameters
    ----------
    soil_lattice_data : ndarray
        Array of soil_lattice data for each timestep.
    target_site : int
        The value of the site that is to be considered part of the cluster.
    
    Returns
    -------
    cluster_sizes : ndarray
        Array of cluster sizes for each timestep.
    """
    cluster_sizes = []
    for i in range(len(soil_lattice_data)):
        m = soil_lattice_data[i] == target_site
        lw, num = ndimage.label(m)
        sizes = ndimage.sum(m, lw, index=np.arange(num + 1))
        cluster_sizes.append(sizes)
    return cluster_sizes


def main():

    # initialize the parameters
    steps_per_latticepoint = 1_000  # number of time steps for each lattice point
    L = 200  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**2  # number of bacteria moves
    rho = 1  # reproduction rate
    theta = 0.14  # death rate
    sigma = 0.2  # soil filling rate
    delta = 0 # nutrient decay rate

    steps_to_record = np.arange(n_steps//2, n_steps, 20 * L**2, dtype=np.int32)

    # run the simulation
    soil_lattice_data = run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record)

    # calculate the cluster sizes
    empty_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 0))
    nutrient_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 1))
    soil_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 2))
    worm_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 3))

    # histogram and plot all the cluster sizes
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(max(empty_cluster_sizes), max(nutrient_cluster_sizes), max(soil_cluster_sizes), max(worm_cluster_sizes))
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)
    # Calculate histograms and plot
    for cluster_sizes, label in zip([empty_cluster_sizes, nutrient_cluster_sizes, soil_cluster_sizes, worm_cluster_sizes], ['empty sites', 'nutrient sites', 'soil sites', 'worm sites']):
        hist, edges = np.histogram(cluster_sizes, bins=bins, density=False)
        bin_widths = np.diff(edges)
        hist = hist / bin_widths  # Normalize by bin width
        plt.plot(edges[:-1], hist, 'x', label=label)

    plt.title(f'{L=}, {rho=}, {theta=}, {sigma=}, {delta=}')
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')

    # plot power law with exponent -2
    x = np.array(edges[:-1])
    plt.plot(x, 10**5*x**-2, label=r'$\tau=2$ power law', linestyle='--')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    # plt.savefig(f'src/nutrient_model/cluster_sizes_{theta=}_{sigma=}.png', dpi=300)
    plt.savefig(f'src/nutrient_model/cluster_sizes_{L=}_{theta=}_{sigma=}.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()

