import numpy as np
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt

def calculate_cluster_sizes(soil_lattice, target_site=1):
    """Calculate the cluster sizes for a single timestep.

    Parameters
    ----------
    soil_lattice : ndarray
        2D array of soil_lattice data for a single timestep.
    target_site : int
        The value of the site that is to be considered part of the cluster.
    
    Returns
    -------
    cluster_sizes : ndarray
        Array of cluster sizes for the timestep.
    """
    m = soil_lattice == target_site
    lw, num = ndimage.label(m)
    cluster_sizes = ndimage.sum(m, lw, index=np.arange(num + 1))
    return cluster_sizes


def main():
    L = 1024
    sigma = 1
    theta = 0.042
    rhofactor = 4
    n_steps = 10000
    portion_size = L * L
    record_every = 50  # Record cluster sizes every 50 steps

    filename = f'docs/data/twospec_samenutrient/lattice_anim_L_{L}_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.bin'

    all_cluster_sizes = []

    with open(filename, 'rb') as f:
        for current_chunk in tqdm(range(n_steps)):
            if current_chunk % record_every == 0:  # Only calculate cluster sizes every record_every steps
                f.seek(current_chunk * portion_size)
                portion = np.fromfile(f, dtype=np.uint8, count=portion_size)
                portion = portion.reshape((L, L))  # Reshape the portion to a 2D array

                # Calculate the cluster sizes for the 2 state
                cluster_sizes = calculate_cluster_sizes(portion, target_site=2)
                all_cluster_sizes.extend(cluster_sizes)

    # Create a histogram of the cluster sizes
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(all_cluster_sizes)
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

    hist, edges = np.histogram(all_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    plt.plot(edges[:-1], hist, 'x', label='soil sites')
    plt.title(f'Histogram of cluster sizes\n{L=}, $\\sigma$={sigma}, $\\theta$={theta}, $\\rho_2/\\rho_1$={rhofactor}')
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.xscale('log')
    plt.yscale('log')
    ylim = plt.ylim()
    # plot power law with exponent -3, -2
    x = np.array(edges[:-1])
    plt.plot(x, hist[0]*x**-2, label=r'$\tau=2$ power law', linestyle='--', alpha=0.4)
    plt.plot(x, hist[0]*x**-2.5, label=r'$\tau=2.5$ power law', linestyle='--', alpha=0.4)
    plt.plot(x, hist[0]*x**-3, label=r'$\tau=3$ power law', linestyle='--', alpha=0.4)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f'src/two_species_same_nutrient/plots/cluster_sizes/L_{L}_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.png', dpi=300)
    plt.show()


def viz_intervals():
    L = 1024
    sigma = 1
    theta = 0.042
    rhofactor = 4
    n_steps = 10000
    portion_size = L * L
    record_every = 50  # Record cluster sizes every 50 steps

    filename = f'docs/data/twospec_samenutrient/lattice_anim_L_{L}_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.bin'

    all_cluster_sizes = []

    with open(filename, 'rb') as f:
        for current_chunk in tqdm(range(n_steps)):
            if current_chunk % record_every == 0:  # Only calculate cluster sizes every record_every steps
                f.seek(current_chunk * portion_size)
                portion = np.fromfile(f, dtype=np.uint8, count=portion_size)
                portion = portion.reshape((L, L))  # Reshape the portion to a 2D array

                # Calculate the cluster sizes for the 2 state
                cluster_sizes = calculate_cluster_sizes(portion, target_site=2)
                all_cluster_sizes.extend(cluster_sizes)

    # Create a histogram of the cluster sizes
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(all_cluster_sizes)
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

    hist, edges = np.histogram(all_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    plt.plot(edges[:-1], hist, 'x', label='soil sites')
    plt.title(f'Histogram of cluster sizes\n{L=}, $\\sigma$={sigma}, $\\theta$={theta}, $\\rho_2/\\rho_1$={rhofactor}')
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.xscale('log')
    plt.yscale('log')
    ylim = plt.ylim()
    # plot power law with exponent -3, -2
    x = np.array(edges[:-1])
    plt.plot(x, hist[0]*x**-2, label=r'$\tau=2$ power law', linestyle='--', alpha=0.4)
    plt.plot(x, hist[0]*x**-2.5, label=r'$\tau=2.5$ power law', linestyle='--', alpha=0.4)
    plt.plot(x, hist[0]*x**-3, label=r'$\tau=3$ power law', linestyle='--', alpha=0.4)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f'src/two_species_same_nutrient/plots/cluster_sizes/L_{L}_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()