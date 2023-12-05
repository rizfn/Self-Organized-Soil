import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from numba import njit
from nutrient_utils import run_stochastic_3D


def calculate_cluster_sizes(soil_lattice_data, target_site=1):
    """Calculate the cluster sizes for each timestep.

    Parameters
    ----------
    soil_lattice_data : ndarray
        3D array of soil_lattice data for each timestep.
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
    L = 100  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**3  # number of bacteria moves
    rho = 1  # reproduction rate
    theta = 0.04  # death rate
    sigma = 1  # soil filling rate
    delta = 0 # nutrient decay rate

    steps_to_record = np.arange(n_steps//2, n_steps, 10 * L**3, dtype=np.int32)

    # run the simulation
    soil_lattice_data = run_stochastic_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record)

    # calculate the cluster sizes
    soil_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 2))
    soil_cluster_sizes = soil_cluster_sizes[soil_cluster_sizes > 0]

    # plot cluster size distribution with power law of slope tau
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(soil_cluster_sizes)
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)
    # Calculate histograms and plot
    hist, edges = np.histogram(soil_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    plt.plot(edges[:-1], hist, 'x', label='Soil Sites')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Cluster Size')
    plt.ylabel('Probability Density')

    # Convert to numpy arrays
    x = np.array(edges[:-1])
    y = np.array(hist)
    # slice to only fit to the middle, and remove zeros to avoid log(0)
    x, y = x[num_bins//4:3*num_bins//4], y[num_bins//4:3*num_bins//4]
    # x, y = x[:num_bins//2], y[:num_bins//2]
    mask = (y > 0) & (x > 0)
    x = x[mask]
    y = y[mask]

    # Calculate the logarithm of x and y
    log_x = np.log10(x)
    log_y = np.log10(y)

    # Use polyfit to fit a 1-degree polynomial to the log-log data
    coeffs = np.polyfit(log_x, log_y, 1)
    # The slope of the line is -gamma
    gamma = -coeffs[0]

    # Plot the power law with the calculated exponent gamma
    plt.plot(x, 10**coeffs[1] * x**-gamma, label=r'$\gamma$ = ' + f'{gamma:.2f} power law', linestyle='--')
    plt.title(f'{L=}, {rho=}, {theta=}, {sigma=}, {delta=}')
    plt.legend()
    plt.savefig(f'src/nutrient_model/plots/cluster_sizes_3D_{theta=}_{sigma=}.png', dpi=300)
    plt.show()


    # Calculate the lengths and volumes of the clusters
    lengths = (soil_cluster_sizes)**(1/3)
    volumes = soil_cluster_sizes

    # Calculate the cumulative volumes for each length
    sorted_indices = np.argsort(lengths)
    sorted_lengths = lengths[sorted_indices]
    sorted_volumes = volumes[sorted_indices]
    cumulative_volumes = np.cumsum(sorted_volumes)

    # Normalize the lengths and cumulative volumes
    normalized_lengths = sorted_lengths / L
    normalized_cumulative_volumes = cumulative_volumes / np.max(cumulative_volumes)

    # Convert the normalized lengths and cumulative volumes to a DataFrame
    df = pd.DataFrame({
        'Normalized Lengths': normalized_lengths,
        'Normalized Cumulative Volumes': normalized_cumulative_volumes
    })

    # Group by normalized lengths and take the maximum value in each group
    grouped = df.groupby('Normalized Lengths').max()

    plt.title(f'{L=}, {rho=}, {theta=}, {sigma=}, {delta=}')

    # Plot the log of the maximum normalized cumulative volumes against the log of the normalized lengths
    plt.plot(np.log10(grouped.index), np.log10(grouped['Normalized Cumulative Volumes']), marker='.', linestyle='none', label='Simulation')
    plt.xlabel('Log of Normalized Length')
    plt.ylabel('Log of Normalized Cumulative Volumes')

    # plot power-law with exponent tau
    tau = (4 - gamma) / 3    
    x_match = 1  # the x-value where the power law and the line intersect
    idx = np.abs(np.log10(normalized_lengths) + x_match).argmin()
    y_line = tau * -x_match
    y_shift = np.log10(normalized_cumulative_volumes)[idx] - y_line

    # Plot the line with the calculated y-shift
    plt.plot(np.log10(normalized_lengths), tau * np.log10(normalized_lengths) + y_shift, label=f'Slope = {tau:.2f}', linestyle='--')
    plt.legend()
    plt.savefig(f'src/nutrient_model/plots/fractal_dimension_3D_{theta=}_{sigma=}.png', dpi=300)
    plt.show()





if __name__ == "__main__":
    main()

