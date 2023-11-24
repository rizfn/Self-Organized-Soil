import numpy as np
from tqdm import tqdm
import pandas as pd
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
    L = 500  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**2  # number of bacteria moves
    rho = 1  # reproduction rate
    theta = 0.06  # death rate
    sigma = 0.68  # soil filling rate
    delta = 0 # nutrient decay rate

    steps_to_record = np.arange(n_steps//2, n_steps, 20 * L**2, dtype=np.int32)

    # run the simulation
    soil_lattice_data = run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record)

    # calculate the cluster sizes
    soil_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 2))

     # Calculate the lengths and areas of the clusters
    lengths = np.sqrt(soil_cluster_sizes)
    areas = soil_cluster_sizes

    # Calculate the cumulative areas for each length
    sorted_indices = np.argsort(lengths)
    sorted_lengths = lengths[sorted_indices]
    sorted_areas = areas[sorted_indices]
    cumulative_areas = np.cumsum(sorted_areas)

    # Normalize the lengths and cumulative areas
    normalized_lengths = sorted_lengths / L
    normalized_cumulative_areas = cumulative_areas / np.max(cumulative_areas)

    # Convert the normalized lengths and cumulative areas to a DataFrame
    df = pd.DataFrame({
        'Normalized Lengths': normalized_lengths,
        'Normalized Cumulative Areas': normalized_cumulative_areas
    })

    # Group by normalized lengths and take the maximum value in each group
    grouped = df.groupby('Normalized Lengths').max()

    plt.title(f'{L=}, {rho=}, {theta=}, {sigma=}, {delta=}')

    # Plot the log of the maximum normalized cumulative areas against the log of the normalized lengths
    plt.plot(np.log10(grouped.index), np.log10(grouped['Normalized Cumulative Areas']), marker='x', linestyle='none', label='Simulation')
    plt.xlabel('Log of Normalized Length')
    plt.ylabel('Log of Normalized Cumulative Area')

    # plot power-law with exponent tau
    tau = 1.1
    plt.plot(np.log10(normalized_lengths), tau * np.log10(normalized_lengths) + 1.6, label=f'Slope = {tau}', linestyle='--')

    plt.legend()
    plt.savefig('src/nutrient_model/fractal_dimension.png', dpi=300)
    plt.show()





if __name__ == "__main__":
    main()

