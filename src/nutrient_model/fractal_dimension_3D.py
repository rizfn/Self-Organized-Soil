import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
from numba import njit


@njit
def neighbours_3D(c, L):
    """Find the neighbouring sites of a site on a square lattice.
    
    Parameters
    ----------
    c : numpy.ndarray
        Coordinates of the site.
    L : int
    Side length of the square lattice.
    
    Returns
    -------
    numpy.ndarray
    Coordinates of the neighbouring sites.
    """
    return np.array([[(c[0]-1)%L, c[1], c[2]], [(c[0]+1)%L, c[1], c[2]], [c[0], (c[1]-1)%L, c[2]], [c[0], (c[1]+1)%L, c[2]], [c[0], c[1], (c[2]-1)%L], [c[0], c[1], (c[2]+1)%L]])

@njit
def init_lattice_3D(L):
    """Initialize the 3D lattice.

    Parameters
    ----------
    L : int
        Side length of the cubic lattice.
    N : int
        Number of bacteria to place on the lattice.

    Returns
    -------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    """

    # note about lattice:
    #   0 = empty
    #   1 = nutrient
    #   2 = soil
    #   3 = worm
    # start with 25-25-25-25
    soil_lattice = np.random.choice(np.arange(0, 4), size=(L, L, L))

    return soil_lattice

@njit
def update_stochastic_3D(soil_lattice, L, rho, theta, sigma, delta):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the square lattice.
    rho : float
        Reproduction rate.
    theta : float
        Death rate.
    sigma : float
        Soil filling rate.
    delta : float
        Nutrient decay rate.
    
    Returns:
    --------
    None
    """

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, L)

    if soil_lattice[site[0], site[1], site[2]] == 0:
        # choose a random neighbour
        nbr = neighbours_3D(site, L)[np.random.randint(6)]
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 2

    elif soil_lattice[site[0], site[1], site[2]] == 1:
        is_filled = False
        # choose a random neighbour
        nbr = neighbours_3D(site, L)[np.random.randint(6)]
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 2
                is_filled = True
        if not is_filled:
            # decay to empty with rate delta
            if np.random.rand() < delta:
                soil_lattice[site[0], site[1], site[2]] = 0

    elif soil_lattice[site[0], site[1], site[2]] == 3:
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = neighbours_3D(site, L)[np.random.randint(6)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 3
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho:
                    soil_lattice[site[0], site[1], site[2]] = 3
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is a worm
            elif new_site_value == 3:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = 3


@njit
def run_stochastic_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho : float
        Reproduction rate.
    theta : float
        Death rate.
    sigma : float
        Soil filling rate.
    delta : float
        Nutrient decay rate.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soil_lattice_data : ndarray
        List of soil_lattice data for specific timesteps.
    """
    N = int(L**2 / 10)  # initial number of bacteria
    soil_lattice = init_lattice_3D(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic_3D(soil_lattice, L, rho, theta, sigma, delta)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data



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
    L = 50  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**3  # number of bacteria moves
    rho = 1  # reproduction rate
    theta = 0.04  # death rate
    sigma = 0.68  # soil filling rate
    delta = 0 # nutrient decay rate

    steps_to_record = np.arange(n_steps//2, n_steps, 20 * L**2, dtype=np.int32)

    # run the simulation
    soil_lattice_data = run_stochastic_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record)

    # calculate the cluster sizes
    soil_cluster_sizes = np.concatenate(calculate_cluster_sizes(soil_lattice_data, 2))

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
    plt.plot(np.log10(grouped.index), np.log10(grouped['Normalized Cumulative Volumes']), marker='x', linestyle='none', label='Simulation')
    plt.xlabel('Log of Normalized Length')
    plt.ylabel('Log of Normalized Cumulative Volumes')

    # plot power-law with exponent tau
    tau = 0.5
    plt.plot(np.log10(normalized_lengths), tau * np.log10(normalized_lengths) , label=f'Slope = {tau}', linestyle='--')

    # plot power-law with exponent tau
    tau = 0.4
    plt.plot(np.log10(normalized_lengths), tau * np.log10(normalized_lengths) - 0.1 , label=f'Slope = {tau}', linestyle='--')

    plt.legend()
    plt.savefig(f'src/nutrient_model/fractal_dimension_3D_{theta=}_{sigma=}.png', dpi=300)
    plt.show()





if __name__ == "__main__":
    main()

