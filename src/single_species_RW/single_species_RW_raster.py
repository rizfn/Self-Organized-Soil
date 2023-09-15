import numpy as np
from numba import njit, jit, prange
from tqdm import tqdm
import pandas as pd

@njit
def neighbours(c, L):
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

    return np.array([[(c[0]-1)%L, c[1]], [(c[0]+1)%L, c[1]], [c[0], (c[1]-1)%L], [c[0], (c[1]+1)%L]])


@njit
def init_lattice(L, N):
    """Initialize the lattice with N bacteria randomly placed on the lattice.

    Parameters
    ----------
    L : int
        Side length of the square lattice.
    N : int
        Number of bacteria to place on the lattice.

    Returns
    -------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    """

    soil_lattice = np.ones((L, L), dtype=np.int8)
    # note about lattice:
    #   0 = empty
    #   1 = soil
    #   2 = bacteria
    # set half the sites to 0
    empty_sites = np.random.choice(L*L, size=L*L//2, replace=False)
    for site in empty_sites:
        row = site // L
        col = site % L
        soil_lattice[row, col] = 0
    # choose random sites to place N bacteria
    sites = np.random.choice(L*L, size=N, replace=False)
    # place bacteria on the lattice
    for site in sites:
        row = site // L
        col = site % L
        soil_lattice[row, col] = 2
    return soil_lattice


@njit
def update(soil_lattice, L, r, d, s):
    """Update the lattice. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the square lattice.
    r : float
        Reproduction rate.
    d : float
        Death rate.
    s : float
        Soil filling rate.
    
    Returns:
    --------
    None
    """

    # NEW SOIL FILLING MECHANICS
    empty_sites = np.argwhere(soil_lattice == 0)
    should_be_filled = np.random.rand(len(empty_sites)) < s
    for i, site in enumerate(empty_sites):
        if should_be_filled[i]:
            soil_lattice[site[0], site[1]] = 1

    # NEW DEATH MECHANICS
    # find all sites which have bacteria
    bacteria_sites = np.argwhere(soil_lattice == 2)
    should_be_killed = np.random.rand(len(bacteria_sites)) < d
    for i, site in enumerate(bacteria_sites):
        if should_be_killed[i]:
            soil_lattice[site[0], site[1]] = 0
    

    # find bacteria sites
    bacteria_sites = np.argwhere(soil_lattice == 2)

    for site in bacteria_sites:
        # select a random neighbour
        new_site = neighbours(site, L)[np.random.randint(4)]
        # check the value of the new site
        new_site_value = soil_lattice[new_site[0], new_site[1]]
        # move the bacteria
        soil_lattice[new_site[0], new_site[1]] = 2
        soil_lattice[site[0], site[1]] = 0

        # check if the new site is soil
        if new_site_value == 1:
            # find neighbouring sites
            neighbours_sites = neighbours(new_site, L)
            for nbr in neighbours_sites:  # todo: Optimize
                if (nbr[0], nbr[1]) != (site[0], site[1]):
                    if soil_lattice[nbr[0], nbr[1]] == 0:
                        if np.random.rand() < r:
                            soil_lattice[nbr[0], nbr[1]] = 2
                            break

        # check if the new site is a bacteria
        elif new_site_value == 2:
            # keep both with bacteria (undo the vacant space in original site)
            soil_lattice[site[0], site[1]] = 2


@njit
def run(n_steps, L, r, d, s, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    r : float
        Reproduction rate.
    d : float
        Death rate.
    s : float
        Soil filling rate.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soil_lattice_data : ndarray
        List of soil_lattice data for specific timesteps.
    """
    N = int(L**2 / 10)  # initial number of bacteria
    soil_lattice = init_lattice(L, N)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update(soil_lattice, L, r, d, s)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


def run_raster(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    '''Run the simulation for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    r : float
        Reproduction rate.
    d_list : ndarray
        List of death rates.
    s_list : ndarray
        List of soil filling rates.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].
    
    Returns
    -------
    soil_lattice_list : list
        List of soil_lattice data for specific timesteps and parameters.
    '''
    grid = np.meshgrid(d_list, s_list)
    ds_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_lattice_list = []
    for i in tqdm(range(len(ds_pairs))):  # todo: parallelize
        d, s = ds_pairs[i]
        soil_lattice_data = run(n_steps, L, r, d, s, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of bacteria moves
    L = 20  # side length of the square lattice
    r = 1  # reproduction rate
    d = np.linspace(0, 0.3, 20)  # death rate
    s = np.linspace(0, 0.3, 20)  # soil filling rate

    soil_lattice_data = run_raster(n_steps, L, r, d, s)

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/single_species/soil_lattice_data_{r=}.json", orient="records")



# def main():

#     # initialize the parameters
#     n_steps = 100_000  # number of bacteria moves
#     L = 20  # side length of the square lattice
#     N = int(L**2 / 10)  # initial number of bacteria
#     r = 1  # reproduction rate
#     d = np.linspace(0, 0.3, 20)  # death rate
#     s = np.linspace(0, 0.3, 20)  # soil filling rate
#     soil_lattice_data = []

#     for d_i in tqdm(d):
#         for s_i in s:
#             soil_lattice = init_lattice(L, N)
#             for step in range(1, n_steps+1):
#                 update(soil_lattice, L, r, d_i, s_i)
#                 if step in [100, 1000, 10000, 100000]:
#                     soil_lattice_data.append({"d": d_i, "s": s_i, "step": step, "soil_lattice": soil_lattice.copy()})

#     soil_lattice_data = pd.DataFrame(soil_lattice_data)

#     # save the data
#     soil_lattice_data.to_json(f"docs/data/single_species/soil_lattice_data_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
