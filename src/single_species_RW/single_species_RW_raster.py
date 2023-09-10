import numpy as np
from numba import njit
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

    # fill every empty lattice site with soil
    empty_sites = np.argwhere(soil_lattice == 0)
    for empty_site in empty_sites:
        if np.random.rand() < s:
            soil_lattice[empty_site[0], empty_site[1]] = 1
    
    # find bacteria sites
    bacteria_sites = np.argwhere(soil_lattice == 2)

    # check if there are no bacteria left
    if len(bacteria_sites) == 0:
        return

    # choose a random bacteria site
    site = bacteria_sites[np.random.randint(len(bacteria_sites))]
    # move to a random neighbour
    new_site = neighbours(site, L)[np.random.randint(4)]
    # check the value of the new site
    new_site_value = soil_lattice[new_site[0], new_site[1]]
    # move the bacteria
    soil_lattice[new_site[0], new_site[1]] = 2
    soil_lattice[site[0], site[1]] = 0

    # check if the new site was empty
    if new_site_value == 0:
        # check if the bacteria dies
        if np.random.rand() < d:
            soil_lattice[new_site[0], new_site[1]] = 0

    # check if the new site is soil
    elif new_site_value == 1:
        # find neighbouring sites
        neighbours_sites = neighbours(new_site, L)
        # filter for empty sites
        empty_sites = [[nbr[0], nbr[1]] if soil_lattice[nbr[0], nbr[1]] == 0 else None for nbr in neighbours_sites]
        # for each empty site, check if a new bacteria is born
        for empty_site in empty_sites:
            if empty_site is not None:
                if np.random.rand() < r:
                    soil_lattice[empty_site[0], empty_site[1]] = 2

    # # check if the new site is a bacteria
    # elif new_site_value == 2:
    #     # keep both with bacteria
    #     soil_lattice[new_site[0], new_site[1]] = 2
    #     soil_lattice[site[0], site[1]] = 2


    # NOTE: when the new site is a bacteria, nothing happens
    #       in effect, the bacteria overwrite each other, like they're "eating" each other
    

def main():

    # initialize the parameters
    n_steps = 100_000  # number of bacteria moves
    L = 10  # side length of the square lattice
    N = 10  # initial number of bacteria
    r = 0.5  # reproduction rate
    d = np.linspace(0, 1, 100)  # death rate
    s = np.linspace(0, 0.5, 100)  # soil filling rate
    soil_lattice_data = []

    for d_i in tqdm(d):
        for s_i in s:
            soil_lattice = init_lattice(L, N)
            for step in range(1, n_steps+1):
                update(soil_lattice, L, r, d_i, s_i)
                if step in [100, 1000, 10000, 100000]:
                    soil_lattice_data.append({"d": d_i, "s": s_i, "step": step, "soil_lattice": soil_lattice.copy()})

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/soil_lattice_data_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
