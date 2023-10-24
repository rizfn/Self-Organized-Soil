import numpy as np
from tqdm import tqdm
import pandas as pd
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
def init_lattice_3D(L, N):
    """Initialize the lattice with N bacteria randomly placed on the lattice.

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
    #   0 = empty red
    #   1 = empty blue
    #   2 = soil
    #   3 = red worm
    #   4 = blue worm
    
    # start with 33-33-33 soil, emptyR, emptyB
    soil_lattice = np.random.choice(np.arange(0, 3), size=(L, L, L))
    # choose random sites to place N bacteria
    sites = np.random.randint(0, L, size=(N, 3))
    for site in sites[:N//2]:
        soil_lattice[site[0], site[1], site[2]] = 3
    for site in sites[N//2:]:
        soil_lattice[site[0], site[1], site[2]] = 4

    return soil_lattice



@njit
def update_stochastic_3D(soil_lattice, L, r, d, s):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the cubic lattice.
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

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, L)

    if (soil_lattice[site[0], site[1], site[2]] == 0) or (soil_lattice[site[0], site[1], site[2]] == 1):
        nbr = neighbours_3D(site, L)[np.random.randint(6)]
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:
            # fill with soil-filling rate
            if np.random.rand() < s:
                soil_lattice[site[0], site[1], site[2]] = 2


    elif soil_lattice[site[0], site[1], site[2]] == 3:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = neighbours_3D(site, L)[np.random.randint(6)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 3
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is soil
            if new_site_value == 2:
                # find neighbouring sites
                neighbours_sites = neighbours_3D(new_site, L)
                # choose a random neighbour
                nbr = neighbours_sites[np.random.randint(6)]
                while (nbr[0], nbr[1], nbr[2]) == (site[0], site[1], site[2]): # todo: Optimize
                    nbr = neighbours_sites[np.random.randint(6)]
                # check if random neighbour is empty blue, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1], nbr[2]] == 1:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1], nbr[2]] = 3
            # check if the new site is a bacteria
            elif (new_site_value == 3) or (new_site_value == 4):
                # swap the bacteria positions (undo the vacant space in the original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value

    elif soil_lattice[site[0], site[1], site[2]] == 4:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1], site[2]] = 1
        else:
            # move into a neighbour
            new_site = neighbours_3D(site, L)[np.random.randint(6)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 4
            soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is soil
            if new_site_value == 2:
                # find neighbouring sites
                neighbours_sites = neighbours_3D(new_site, L)
                # choose a random neighbour
                nbr = neighbours_sites[np.random.randint(6)]
                while (nbr[0], nbr[1], nbr[2]) == (site[0], site[1], site[2]): # todo: Optimize
                    nbr = neighbours_sites[np.random.randint(6)]
                # check if random neighbour is empty red, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1], nbr[2]] == 0:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1], nbr[2]] = 4
            # check if the new site is a bacteria
            elif (new_site_value == 3) or (new_site_value == 4):
                # swap the bacteria positions (undo the vacant space in the original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value


@njit
def run_stochastic_3D(n_steps, L, r, d, s, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

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
    N = int(L**3 / 10)  # initial number of bacteria
    soil_lattice = init_lattice_3D(L, N)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic_3D(soil_lattice, L, r, d, s)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


def run_raster_stochastic_3D(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the simulation for n_steps timesteps.
    
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
    """
    grid = np.meshgrid(d_list, s_list)
    ds_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_lattice_list = []
    for i in tqdm(range(len(ds_pairs))):  # todo: parallelize
        d, s = ds_pairs[i]
        soil_lattice_data = run_stochastic_3D(n_steps, L, r, d, s, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 10_000_000  # number of bacteria moves (more for stochastic)
    L = 20  # side length of the cubic lattice
    r = 1  # reproduction rate
    # d = np.linspace(0, 0.3, 20)  # death rate
    # s = np.linspace(0, 0.3, 20)  # soil filling rate
    d = np.linspace(0, 0.3, 20)  # death rate (same as mean field)
    s = np.linspace(0, 0.8, 20)  # soil filling rate (same as mean field)

    soil_lattice_data = run_raster_stochastic_3D(n_steps, L, r, d, s, np.geomspace(10000, n_steps, int(np.log10(n_steps/10000))+1, dtype=np.int32))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/single_species/3D_stochastic_dynamics_{r=}.json", orient="records")
    


if __name__ == "__main__":
    main()
