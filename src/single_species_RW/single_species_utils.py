import numpy as np
from numba import njit


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

    # note about lattice:
    #   0 = empty
    #   1 = soil
    #   2 = bacteria
    # start with 50-50 soil and empty
    soil_lattice = np.random.choice(np.arange(0, 2), size=(L, L))
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
    

    # find bacteria sites  # todo: choose random bacteria
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


@njit
def update_wellmixed(soil_lattice, L, r, d, s):
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

    empty_sites = np.argwhere(soil_lattice == 0)
    should_be_filled = np.random.rand(len(empty_sites)) < s
    for i, site in enumerate(empty_sites):
        if should_be_filled[i]:
            soil_lattice[site[0], site[1]] = 1

    # find all sites which have bacteria
    bacteria_sites = np.argwhere(soil_lattice == 2)
    should_be_killed = np.random.rand(len(bacteria_sites)) < d
    for i, site in enumerate(bacteria_sites):
        if should_be_killed[i]:
            soil_lattice[site[0], site[1]] = 0
    

    for site in bacteria_sites:
        # select a random target site
        new_site = np.random.randint(0, L), np.random.randint(0, L)
        # check the value of the new site
        new_site_value = soil_lattice[new_site[0], new_site[1]]
        # move the bacteria
        soil_lattice[new_site[0], new_site[1]] = 2
        soil_lattice[site[0], site[1]] = 0

        # check if the new site is soil
        if new_site_value == 1:
            # choose a third random site
            random_spawnpoint = np.random.randint(0, L), np.random.randint(0, L)
            if soil_lattice[random_spawnpoint[0], random_spawnpoint[1]] == 0:
                if np.random.rand() < r:
                    soil_lattice[random_spawnpoint[0], random_spawnpoint[1]] = 2
                    break

        # check if the new site is a bacteria
        elif new_site_value == 2:
            # keep both with bacteria (undo the vacant space in original site)
            soil_lattice[site[0], site[1]] = 2


@njit
def update_stochastic(soil_lattice, L, r, d, s):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
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

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L)

    if soil_lattice[site[0], site[1]] == 0:
        # fill with soil-filling rate
        if np.random.rand() < s:
            soil_lattice[site[0], site[1]] = 1

    elif soil_lattice[site[0], site[1]] == 2:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
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
def run_stochastic(n_steps, L, r, d, s, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
    N = int(L**2 / 10)  # initial number of bacteria
    soil_lattice = init_lattice(L, N)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic(soil_lattice, L, r, d, s)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


