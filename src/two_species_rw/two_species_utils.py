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
    #   0 = empty red
    #   1 = empty blue
    #   2 = soil
    #   3 = red bacteria
    #   4 = blue bacteria

    # start 33-33-33 emptyR-emptyB-soil
    soil_lattice = np.random.choice(np.arange(0, 3), size=(L, L))

    # place N/2 red bacteria and N/2 blue bacteria randomly
    sites = np.random.choice(L*L, size=N, replace=False)
    for site in sites[:N//2]:
        row = site // L
        col = site % L
        soil_lattice[row, col] = 3
    for site in sites[N//2:]:
        row = site // L
        col = site % L
        soil_lattice[row, col] = 4

    return soil_lattice


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

    elif soil_lattice[site[0], site[1]] == 1:
        # fill with soil-filling rate
        if np.random.rand() < s:
            soil_lattice[site[0], site[1]] = 2

    elif soil_lattice[site[0], site[1]] == 3:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = neighbours(site, L)[np.random.randint(4)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1]] = 3
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is soil
            if new_site_value == 1:
                # find neighbouring sites
                neighbours_sites = neighbours(new_site, L)
                # choose a random neighbour
                nbr = neighbours_sites[np.random.randint(4)]
                while (nbr[0], nbr[1]) == (site[0], site[1]): # todo: Optimize
                    nbr = neighbours_sites[np.random.randint(4)]
                # check if random neighbour is empty blue, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1]] == 1:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1]] = 3
            # check if the new site is a red bacteria
            elif new_site_value == 3:
                # keep both with bacteria (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3
            # check if the new site is a blue bacteria
            elif new_site_value == 4:
                # swap the positions (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 4

    elif soil_lattice[site[0], site[1]] == 4:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1]] = 1
        else:
            # move into a neighbour
            new_site = neighbours(site, L)[np.random.randint(4)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1]] = 4
            soil_lattice[site[0], site[1]] = 1
            # check if the new site is soil
            if new_site_value == 2:
                # find neighbouring sites
                neighbours_sites = neighbours(new_site, L)
                # choose a random neighbour
                nbr = neighbours_sites[np.random.randint(4)]
                while (nbr[0], nbr[1]) == (site[0], site[1]): # todo: Optimize
                    nbr = neighbours_sites[np.random.randint(4)]
                # check if random neighbour is empty red, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1]] == 0:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1]] = 4
            # check if the new site is a blue bacteria
            elif new_site_value == 4:
                # keep both with bacteria (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 4
            # check if the new site is a red bacteria
            elif new_site_value == 3:
                # swap the positions (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3


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


@njit
def update_stochastic_wellmixed(soil_lattice, L, r, d, s):
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

    elif soil_lattice[site[0], site[1]] == 1:
        # fill with soil-filling rate
        if np.random.rand() < s:
            soil_lattice[site[0], site[1]] = 2

    elif soil_lattice[site[0], site[1]] == 3:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a random site
            new_site = np.random.randint(0, L), np.random.randint(0, L)
            while (new_site == site):
                new_site = np.random.randint(0, L), np.random.randint(0, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1]] = 3
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is soil
            if new_site_value == 1:
                # choose a random reproduction site
                nbr = np.random.randint(0, L), np.random.randint(0, L)
                while (nbr == site) or (nbr == new_site): # todo: Optimize
                    nbr = np.random.randint(0, L), np.random.randint(0, L)
                # check if random repro site is empty blue, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1]] == 1:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1]] = 3
            # check if the new site is a red bacteria
            elif new_site_value == 3:
                # keep both with bacteria (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3
            # check if the new site is a blue bacteria
            elif new_site_value == 4:
                # swap the positions (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 4

    elif soil_lattice[site[0], site[1]] == 4:
        # check for death
        if np.random.rand() < d:
            soil_lattice[site[0], site[1]] = 1
        else:
            # move into a random site
            new_site = np.random.randint(0, L), np.random.randint(0, L)
            while (new_site == site):
                new_site = np.random.randint(0, L), np.random.randint(0, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the bacteria
            soil_lattice[new_site[0], new_site[1]] = 4
            soil_lattice[site[0], site[1]] = 1
            # check if the new site is soil
            if new_site_value == 2:
                # choose a random reproduction site
                nbr = np.random.randint(0, L), np.random.randint(0, L)
                while (nbr == site) or (nbr == new_site): # todo: Optimize
                    nbr = np.random.randint(0, L), np.random.randint(0, L)
                # check if random neighbour is empty red, if so, reproduce with rate r
                if soil_lattice[nbr[0], nbr[1]] == 0:
                    if np.random.rand() < r:
                        soil_lattice[nbr[0], nbr[1]] = 4
            # check if the new site is a blue bacteria
            elif new_site_value == 4:
                # keep both with bacteria (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 4
            # check if the new site is a red bacteria
            elif new_site_value == 3:
                # swap the positions (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3


@njit
def run_stochastic_wellmixed(n_steps, L, r, d, s, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the well-mixed stochastic simulation for n_steps timesteps.

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
        update_stochastic_wellmixed(soil_lattice, L, r, d, s)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


def ode_integrate(s, d, r, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the single species model.

    Parameters
    ----------
    s : float
        Soil filling rate.
    d : float
        Death rate.
    r : float
        Reproduction rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    
    Returns
    -------
    T : list
        List of times.
    S : list
        List of soil fractions.
    E_R : list
        List of red empty fractions.
    E_B : list
        List of blue empty fractions.
    R : list
        List of red bacteria fractions.
    B : list
        List of blue bacteria fractions.
    """

    B_0 = 0.05  # initial fraction of red/blue bacteria
    E_0 = (1 - 2*B_0) / 3  # initial number of empty red/blue sites
    S_0 = 1 - 2*B_0 - 2*E_0  # initial number of soil sites

    dt = stoptime / nsteps

    T = [0]
    S = [S_0]
    E_R = [E_0]
    E_B = [E_0]
    R = [B_0]
    B = [B_0]


    for i in range(nsteps):
        S.append(S[i] + dt * (s*(E_R[i]+E_B[i]) - S[i]*(R[i]+B[i])))
        E_R.append(E_R[i] + dt * (R[i]*(S[i]+E_B[i]) + d*R[i] - s*E_R[i] - r*B[i]*S[i]*E_R[i] - B[i]*E_R[i]))
        E_B.append(E_B[i] + dt * (B[i]*(S[i]+E_R[i]) + d*B[i] - s*E_B[i] - r*R[i]*S[i]*E_B[i] - R[i]*E_B[i]))
        R.append(R[i] + dt * (r*R[i]*S[i]*E_B[i] - d*R[i]))
        B.append(B[i] + dt * (r*B[i]*S[i]*E_R[i] - d*B[i]))
        T.append(T[i] + dt)
    
    return T, S, E_R, E_B, R, B

