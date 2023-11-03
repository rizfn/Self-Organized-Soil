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
def init_lattice(L):
    """Initialize the lattice with worms randomly placed.

    Parameters
    ----------
    L : int
        Side length of the square lattice.

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
    # start with 25-25-25-25 soil and empty
    soil_lattice = np.random.choice(np.arange(0, 4), size=(L, L))
    return soil_lattice


@njit
def update_stochastic(soil_lattice, L, rho, theta, sigma, delta):
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
    site = np.random.randint(0, L), np.random.randint(0, L)

    if soil_lattice[site[0], site[1]] == 0:
        # choose a random neighbour
        nbr = neighbours(site, L)[np.random.randint(4)]
        if soil_lattice[nbr[0], nbr[1]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 2

    elif soil_lattice[site[0], site[1]] == 1:
        is_filled = False
        # choose a random neighbour
        nbr = neighbours(site, L)[np.random.randint(4)]
        if soil_lattice[nbr[0], nbr[1]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 2
                is_filled = True
        if not is_filled:
            # decay to empty with rate delta
            if np.random.rand() < delta:
                soil_lattice[site[0], site[1]] = 0

    elif soil_lattice[site[0], site[1]] == 3:
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = neighbours(site, L)[np.random.randint(4)]
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the worm
            soil_lattice[new_site[0], new_site[1]] = 3
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho:
                    soil_lattice[site[0], site[1]] = 3
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                soil_lattice[site[0], site[1]] = 1
            # check if the new site is a worm
            elif new_site_value == 3:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3


@njit
def run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
    soil_lattice = init_lattice(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic(soil_lattice, L, rho, theta, sigma, delta)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


## TODO: ADD THE WELL-MIXED MODEL
@njit
def update_stochastic_wellmixed(soil_lattice, L, r, d, s):
    return None

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
    soil_lattice = init_lattice(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic_wellmixed(soil_lattice, L, r, d, s)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data



def ode_integrate(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the single species model.

    Parameters
    ----------
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm eproduction rate.
    delta : float
        Nutrient decay rate.
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
    E : list
        List of empty fractions.
    N : list
        List of nutrient fractions.
    W : list
        List of worm fractions.
    """

    W_0 = 0.1  # initial fraction of worms
    E_0 = (1 - W_0) / 3  # initial number of empty sites
    S_0 = (1 - W_0) / 3 # initial number of soil sites
    N_0 = 1 - W_0 - E_0 - S_0  # initial number of nutrient sites

    dt = stoptime / nsteps

    S = [S_0]
    W = [W_0]
    E = [E_0]
    N = [N_0]
    T = [0]


    for i in range(nsteps):
        S.append(S[i] + dt * (sigma*S[i]*(E[i]+N[i]) - W[i]*S[i]))
        E.append(E[i] + dt * ((1-rho)*W[i]*N[i] + theta*W[i] - sigma*S[i]*E[i] + delta*N[i]))
        N.append(N[i] + dt * (W[i]*S[i] - W[i]*N[i] - sigma*S[i]*N[i] - delta*N[i]))
        W.append(W[i] + dt * (rho*W[i]*N[i] - theta*W[i]))
        T.append(T[i] + dt)
    
    return T, S, E, N, W
