import numpy as np
from numba import njit


@njit
def get_random_neighbour(c, L):
    """Get a random neighbour of a site on a square lattice.
    
    Parameters
    ----------
    c : numpy.ndarray
        Coordinates of the site.
    L : int
    Side length of the square lattice.
    
    Returns
    -------
    numpy.ndarray
    Coordinates of the random neighbouring site.
    """
    c = np.array(c)
    # choose a random coordinate to change
    coord_changing = np.random.randint(0, 2)
    # choose a random direction to change the coordinate
    change = 2 * np.random.randint(0, 2) - 1
    # change the coordinate
    c[coord_changing] = (c[coord_changing] + change) % L
    return c


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
        Lattice with worms randomly placed on it.
    """

    # note about lattice:
    #   0 = empty
    #   1 = nutrient
    #   2 = soil
    #   3 = worm green
    #   4 = worm blue

    # start with equal number of everything
    soil_lattice = np.random.choice(np.arange(0, 5), size=(L, L)).astype(np.int8)
    return soil_lattice


@njit
def update(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with worms randomly placed on it.
    L : int
        Side length of the square lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    mu1 : float
        Nutrient creation rate of green worms.
    mu2 : float
        Nutrient creation rate of blue worms.

    Returns:
    --------
    None
    """

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L)

    if (soil_lattice[site[0], site[1]] == 0) or (soil_lattice[site[0], site[1]] == 1):  # empty or nutrient
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 2

    elif soil_lattice[site[0], site[1]] == 3:  # green worm
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the worm
            soil_lattice[new_site[0], new_site[1]] = 3
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho1:
                    soil_lattice[site[0], site[1]] = 3
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                if np.random.rand() < mu1:
                    soil_lattice[site[0], site[1]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = new_site_value

    elif soil_lattice[site[0], site[1]] == 4:  # blue worm
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the worm
            soil_lattice[new_site[0], new_site[1]] = 4
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1]] = 4
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                if np.random.rand() < mu2:
                    soil_lattice[site[0], site[1]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = new_site_value


@njit
def run(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Death rate.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    mu1 : float 
        Nutrient creation rate of green worms.
    mu2 : float
        Nutrient creation rate of blue worms.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soil_lattice_data : ndarray
        List of soil_lattice data for specific timesteps.
    """
    soil_lattice = init_lattice(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2)
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



@njit
def get_random_neighbour_3D(c, L):
    """Get a random neighbour of a site on a cubic lattice.
    
    Parameters
    ----------
    c : numpy.ndarray
        Coordinates of the site.
    L : int
    Side length of the square lattice.
    
    Returns
    -------
    numpy.ndarray
    Coordinates of the random neighbouring site.
    """
    c = np.array(c)
    # choose a random coordinate to change
    coord_changing = np.random.randint(3)
    # choose a random direction to change the coordinate
    change = 2 * np.random.randint(2) - 1
    # change the coordinate
    c[coord_changing] = (c[coord_changing] + change) % L
    return c

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
    #   3 = green worm
    #   4 = blue worm
    # start with equal number of everything
    soil_lattice = np.random.choice(np.arange(0, 5), size=(L, L, L))

    return soil_lattice

@njit
def update_3D(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the square lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho1 : float
        Green worm reproduction rate.
    rho2 : float
        Blue worm reproduction rate.
    mu1 : float
        Green worm nutrient-creating rate.
    mu2 : float
        Blue worm nutrient-creating rate.
            
    Returns:
    --------
    None
    """

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, L)

    if (soil_lattice[site[0], site[1], site[2]] == 0) or (soil_lattice[site[0], site[1], site[2]] == 1):  # if empty or nutrient
        # choose a random neighbour
        nbr = get_random_neighbour_3D(site, L)
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 2


    elif soil_lattice[site[0], site[1], site[2]] == 3:  # if green worm
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour_3D(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 3
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho1:
                    soil_lattice[site[0], site[1], site[2]] = 3
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                if np.random.rand() < mu1:
                    soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value

    elif soil_lattice[site[0], site[1], site[2]] == 4:  # if blue worm
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour_3D(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 4
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1], site[2]] = 4
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                if np.random.rand() < mu2:
                    soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value


@njit
def run_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Death rate of worms.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    mu1 : float
        Nutrient-creating rate of green worms.
    mu2 : float
        Nutrient-creating rate of blue worms.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soil_lattice_data : ndarray
        List of soil_lattice data for specific timesteps.
    """
    soil_lattice = init_lattice_3D(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_3D(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data




@njit
def ode_derivatives(S, E, N, W_G, W_B, sigma, theta, rho1, rho2, mu1, mu2):
    """Calculate the derivatives of S, E, N, W.

    This function is not called directly, but rather through `ode_integrate_rk4`
    
    Parameters
    ----------
    S : float
        Soil fraction.
    E : float
        Empty fraction.
    N : float
        Nutrient fraction.
    W_G : float
        Green worm fraction.
    W_B : float
        Blue worm fraction.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho1 : float
        Green worm reproduction rate.
    rho2 : float
        Blue worm reproduction rate.
    mu1 : float
        Green worm nutrient-creating rate.
    mu2 : float
        Blue worm nutrient-creating rate.

    Returns
    -------
    dS : float
        Derivative of soil fraction.
    dE : float
        Derivative of empty fraction.
    dN : float
        Derivative of nutrient fraction.
    dW_G : float
        Derivative of green worm fraction.
    dW_B : float
        Derivative of blue worm fraction.
    """

    dS = sigma*S*(E+N) - S*(W_G+W_B)
    dE = S*((1-mu1)*W_G + (1-mu2)*W_B) + N*((1-rho1)*W_G + (1-rho2)*W_B) + theta*(W_G+W_B) - sigma*S*E
    dN = S*(mu1*W_G + mu2*W_B) - N*(W_G+W_B) - sigma*S*N
    dW_G = rho1*W_G*N - theta*W_G
    dW_B = rho2*W_B*N - theta*W_B

    return dS, dE, dN, dW_G, dW_B


@njit
def ode_integrate_rk4(sigma, theta, rho1, rho2, mu1, mu2, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method.
    
    Parameters
    ----------
    
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho1 : float
        Green worm reproduction rate.
    rho2 : float
        Blue worm reproduction rate.
    mu1 : float
        Green worm nutrient-creating rate.
    mu2 : float
        Blue worm nutrient-creating rate.
    stoptime : int, optional
        Time to stop the integration at, by default 100_000.
    nsteps : int, optional
        Number of steps to take, by default 100_000.

    Returns
    -------
    T : ndarray
        List of times.
    S : ndarray
        List of soil fractions.
    E : ndarray
        List of empty fractions.
    N : ndarray
        List of nutrient fractions.
    WG : ndarray
        List of green worm fractions.
    WB : ndarray
        List of blue worm fractions.
    """

    # initial condition with equal fractions: todo: Play around to see absorbing boundaries change in param space
    E_0, N_0, S_0, WG_0, WB_0 = 1/5, 1/5, 1/5, 1/5, 1/5

    dt = stoptime / nsteps

    S = np.zeros(nsteps+1)
    E = np.zeros(nsteps+1)
    N = np.zeros(nsteps+1)
    WG = np.zeros(nsteps+1)
    WB = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    S[0] = S_0
    E[0] = E_0
    N[0] = N_0
    WG[0] = WG_0
    WB[0] = WB_0
    T[0] = 0

    for i in range(nsteps):
        k1_S, k1_E, k1_N, k1_WG, k1_WB = ode_derivatives(S[i], E[i], N[i], WG[i], WB[i], sigma, theta, rho1, rho2, mu1, mu2)

        S_temp = S[i] + 0.5 * dt * k1_S
        E_temp = E[i] + 0.5 * dt * k1_E
        N_temp = N[i] + 0.5 * dt * k1_N
        WG_temp = WG[i] + 0.5 * dt * k1_WG
        WB_temp = WB[i] + 0.5 * dt * k1_WB
        k2_S, k2_E, k2_N, k2_WG, k2_WB = ode_derivatives(S_temp, E_temp, N_temp, WG_temp, WB_temp, sigma, theta, rho1, rho2, mu1, mu2)

        S_temp = S[i] + 0.5 * dt * k2_S
        E_temp = E[i] + 0.5 * dt * k2_E
        N_temp = N[i] + 0.5 * dt * k2_N
        WG_temp = WG[i] + 0.5 * dt * k2_WG
        WB_temp = WB[i] + 0.5 * dt * k2_WB
        k3_S, k3_E, k3_N, k3_WG, k3_WB = ode_derivatives(S_temp, E_temp, N_temp, WG_temp, WB_temp, sigma, theta, rho1, rho2, mu1, mu2)

        S_temp = S[i] + dt * k3_S
        E_temp = E[i] + dt * k3_E
        N_temp = N[i] + dt * k3_N
        WG_temp = WG[i] + dt * k3_WG
        WB_temp = WB[i] + dt * k3_WB
        k4_S, k4_E, k4_N, k4_WG, k4_WB = ode_derivatives(S_temp, E_temp, N_temp, WG_temp, WB_temp, sigma, theta, rho1, rho2, mu1, mu2)

        S[i+1] = S[i] + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E[i+1] = E[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        N[i+1] = N[i] + (dt / 6) * (k1_N + 2 * k2_N + 2 * k3_N + k4_N)
        WG[i+1] = WG[i] + (dt / 6) * (k1_WG + 2 * k2_WG + 2 * k3_WG + k4_WG)
        WB[i+1] = WB[i] + (dt / 6) * (k1_WB + 2 * k2_WB + 2 * k3_WB + k4_WB)
        T[i+1] = T[i] + dt

    return T, S, E, N, WG, WB
