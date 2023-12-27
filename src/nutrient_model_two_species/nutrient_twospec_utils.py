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
    #   1 = nutrient green
    #   2 = nutrient blue
    #   3 = soil
    #   4 = worm green
    #   5 = worm blue

    # start with equal number of everything
    soil_lattice = np.random.choice(np.arange(0, 6), size=(L, L)).astype(np.int8)
    return soil_lattice


@njit
def update(soil_lattice, L, rho1, rho2, theta1, theta2, sigma, delta):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the square lattice.
    rho1 : float
        Reproduction rate for green worms.
    rho2 : float
        Reproduction rate for blue worms.
    theta1 : float
        Death rate for green worms.
    theta2 : float
        Death rate for blue worms.
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

    if soil_lattice[site[0], site[1]] == 0:  # empty
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 3:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 3

    elif soil_lattice[site[0], site[1]] == 1:  # green nutrient
        is_filled = False
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 3:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 3
                is_filled = True
        if not is_filled:
            # decay to empty with rate delta
            if np.random.rand() < delta:
                soil_lattice[site[0], site[1]] = 0

    elif soil_lattice[site[0], site[1]] == 2:  # blue nutrient
        is_filled = False
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 3:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 3
                is_filled = True
        if not is_filled:
            # decay to empty with rate delta
            if np.random.rand() < delta:
                soil_lattice[site[0], site[1]] = 0


    elif soil_lattice[site[0], site[1]] == 4:  # green worm
        # check for death
        if np.random.rand() < theta1:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the worm
            soil_lattice[new_site[0], new_site[1]] = 4
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is blue nutrient
            if new_site_value == 2:
                # reproduce behind you
                if np.random.rand() < rho1:
                    soil_lattice[site[0], site[1]] = 4
            # check if the new site is soil
            elif new_site_value == 3:
                # leave nutrient behind
                soil_lattice[site[0], site[1]] = 1
            # check if the new site is a worm
            elif (new_site_value == 4) or (new_site_value == 5):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = new_site_value

    elif soil_lattice[site[0], site[1]] == 5:  # blue worm
        # check for death
        if np.random.rand() < theta2:
            soil_lattice[site[0], site[1]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1]]
            # move the worm
            soil_lattice[new_site[0], new_site[1]] = 5
            soil_lattice[site[0], site[1]] = 0
            # check if the new site is green nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1]] = 5
            # check if the new site is soil
            elif new_site_value == 3:
                # leave nutrient behind
                soil_lattice[site[0], site[1]] = 2
            # check if the new site is a worm
            elif (new_site_value == 4) or (new_site_value == 5):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = new_site_value


@njit
def run(n_steps, L, rho1, rho2, theta1, theta2, sigma, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    theta1 : float
        Death rate of green worms.
    theta2 : float
        Death rate of blue worms.
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
    soil_lattice = init_lattice(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update(soil_lattice, L, rho1, rho2, theta1, theta2, sigma, delta)
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
    #   1 = green nutrient
    #   2 = blue nutrient
    #   3 = soil
    #   4 = green worm
    #   5 = blue worm
    # start with equal number of everything
    soil_lattice = np.random.choice(np.arange(0, 6), size=(L, L, L))

    return soil_lattice

@njit
def update_stochastic_3D(soil_lattice, L, rho1, rho2, theta1, theta2, sigma, delta):
    """Update the lattice stochastically. Called once every timestep.

    The function mutates a global variable, to avoid slowdowns from numba primitives.
    It works by choosing a random site, and then giving a dynamics ascribed to the said site.
    
    Parameters:
    -----------
    soil_lattice : numpy.ndarray
        Lattice with bacteria randomly placed on it.
    L : int
        Side length of the square lattice.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    theta1 : float
        Death rate of green worms.
    theta2 : float
        Death rate of blue worms.
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

    if soil_lattice[site[0], site[1], site[2]] == 0:  # if empty
        # choose a random neighbour
        nbr = get_random_neighbour_3D(site, L)
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 3:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 3

    elif (soil_lattice[site[0], site[1], site[2]] == 1) or (soil_lattice[site[0], site[1], site[2]] == 2):  # if nutrient
        is_filled = False
        # choose a random neighbour
        nbr = get_random_neighbour_3D(site, L)
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 3:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 3
                is_filled = True
        if not is_filled:
            # decay to empty with rate delta
            if np.random.rand() < delta:
                soil_lattice[site[0], site[1], site[2]] = 0

    elif soil_lattice[site[0], site[1], site[2]] == 4:  # if green worm
        # check for death
        if np.random.rand() < theta1:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour_3D(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 4
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is blue nutrient
            if new_site_value == 2:
                # reproduce behind you
                if np.random.rand() < rho1:
                    soil_lattice[site[0], site[1], site[2]] = 4
            # check if the new site is soil
            elif new_site_value == 3:
                # leave nutrient behind
                soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value

    elif soil_lattice[site[0], site[1], site[2]] == 5:  # if blue worm
        # check for death
        if np.random.rand() < theta2:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour_3D(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 5
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is green nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1], site[2]] = 5
            # check if the new site is soil
            elif new_site_value == 3:
                # leave nutrient behind
                soil_lattice[site[0], site[1], site[2]] = 2
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value


@njit
def run_stochastic_3D(n_steps, L, rho1, rho2, theta1, theta2, sigma, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    theta1 : float
        Death rate of green worms.
    theta2 : float
        Death rate of blue worms.
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
    soil_lattice = init_lattice_3D(L)

    soil_lattice_data = np.zeros((len(steps_to_record), L, L, L), dtype=np.int8)

    for step in range(1, n_steps+1):
        update_stochastic_3D(soil_lattice, L, rho1, rho2, theta1, theta2, sigma, delta)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data




def ode_integrate(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the nutrient model.

    Parameters
    ----------
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm reproduction rate.
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

@njit
def ode_derivatives(S, E, N_G, N_B, W_G, W_B, sigma, theta1, theta2, rho1, rho2, delta):
    """Calculate the derivatives of S, E, N, W.

    This function is not called directly, but rather through `ode_integrate_rk4`
    
    Parameters
    ----------
    S : float
        Soil fraction.
    E : float
        Empty fraction.
    N_G : float
        Green nutrient fraction.
    N_B : float
        Blue nutrient fraction.
    W_G : float
        Green worm fraction.
    W_B : float
        Blue worm fraction.
    sigma : float
        Soil filling rate.
    theta1 : float
        Green worm death rate.
    theta2 : float
        Blue worm death rate.
    rho1 : float
        Green worm reproduction rate.
    rho2 : float
        Blue worm reproduction rate.
    delta : float
        Nutrient decay rate.

    Returns
    -------
    dS : float
        Derivative of soil fraction.
    dE : float
        Derivative of empty fraction.
    dN_G : float
        Derivative of green nutrient fraction.
    dN_B : float
        Derivative of blue nutrient fraction.
    dW_G : float
        Derivative of green worm fraction.
    dW_B : float
        Derivative of blue worm fraction.
    """

    dS = sigma*S*(E+N_G+N_B) - (W_G+W_B)*S
    dE = (1-rho1)*W_G*N_B + (1-rho2)*W_B*N_G + theta1*W_G + theta2*W_B - sigma*S*E + delta*(N_G+N_B)
    dN_G = W_G*S - W_B*N_G - sigma*S*N_G - delta*N_G
    dN_B = W_B*S - W_G*N_B - sigma*S*N_B - delta*N_B
    dW_G = rho1*W_G*N_B - theta1*W_G
    dW_B = rho2*W_B*N_G - theta2*W_B

    return dS, dE, dN_G, dN_B, dW_G, dW_B


@njit
def ode_integrate_rk4(sigma, theta1, theta2, rho1, rho2, delta, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method.
    
    Parameters
    ----------
    
    sigma : float
        Soil filling rate.
    theta1 : float
        Green worm death rate.
    theta2 : float
        Blue worm death rate.
    rho1 : float
        Green worm reproduction rate.
    rho2 : float
        Blue worm reproduction rate.
    delta : float
        Nutrient decay rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100_000.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.

    Returns
    -------
    T : ndarray
        List of times.
    S : ndarray
        List of soil fractions.
    E : ndarray
        List of empty fractions.
    NG : ndarray
        List of green nutrient fractions.
    NB : ndarray
        List of blue nutrient fractions.
    WG : ndarray
        List of green worm fractions.
    WB : ndarray
        List of blue worm fractions.
    """

    # initial condition with equal fractions: todo: Play around to see absorbing boundaries change in param space
    E_0, NG_0, NB_0, S_0, WG_0, WB_0 = 1/6, 1/6, 1/6, 1/6, 1/6, 1/6

    dt = stoptime / nsteps

    S = np.zeros(nsteps+1)
    E = np.zeros(nsteps+1)
    NG = np.zeros(nsteps+1)
    NB = np.zeros(nsteps+1)
    WG = np.zeros(nsteps+1)
    WB = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    S[0] = S_0
    E[0] = E_0
    NG[0] = NG_0
    NB[0] = NB_0
    WG[0] = WG_0
    WB[0] = WB_0
    T[0] = 0

    for i in range(nsteps):
        k1_S, k1_E, k1_NG, k1_NB, k1_WG, k1_WB = ode_derivatives(S[i], E[i], NG[i], NB[i], WG[i], WB[i], sigma, theta1, theta2, rho1, rho2, delta)

        S_temp = S[i] + 0.5 * dt * k1_S
        E_temp = E[i] + 0.5 * dt * k1_E
        NG_temp = NG[i] + 0.5 * dt * k1_NG
        NB_temp = NB[i] + 0.5 * dt * k1_NB
        WG_temp = WG[i] + 0.5 * dt * k1_WG
        WB_temp = WB[i] + 0.5 * dt * k1_WB
        k2_S, k2_E, k2_NG, k2_NB, k2_WG, k2_WB = ode_derivatives(S_temp, E_temp, NG_temp, NB_temp, WG_temp, WB_temp, sigma, theta1, theta2, rho1, rho2, delta)

        S_temp = S[i] + 0.5 * dt * k2_S
        E_temp = E[i] + 0.5 * dt * k2_E
        NG_temp = NG[i] + 0.5 * dt * k2_NG
        NB_temp = NB[i] + 0.5 * dt * k2_NB
        WG_temp = WG[i] + 0.5 * dt * k2_WG
        WB_temp = WB[i] + 0.5 * dt * k2_WB
        k3_S, k3_E, k3_NG, k3_NB, k3_WG, k3_WB = ode_derivatives(S_temp, E_temp, NG_temp, NB_temp, WG_temp, WB_temp, sigma, theta1, theta2, rho1, rho2, delta)

        S_temp = S[i] + dt * k3_S
        E_temp = E[i] + dt * k3_E
        NG_temp = NG[i] + dt * k3_NG
        NB_temp = NB[i] + dt * k3_NB
        WG_temp = WG[i] + dt * k3_WG
        WB_temp = WB[i] + dt * k3_WB
        k4_S, k4_E, k4_NG, k4_NB, k4_WG, k4_WB = ode_derivatives(S_temp, E_temp, NG_temp, NB_temp, WG_temp, WB_temp, sigma, theta1, theta2, rho1, rho2, delta)

        S[i+1] = S[i] + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E[i+1] = E[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        NG[i+1] = NG[i] + (dt / 6) * (k1_NG + 2 * k2_NG + 2 * k3_NG + k4_NG)
        NB[i+1] = NB[i] + (dt / 6) * (k1_NB + 2 * k2_NB + 2 * k3_NB + k4_NB)
        WG[i+1] = WG[i] + (dt / 6) * (k1_WG + 2 * k2_WG + 2 * k3_WG + k4_WG)
        WB[i+1] = WB[i] + (dt / 6) * (k1_WB + 2 * k2_WB + 2 * k3_WB + k4_WB)
        T[i+1] = T[i] + dt

    return T, S, E, NG, NB, WG, WB
