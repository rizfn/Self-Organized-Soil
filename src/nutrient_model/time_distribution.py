import numpy as np
import matplotlib.pyplot as plt
from nutrient_utils import update_stochastic, init_lattice, get_random_neighbour, update_stochastic_3D, init_lattice_3D, get_random_neighbour_3D
from numba import njit
from time import perf_counter


@njit
def calc_lifetimes(n_steps, L, rho, theta, sigma, delta, target_site=2):
    """Calculate the time distributions for target lifetimes.

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
    target_site : int, optional
        The target site to calculate the lifetime for, by default 2 (soil)

    Returns
    -------
    time_period_data : ndarray
        List of time periods for the target site.
    """
    soil_lattice = init_lattice(L)
    old_lattice = soil_lattice.copy()
    n_target_old = np.sum(soil_lattice == target_site)
    target_time_period = np.zeros((L, L), dtype=np.int32)

    time_period_data = np.zeros(n_steps, dtype=np.int32)

    for step in range(n_steps):
        update_stochastic(soil_lattice, L, rho, theta, sigma, delta)
        n_target = np.sum(soil_lattice == target_site)
        if n_target > n_target_old:  # if a target particle has been added
            changed_site = np.where(old_lattice != soil_lattice)  # the site in the lattice where the target particle was added
            target_time_period[changed_site[0][0], changed_site[1][0]] = 0
        elif n_target < n_target_old: # if a target particle has been removed
            changed_site = np.where(old_lattice != soil_lattice)  # the site in the lattice where the target particle was removed
            time_period_data[step] = target_time_period[changed_site[0][0], changed_site[1][0]]
        target_time_period += 1
        old_lattice = soil_lattice.copy()
        n_target_old = n_target

    return time_period_data[time_period_data != 0]


@njit
def update_soil_lifetimes(soil_lattice, L, rho, theta, sigma, delta, target_time_period):
    """Updates the lattice and calculates soil lifetimes. Called once every timestep.
    
    The function mutates a global variable, to avoid slowdowns from numba primitives.
    
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
    target_time_period : numpy.ndarray
        Lattice with the lifetimes of the target site.
    
    Returns:
    --------
    time_period : int
        The lifetime of the target site that died this step (0 if no deaths).
    """

    target_time_period += 1  # increment all time periods by 1
    time_period = 0  # default value of return

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L)

    if soil_lattice[site[0], site[1]] == 0:
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 2
                target_time_period[site[0], site[1]] = 0

    elif soil_lattice[site[0], site[1]] == 1:
        is_filled = False
        # choose a random neighbour
        nbr = get_random_neighbour(site, L)
        if soil_lattice[nbr[0], nbr[1]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1]] = 2
                target_time_period[site[0], site[1]] = 0
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
            new_site = get_random_neighbour(site, L)
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
                time_period = target_time_period[site[0], site[1]]
            # check if the new site is a worm
            elif new_site_value == 3:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = 3
    return time_period

@njit
def calc_soil_lifetimes(n_steps, L, rho, theta, sigma, delta):
    """Calculate the time distributions for soil lifetimes.

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

    Returns
    -------
    time_period_data : ndarray
        List of time periods for the target site.
    """
    soil_lattice = init_lattice(L)
    target_time_period = np.zeros((L, L), dtype=np.int32)

    time_period_data = np.zeros(n_steps, dtype=np.int32)

    for step in range(n_steps):
        time_period = update_soil_lifetimes(soil_lattice, L, rho, theta, sigma, delta, target_time_period)
        time_period_data[step] = time_period

    return time_period_data[time_period_data != 0]



@njit
def calc_lifetimes_3D(n_steps, L, rho, theta, sigma, delta, target_site=2):
    """Calculate the time distributions for target lifetimes.

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
    target_site : int, optional
        The target site to calculate the lifetime for, by default 2 (soil)

    Returns
    -------
    time_period_data : ndarray
        List of time periods for the target site.
    """
    soil_lattice = init_lattice_3D(L)
    old_lattice = soil_lattice.copy()
    n_target_old = np.sum(soil_lattice == target_site)
    target_time_period = np.zeros((L, L, L), dtype=np.int32)

    time_period_data = np.zeros(n_steps, dtype=np.int32)

    for step in range(n_steps):
        update_stochastic_3D(soil_lattice, L, rho, theta, sigma, delta)
        n_target = np.sum(soil_lattice == target_site)
        if n_target > n_target_old:  # if a target particle has been added
            changed_site = np.where(old_lattice != soil_lattice)  # the site in the lattice where the target particle was added
            target_time_period[changed_site[0][0], changed_site[1][0], changed_site[2][0]] = 0
        elif n_target < n_target_old: # if a target particle has been removed
            changed_site = np.where(old_lattice != soil_lattice)  # the site in the lattice where the target particle was removed
            time_period_data[step] = target_time_period[changed_site[0][0], changed_site[1][0], changed_site[2][0]]
        target_time_period += 1
        old_lattice = soil_lattice.copy()
        n_target_old = n_target

    return time_period_data[time_period_data != 0]


@njit
def update_soil_lifetimes_3D(soil_lattice, L, rho, theta, sigma, delta, target_time_period):
    """Updates the lattice and calculates lifetimes. Called once every timestep.

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
    target_time_period : numpy.ndarray
        Lattice with the lifetimes of the target site.
    
    Returns:
    --------
    time_period : int
        The lifetime of the target site that died this step (0 if no deaths).
    """

    target_time_period += 1  # increment all time periods by 1
    time_period = 0  # default value of return

    # select a random site
    site = np.random.randint(0, L), np.random.randint(0, L), np.random.randint(0, L)

    if soil_lattice[site[0], site[1], site[2]] == 0:
        # choose a random neighbour
        nbr = get_random_neighbour_3D(site, L)
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 2
                target_time_period[site[0], site[1], site[2]] = 0

    elif soil_lattice[site[0], site[1], site[2]] == 1:
        is_filled = False
        # choose a random neighbour
        nbr = get_random_neighbour_3D(site, L)
        if soil_lattice[nbr[0], nbr[1], nbr[2]] == 2:  # if neighbour is soil
            # fill with soil-filling rate
            if np.random.rand() < sigma:
                soil_lattice[site[0], site[1], site[2]] = 2
                target_time_period[site[0], site[1], site[2]] = 0
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
            new_site = get_random_neighbour_3D(site, L)
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
                time_period = target_time_period[site[0], site[1], site[2]]
            # check if the new site is a worm
            elif new_site_value == 3:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = 3
    return time_period


@njit
def calc_soil_lifetimes_3D(n_steps, L, rho, theta, sigma, delta):
    """Calculate the time distributions for target lifetimes.

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
    target_site : int, optional
        The target site to calculate the lifetime for, by default 2 (soil)

    Returns
    -------
    time_period_data : ndarray
        List of time periods for the target site.
    """
    soil_lattice = init_lattice_3D(L)
    target_time_period = np.zeros((L, L, L), dtype=np.int32)

    time_period_data = np.zeros(n_steps, dtype=np.int32)

    for step in range(n_steps):
        time_period = update_soil_lifetimes_3D(soil_lattice, L, rho, theta, sigma, delta, target_time_period)
        time_period_data[step] = time_period

    return time_period_data[time_period_data != 0]




def main():
    steps_per_latticepoint = 500
    rho = 1
    delta = 0
    theta = 0.06
    sigma = 0.37

    target_site = 2  # soil lifetimes

    # # 2D
    # L = 100
    # n_steps = steps_per_latticepoint * L**2
    # time_period_data = calc_lifetimes(n_steps, L, rho, theta, sigma, delta, target_site=target_site)

    # 3D
    L = 50
    n_steps = steps_per_latticepoint * L**3
    start_time = perf_counter()
    # time_period_data = calc_lifetimes_3D(n_steps, L, rho, theta, sigma, delta, target_site=target_site)
    time_period_data = calc_soil_lifetimes_3D(n_steps, L, rho, theta, sigma, delta)
    time = perf_counter() - start_time
    print(f"{L=},\t{time=}")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].grid(True, alpha=0.5)
    axs[1].grid(True, alpha=0.5)

    plt.suptitle(f"Time distribution for soil\n{L=}, {rho=}, {theta=}, {sigma=}, {delta=}")

    time_period_data = time_period_data / L**3
    axs[0].hist(time_period_data, bins=100, alpha=0.8, linewidth=0.5, edgecolor="black")
    axs[0].set_xlabel('Soil Lifetime (steps / L^3)')
    axs[0].set_ylabel('Frequency')
    axs[1].hist(time_period_data, bins=100, alpha=0.8, linewidth=0.5, edgecolor="black")
    axs[1].set_xlabel('Soil Lifetime (steps / L^3)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_yscale("log")
    plt.savefig(f"src/nutrient_model/plots/soil_lifetimes_{theta=}_{sigma=}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
