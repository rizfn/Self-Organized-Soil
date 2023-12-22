import numpy as np
import matplotlib.pyplot as plt
from nutrient_utils import update_stochastic, init_lattice, update_stochastic_3D, init_lattice_3D
from numba import njit


# todo: Optimize by putting it in update, instead of returning and checking changes in lattice
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
    L = 20
    n_steps = steps_per_latticepoint * L**3
    time_period_data = calc_lifetimes_3D(n_steps, L, rho, theta, sigma, delta, target_site=target_site)

    plt.hist(time_period_data, bins=100)
    plt.show()




if __name__ == "__main__":
    main()
