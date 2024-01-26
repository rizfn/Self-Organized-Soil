import numpy as np
import matplotlib.pyplot as plt
from twospec_samenutrient_utils import get_random_neighbour_3D
from numba import njit


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
    #   5 = purple worm
    # start with equal number of everything
    soil_lattice = np.random.choice(np.arange(0, 6), size=(L, L, L))

    return soil_lattice

@njit
def update_3D(soil_lattice, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3):
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
    rho3 : float
        Purple worm reproduction rate.
    mu1 : float
        Green worm nutrient-creating rate.
    mu2 : float
        Blue worm nutrient-creating rate.
    mu3 : float
        Purple worm nutrient-creating rate.
            
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
            elif (new_site_value == 3) or (new_site_value == 4) or (new_site_value == 5):
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
            elif (new_site_value == 3) or (new_site_value == 4) or (new_site_value == 5):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value

    elif soil_lattice[site[0], site[1], site[2]] == 5:  # if purple worm
        # check for death
        if np.random.rand() < theta:
            soil_lattice[site[0], site[1], site[2]] = 0
        else:
            # move into a neighbour
            new_site = get_random_neighbour_3D(site, L)
            # check the value of the new site
            new_site_value = soil_lattice[new_site[0], new_site[1], new_site[2]]
            # move the worm
            soil_lattice[new_site[0], new_site[1], new_site[2]] = 5
            soil_lattice[site[0], site[1], site[2]] = 0
            # check if the new site is nutrient
            if new_site_value == 1:
                # reproduce behind you
                if np.random.rand() < rho3:
                    soil_lattice[site[0], site[1], site[2]] = 5
            # check if the new site is soil
            elif new_site_value == 2:
                # leave nutrient behind
                if np.random.rand() < mu3:
                    soil_lattice[site[0], site[1], site[2]] = 1
            # check if the new site is a worm
            elif (new_site_value == 3) or (new_site_value == 4) or (new_site_value == 5):
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value



@njit
def run_3D(n_steps, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
    rho3 : float
        Reproduction rate of purple worms.
    mu1 : float
        Nutrient-creating rate of green worms.
    mu2 : float
        Nutrient-creating rate of blue worms.
    mu3 : float
        Nutrient-creating rate of purple worms.
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
        update_3D(soil_lattice, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3)
        if step in steps_to_record:
            soil_lattice_data[steps_to_record == step] = soil_lattice

    return soil_lattice_data


@njit
def run_timeseries_3D(n_steps, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the cubic lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Death rate.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    rho3 : float
        Reproduction rate of purple worms.
    mu1 : float
        Nutrient creation rate of green worms.
    mu2 : float
        Nutrient creation rate of blue worms.
    mu3 : float
        Nutrient creation rate of purple worms.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    emptys : ndarray
        Fraction of empty lattice points at each timestep.
    nutrients : ndarray
        Fraction of nutrient lattice points at each timestep.
    soil : ndarray
        Fraction of soil lattice points at each timestep.
    greens : ndarray
        Fraction of green worm lattice points at each timestep.
    blues : ndarray
        Fraction of blue worm lattice points at each timestep.
    purples : ndarray
        Fraction of purple worm lattice points at each timestep.
    """
    soil_lattice = init_lattice_3D(L)

    emptys = np.zeros(len(steps_to_record), dtype=np.int32)
    nutrients = np.zeros_like(emptys)
    soil = np.zeros_like(emptys)
    greens = np.zeros_like(emptys)
    blues = np.zeros_like(emptys)
    purples = np.zeros_like(emptys)
    i = 0  # indexing for recording steps

    for step in range(n_steps+1):
        update_3D(soil_lattice, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3)
        if step in steps_to_record:
            flattened = soil_lattice.flatten()
            counts = np.bincount(flattened, minlength=6)
            emptys[i] = counts[0]
            nutrients[i] = counts[1]
            soil[i] = counts[2]
            greens[i] = counts[3]
            blues[i] = counts[4]
            purples[i] = counts[5]
            i += 1

    emptys = emptys / L**3
    nutrients = nutrients / L**3
    soil = soil / L**3
    greens = greens / L**3
    blues = blues / L**3
    purples = purples / L**3

    return emptys, nutrients, soil, greens, blues, purples




def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    sigma = 0.5
    theta = 0.025
    rho1 = 0.5
    mu1 = 0.5
    rho2 = 1
    mu2 = 0
    rho3 = 0.8
    mu3 = 0

    L = 50  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    steps_to_record = np.arange(0, n_steps+1, L**3, dtype=np.int32)
    emptys, nutrients, soil, greens, blues, purples = run_timeseries_3D(n_steps, L, sigma, theta, rho1, rho2, rho3, mu1, mu2, mu3, steps_to_record=steps_to_record)
    steps_to_record = steps_to_record / L**3


    # L = 250  # side length of the square lattice
    # n_steps = steps_per_latticepoint * L**2  # 2D
    # steps_to_record = np.arange(0, n_steps+1, L**2, dtype=np.int32)
    # emptys, nutrients, soil, greens, blues = run_timeseries(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record=steps_to_record)
    # steps_to_record = steps_to_record / L**2

    
    fig, axs = plt.subplots(figsize=(10, 6))


    axs.plot(steps_to_record, soil, label="soil", c="brown")
    axs.plot(steps_to_record, emptys, label="emptys", c="grey")
    axs.plot(steps_to_record, nutrients, label="nutrients", c="lawngreen")
    axs.plot(steps_to_record, greens, label="green worms", c="green")
    axs.plot(steps_to_record, blues, label="blue worms", c="blue")
    axs.plot(steps_to_record, purples, label="purple worms", c="purple")
    axs.set_title(f"{L=}, {sigma=}, {theta=}, {rho1=}, {mu1=}, {rho2=}, {mu2=}")
    axs.set_xlabel(r"Timestep / L$^3$")
    # axs.set_xlabel(r"Timestep / L$^2$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    # plt.savefig('src/two_species_same_nutrient/plots/lattice_timeseries/_parasite_test.png', dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
