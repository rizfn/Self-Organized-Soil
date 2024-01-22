## VERY different: now only one species needs the nutrient (green), while blue 'eats' green.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter
from twospec_samenutrient_utils import init_lattice, init_lattice_3D, get_random_neighbour, get_random_neighbour_3D
import pandas as pd
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool


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
            # check if the new site is a green worm
            if new_site_value == 3:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1]] = 4
            elif new_site_value == 4:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1]] = new_site_value

def run_simulation_2D(params):
    """Run the stochastic simulation for n_steps timesteps, and return if green/blue worms are alive in the end.
    
    Parameters
    ----------
    params : tuple
        Tuple of parameters to run the simulation with.
        
    Returns
    -------
    alive_information : dict
        Dictionary of the parameters and whether the soil and green/blue worms are alive at the end of the simulation.
    """
    n_steps, L, sigma, theta, rho1, mu1, rho2, mu2 = params
    soil_alive, green_alive, blue_alive = run_alive_2D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2)
    return {"rho1": rho1, "rho2":rho2, "mu1": mu1, "mu2":mu2, "soil_alive":soil_alive, "green_alive": green_alive, "blue_alive": blue_alive}

def run_raster_living_2D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list):
    """Run the parallelized rasterscan for the 2D case for n_steps timesteps.
    
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
    mu1 : float
        Nutrient creation rate of green worms.
    rho2_list : list
        List of reproduction rates of blue worms.
    mu2_list : list
        List of nutrient creation rates of blue worms.
        
    Returns
    -------
    alive_information : list
        List of information on whether the soil and green/blue worms are alive at the end of the simulation and parameters.
    """

    grid = np.meshgrid(rho2_list, mu2_list)
    rho_mu_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
    # Add the other parameters to each pair of rho and mu
    params = [(n_steps, L, sigma, theta, rho1, mu1, rho2, mu2) for rho2, mu2 in rho_mu_pairs]
    alive_information = []
    with Pool() as p:
        with tqdm(total=len(params)) as pbar:
            for result in p.imap(run_simulation_2D, params):
                pbar.update()
                alive_information.append(result)
    return alive_information

@njit
def run_alive_2D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2):
    """Run the stochastic simulation for n_steps timesteps, and return if green/blue worms are alive in the end.

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
    soilAlive : bool
        Whether soil is alive at the end of the simulation.
    greenAlive : bool
        Whether green worms are alive at the end of the simulation.
    blueAlive : bool
        Whether blue worms are alive at the end of the simulation.
    """
    soil_lattice = init_lattice(L)
    for i in range(1, n_steps+1):
        update(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2)
    flattened = soil_lattice.flatten()
    counts = np.bincount(flattened, minlength=5)
    soil_alive = counts[2] > 0
    green_alive = counts[3] > 0
    blue_alive = counts[4] > 0
    return soil_alive, green_alive, blue_alive


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
            # check if the new site is green
            if new_site_value == 3:
                # reproduce behind you
                if np.random.rand() < rho2:
                    soil_lattice[site[0], site[1], site[2]] = 4
            # check if the new site is a blue worm
            elif new_site_value == 4:
                # keep both with worms (undo the vacant space in original site)
                soil_lattice[site[0], site[1], site[2]] = new_site_value

def run_simulation_3D(params):
    """Run the stochastic simulation for n_steps timesteps, and return if green/blue worms are alive in the end.
    
    Parameters
    ----------
    params : tuple
        Tuple of parameters to run the simulation with.
        
    Returns
    -------
    alive_information : dict
        Dictionary of the parameters and whether the soil and green/blue worms are alive at the end of the simulation.
    """
    n_steps, L, sigma, theta, rho1, mu1, rho2, mu2 = params
    soil_alive, green_alive, blue_alive = run_alive_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2)
    return {"rho1": rho1, "rho2":rho2, "mu1": mu1, "mu2":mu2, "soil_alive":soil_alive, "green_alive": green_alive, "blue_alive": blue_alive}

def run_raster_living_3D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list):
    """Run the parallelized rasterscan for the 3D case for n_steps timesteps.
    
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
    mu1 : float
        Nutrient creation rate of green worms.
    rho2_list : list
        List of reproduction rates of blue worms.
    mu2_list : list
        List of nutrient creation rates of blue worms.
        
    Returns
    -------
    alive_information : list
        List of information on whether the soil and green/blue worms are alive at the end of the simulation and parameters.
    """

    grid = np.meshgrid(rho2_list, mu2_list)
    rho_mu_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
    # Add the other parameters to each pair of rho and mu
    params = [(n_steps, L, sigma, theta, rho1, mu1, rho2, mu2) for rho2, mu2 in rho_mu_pairs]
    alive_information = []
    with Pool() as p:
        with tqdm(total=len(params)) as pbar:
            for result in p.imap(run_simulation_3D, params):
                pbar.update()
                alive_information.append(result)
    return alive_information

@njit
def run_alive_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2):
    """Run the stochastic simulation for n_steps timesteps, and return if green/blue worms are alive in the end.

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
    mu1 : float
        Nutrient creation rate of green worms.
    mu2 : float
        Nutrient creation rate of blue worms.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soilAlive : bool
        Whether soil is alive at the end of the simulation.
    greenAlive : bool
        Whether green worms are alive at the end of the simulation.
    blueAlive : bool
        Whether blue worms are alive at the end of the simulation.
    """
    soil_lattice = init_lattice_3D(L)
    for i in range(1, n_steps+1):
        update_3D(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2)
    flattened = soil_lattice.flatten()
    counts = np.bincount(flattened, minlength=5)
    soil_alive = counts[2] > 0
    green_alive = counts[3] > 0
    blue_alive = counts[4] > 0
    return soil_alive, green_alive, blue_alive





def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    sigma = 0.5
    theta = 0.025
    rho1 = 0.5
    mu1 = 0.5
    rho2_list = np.linspace(0, 1, 20)
    mu2_list = np.linspace(0, 1, 20)

    # 3D
    L = 50  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    alive_information = run_raster_living_3D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list)

    # # 2D
    # L = 250  # side length of the square lattice
    # n_steps = steps_per_latticepoint * L**2  # 2D
    # alive_information = run_raster_living_2D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list)


    alive_information = pd.DataFrame(alive_information)

    def map_colors(row):
        if row['green_alive'] and row['blue_alive']:
            return 0  # Yellow
        elif row['green_alive']:
            return 1  # Green
        elif row['blue_alive']:
            return 2  # Blue
        elif row['soil_alive']:
            return 3  # Brown
        else:
            return 4  # Empty

    alive_information['color'] = alive_information.apply(map_colors, axis=1)
    pivot_df = alive_information.pivot(index='rho2', columns='mu2', values='color')
    cmap = colors.ListedColormap(['yellow', 'green', 'blue', 'brown', 'white'])

    data = pivot_df.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Alive worms\n{L=}, {n_steps=:}, {sigma=}, {theta=}, {rho1=}, {mu1=}")
    ax.imshow(data, cmap=cmap, origin='lower', vmin=0, vmax=4)  # Use origin='lower' to start the plot from the bottom left
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, t:round (mu2_list[int(v)],2) if v<len(mu2_list) else ''))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, t: round(rho2_list[int(v)],2) if v<len(rho2_list) else ''))
    ax.set_xlabel('mu2')
    ax.set_ylabel('rho2')
    # plt.savefig(f'src/two_species_same_nutrient/plots/alive_raster_pew/lattice3D_{L=}_{sigma=}_{theta=}.png', dpi=300)
    # plt.savefig(f'src/two_species_same_nutrient/plots/alive_raster_pew/lattice2D_{L=}_{sigma=}_{theta=}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
