import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter
from twospec_samenutrient_utils import update_3D, init_lattice_3D
import pandas as pd
from numba import njit
from tqdm import tqdm
from multiprocessing import Pool


def run_simulation(params):
    n_steps, L, sigma, theta, rho1, mu1, rho2, mu2 = params
    soil_alive, green_alive, blue_alive = run_alive_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2)
    return {"rho1": rho1, "rho2":rho2, "mu1": mu1, "mu2":mu2, "soil_alive":soil_alive, "green_alive": green_alive, "blue_alive": blue_alive}

def run_raster_living_3D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list):
    grid = np.meshgrid(rho2_list, mu2_list)
    rho_mu_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
    # Add the other parameters to each pair of rho and mu
    params = [(n_steps, L, sigma, theta, rho1, mu1, rho2, mu2) for rho2, mu2 in rho_mu_pairs]
    alive_information = []
    with Pool() as p:
        with tqdm(total=len(params)) as pbar:
            for result in p.imap(run_simulation, params):
                pbar.update()
                alive_information.append(result)
    return alive_information

# def run_raster_living_3D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list):
#     """Run the rasterscan for the 3D case for n_steps timesteps.
    
#     Parameters
#     ----------
#     n_steps : int
#         Number of timesteps to run the simulation for.
#     L : int
#         Side length of the cubic lattice.
#     sigma : float
#         Soil filling rate.
#     theta : float
#         Death rate.
#     rho1 : float
#         Reproduction rate of green worms.
#     rho2 : float
#         Reproduction rate of blue worms.
#     mu1 : float
#         Nutrient creation rate of green worms.
#     mu2 : float
#         Nutrient creation rate of blue worms.

#     Returns
#     -------
#     living_information : list
#         List of information on whether the green/blue worms are alive at the end of the simulation and parameters.
#     """
#     grid = np.meshgrid(rho2_list, mu2_list)
#     rho_mu_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
#     alive_information = []
#     for i in tqdm(range(len(rho_mu_pairs))):  # todo: parallelize
#         rho2, mu2 = rho_mu_pairs[i]
#         soil_alive, green_alive, blue_alive = run_alive_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2)
#         alive_information.append({"rho1": rho1, "rho2":rho2, "mu1": mu1, "mu2":mu2, "soil_alive":soil_alive, "green_alive": green_alive, "blue_alive": blue_alive})
#     return alive_information


@njit
def run_alive_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2):
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
    L = 50  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    sigma = 0.5
    theta = 0.025
    rho1 = 0.5
    rho2_list = np.linspace(0, 1, 20)
    mu1 = 0.5
    mu2_list = np.linspace(0, 1, 20)

    alive_information = run_raster_living_3D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list)
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
    plt.savefig(f'src/two_species_same_nutrient/plots/alive_raster/lattice_{L=}_{sigma=}_{theta=}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
