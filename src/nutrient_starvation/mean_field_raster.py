import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_starvation_utils import ode_integrate_rk4


def run_raster(n_steps, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Scans the ODE integrator over s and d for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    r : float
        Reproduction rate.
    d_list : ndarray
        List of death rates.
    s_list : ndarray
        List of soil filling rates.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].
    
    Returns
    -------
    soil_lattice_list : list
        List of soil_lattice data for specific timesteps and parameters.
    """
    grid = np.meshgrid(theta_list, sigma_list)
    ts_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_lattice_list = []
    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta, sigma = ts_pairs[i]
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=100_000)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "vacancy": E[step], "nutrient": N[step], "worm":W[step], "soil": S[step]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 1, 20)  # death rate
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate

    soil_lattice_data = run_raster(n_steps, rho, theta_list, sigma_list, delta)

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/nutrient_starvation/meanfield_{rho=}_{delta=}.json", orient="records")



if __name__ == "__main__":
    main()
