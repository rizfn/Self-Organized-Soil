import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import run_stochastic, run_stochastic_wellmixed, run_stochastic_3D


def run_raster_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the rasterscan for the stochastic case for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho : float
        Reproduction rate.
    theta_list : ndarray
        List of death rates.
    sigma_list : ndarray
        List of soil filling rates.
    delta : float
        Nutrient decay rate.
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
        soil_lattice_data = run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def run_raster_stochastic_3D(n_steps, L, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the rasterscan for the stochastic 3D case for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the cubic lattice.
    rho : float
        Reproduction rate.
    theta_list : ndarray
        List of death rates.
    sigma_list : ndarray
        List of soil filling rates.
    delta : float
        Nutrient decay rate.
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
        soil_lattice_data = run_stochastic_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def main():

    # initialize the parameters
    steps_per_latticepoint = 500  # number of bacteria worm per lattice point
    L = 50  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**2  # number of bacteria moves
    rho = 1  # reproduction rate
    delta = 0
    theta_list = np.linspace(0, 0.3, 20)  # death rate
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate

    # NEIGHBOURS
    # soil_lattice_data = run_raster_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, np.geomspace(100, n_steps, int(np.log10(n_steps/100))+1, dtype=np.int32))
    soil_lattice_data = run_raster_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, np.linspace(n_steps//2, n_steps, 5, dtype=np.int32))
    soil_lattice_data = pd.DataFrame(soil_lattice_data)
    soil_lattice_data.to_json(f"docs/data/nutrient/lattice_oscillation_{L=}_{rho=}_{delta=}.json", orient="records")

    # # NEIGHBOURS, large lattice max timestep
    # soil_lattice_data = run_raster_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, np.array([n_steps]))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/large_lattice_{rho=}_{delta=}.json", orient="records")

    # # 3D
    # L = 75
    # n_steps = steps_per_latticepoint * L**3
    # # soil_lattice_data = run_raster_stochastic_3D(n_steps, L, rho, theta_list, sigma_list, delta, np.array([n_steps]))  # max timestep
    # soil_lattice_data = run_raster_stochastic_3D(n_steps, L, rho, theta_list, sigma_list, delta, np.linspace(n_steps//2, n_steps, 5, dtype=np.int32))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/lattice3D_{rho=}_{delta=}.json", orient="records")

    # # WELLMIXED
    # soil_lattice_data = run_raster_stochastic_wellmixed(n_steps, L, r, d, s, np.geomspace(100, n_steps, int(np.log10(n_steps/100))+1, dtype=np.int32))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/two_species/wellmixed_soil_neighbours_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
