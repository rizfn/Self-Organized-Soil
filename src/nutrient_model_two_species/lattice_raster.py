import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_twospec_utils import run, run_stochastic_3D


def run_raster(n_steps, L, rho1, rho2, theta1, theta2_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the rasterscan for the stochastic case for n_steps timesteps.
    
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
    theta2_list : ndarray
        List of death rates of blue worms.
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
    grid = np.meshgrid(theta2_list, sigma_list)
    ts_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_lattice_list = []
    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta2, sigma = ts_pairs[i]
        soil_lattice_data = run(n_steps, L, rho1, rho2, theta1, theta2, sigma, delta, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"theta1": theta1, "theta2":theta2, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
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
    rho1 = 1  # reproduction rate for green
    rho2 = 1  # reproduction rate for blue
    delta = 0
    theta1 = 0.02  # death rate for green
    theta2_list = np.linspace(0, 0.1, 20)  # death rate
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate

    # NEIGHBOURS
    soil_lattice_data = run_raster(n_steps, L, rho1, rho2, theta1, theta2_list, sigma_list, delta, np.linspace(n_steps//2, n_steps, 5, dtype=np.int32))
    soil_lattice_data = pd.DataFrame(soil_lattice_data)
    soil_lattice_data.to_json(f"docs/data/nutrient_twospec/soil_lattice_{L=}_{rho1=}_{rho2=}_{delta=}_{theta1=}.json", orient="records")

    # # NEIGHBOURS, large lattice max timestep
    # soil_lattice_data = run_raster_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, np.array([n_steps]))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/large_lattice_{rho=}_{delta=}.json", orient="records")

    # # 3D  # todo: double check the saving algorithm works
    # def calculate_fractions(matrix):
    #     flattened = np.array(matrix).flatten()
    #     fractions = pd.Series(flattened).value_counts(normalize=True).sort_index()
    #     fractions = fractions.reindex(range(4), fill_value=0)
    #     return fractions

    # L = 75
    # n_steps = steps_per_latticepoint * L**3
    # # soil_lattice_data = run_raster_stochastic_3D(n_steps, L, rho, theta_list, sigma_list, delta, np.array([n_steps]))  # max timestep
    # soil_lattice_data = run_raster_stochastic_3D(n_steps, L, rho, theta_list, sigma_list, delta, np.linspace(n_steps//2, n_steps, 5, dtype=np.int32))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data[['vacancy', 'nutrient', 'soil', 'worm']] = soil_lattice_data['soil_lattice'].apply(calculate_fractions).apply(pd.Series)
    # grouped = soil_lattice_data.groupby('step')
    # for i, (group_name, group_data) in enumerate(grouped):
    #     group_data.to_json(f'docs/data/nutrient/lattice3D_{L=}_{rho=}_{delta=}/step{i}.json', orient='records')

    # # WELLMIXED
    # soil_lattice_data = run_raster_stochastic_wellmixed(n_steps, L, r, d, s, np.geomspace(100, n_steps, int(np.log10(n_steps/100))+1, dtype=np.int32))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/two_species/wellmixed_soil_neighbours_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
