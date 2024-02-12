import numpy as np
from twospec_samenutrient_utils import run, run_3D
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool




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
    n_steps, L, sigma, theta, rho1, mu1, rho2, mu2, steps_to_record = params
    soil_lattice_data = run(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record)
    soil_lattice_list = []
    for step in steps_to_record:
        soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list

def run_raster_2D(n_steps, L, sigma_list, theta_list, rho1, mu1, rho2, mu2, steps_to_record=np.array([100, 1000, 10000, 100000])):
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

    grid = np.meshgrid(sigma_list, theta_list)
    sigma_theta_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
    # Add the other parameters to each pair of rho and mu
    params = [(n_steps, L, sigma, theta, rho1, mu1, rho2, mu2, steps_to_record) for sigma, theta in sigma_theta_pairs]
    soil_lattice_data = []
    with Pool() as p:
        with tqdm(total=len(params)) as pbar:
            for result in p.imap(run_simulation_2D, params):
                pbar.update()
                soil_lattice_data.extend(result)
    return soil_lattice_data


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
    n_steps, L, sigma, theta, rho1, mu1, rho2, mu2, steps_to_record = params
    soil_lattice_data = run_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record)
    soil_lattice_list = []
    for step in steps_to_record:
        soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list

def run_raster_3D(n_steps, L, sigma_list, theta_list, rho1, mu1, rho2, mu2, steps_to_record=np.array([100, 1000, 10000, 100000])):
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

    grid = np.meshgrid(sigma_list, theta_list)
    sigma_theta_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of rho and mu
    # Add the other parameters to each pair of rho and mu
    params = [(n_steps, L, sigma, theta, rho1, mu1, rho2, mu2, steps_to_record) for sigma, theta in sigma_theta_pairs]
    soil_lattice_data = []
    with Pool() as p:
        with tqdm(total=len(params)) as pbar:
            for result in p.imap(run_simulation_3D, params):
                pbar.update()
                soil_lattice_data.extend(result)
    return soil_lattice_data





def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    sigma_list = np.linspace(0, 1, 20)
    theta_list = np.linspace(0, 0.1, 20)
    rho1 = 0.5
    mu1 = 1
    rho2 = 1
    mu2 = 0

    # 3D
    L = 50  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    steps_to_record = np.linspace(n_steps//2, n_steps, 5, dtype=np.int32)
    raster_data = run_raster_3D(n_steps, L, sigma_list, theta_list, rho1, mu1, rho2, mu2, steps_to_record)
    raster_data = pd.DataFrame(raster_data)

    def calculate_fractions(matrix):
        flattened = np.array(matrix).flatten()
        fractions = pd.Series(flattened).value_counts(normalize=True).sort_index()
        fractions = fractions.reindex(range(5), fill_value=0)
        return fractions
    
    raster_data[['vacancy', 'nutrient', 'soil', 'green', 'blue']] = raster_data['soil_lattice'].apply(calculate_fractions).apply(pd.Series)
    grouped = raster_data.groupby('step')
    dir_path = f'docs/data/twospec_samenutrient/lattice3D_{L=}_{rho1=}_{mu1=}_{rho2=}_{mu2=}/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i, (group_name, group_data) in enumerate(grouped):
        group_data.to_json(dir_path + f'/step{i}.json', orient='records')


    # # 2D
    # L = 100  # side length of the square lattice
    # n_steps = steps_per_latticepoint * L**2  # 2D
    # steps_to_record = np.linspace(n_steps//2, n_steps, 5, dtype=np.int32)
    # raster_data = run_raster_2D(n_steps, L, sigma, theta, rho1, mu1, rho2_list, mu2_list, steps_to_record)
    # raster_data = pd.DataFrame(raster_data)
    # raster_data.to_json(f"docs/data/twospec_samenutrient/lattice_{L=}_{sigma=}_{theta=}_{rho1=}_{mu1=}.json", orient="records")





if __name__ == "__main__":
    main()
