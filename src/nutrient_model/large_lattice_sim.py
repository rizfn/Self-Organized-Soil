import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import run_stochastic


def run_multisim_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the multiple sims for the stochastic case for n_steps timesteps.
    
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
    soil_lattice_list = []
    for i in tqdm(range(len(theta_list))):  # todo: parallelize
        theta = theta_list[i]
        sigma = sigma_list[i]
        soil_lattice_data = run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    L = 400  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**2  # number of timesteps to run the simulation for
    rho = 1  # reproduction rate
    delta = 0
    # theta_list = np.array([0.14, 0.14, 0.1])
    # sigma_list = np.array([0.2, 0.8, 0.55])
    theta_list = np.array([0.14, 0.1, 0.06])
    sigma_list = np.array([0.3, 0.3, 0.3])

    soil_lattice_data = run_multisim_stochastic(n_steps, L, rho, theta_list, sigma_list, delta, steps_to_record=np.array([n_steps]))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)
    soil_lattice_data.to_json(f"docs/data/nutrient/large_lattice_{rho=}_{delta=}.json", orient="records")



if __name__ == "__main__":
    main()
