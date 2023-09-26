import numpy as np
from tqdm import tqdm
import pandas as pd
from single_species_utils import run, ode_integrate


def run_raster(n_steps, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the simulation for n_steps timesteps.
    
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
    grid = np.meshgrid(d_list, s_list)
    ds_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_lattice_list = []
    for i in tqdm(range(len(ds_pairs))):  # todo: parallelize
        d, s = ds_pairs[i]
        T, S, E, B = ode_integrate(s, d, r, stoptime=n_steps, nsteps=100_000)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "vacancy": E[step], "bacteria": B[step], "soil": S[step]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of bacteria moves
    r = 1  # reproduction rate
    # d = np.linspace(0, 0.3, 20)  # death rate
    # s = np.linspace(0, 0.8, 20)  # soil filling rate
    d = np.linspace(0, 1, 100)  # dense death rate
    s = np.linspace(0, 1, 100)  # dense soil filling rate

    soil_lattice_data = run_raster(n_steps, r, d, s)

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    # soil_lattice_data.to_json(f"docs/data/single_species/mean_field_data_{r=}.json", orient="records")
    soil_lattice_data.to_json(f"docs/data/single_species/mean_field_data_dense_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
