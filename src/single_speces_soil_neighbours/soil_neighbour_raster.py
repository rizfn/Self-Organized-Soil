import numpy as np
from tqdm import tqdm
import pandas as pd
from soil_neighbour_utils import run_stochastic, run_stochastic_wellmixed


def run_raster_stochastic(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the rasterscan for the stochastic case for n_steps timesteps.
    
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
        soil_lattice_data = run_stochastic(n_steps, L, r, d, s, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def run_raster_stochastic_wellmixed(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the rasterscan for the well-mixed stochastic case for n_steps timesteps.
    
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
        soil_lattice_data = run_stochastic_wellmixed(n_steps, L, r, d, s, steps_to_record=steps_to_record)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 10_000_000  # number of bacteria moves (more for stochastic)
    L = 50  # side length of the square lattice
    r = 1  # reproduction rate
    d = np.linspace(0, 0.3, 20)  # death rate
    s = np.linspace(0, 1, 20)  # soil filling rate (same as mean field)

    # soil_lattice_data = run_raster_stochastic(n_steps, L, r, d, s, np.geomspace(100, n_steps, int(np.log10(n_steps/100))+1, dtype=np.int32))
    soil_lattice_data = run_raster_stochastic_wellmixed(n_steps, L, r, d, s, np.geomspace(100, n_steps, int(np.log10(n_steps/100))+1, dtype=np.int32))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    # soil_lattice_data.to_json(f"docs/data/single_species/soil_neighbours_{r=}.json", orient="records")
    soil_lattice_data.to_json(f"docs/data/single_species/wellmixed_soil_neighbours_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
