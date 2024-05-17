import numpy as np
from tqdm import tqdm
import pandas as pd
from soil_neighbour_utils import run_predatorprey
import multiprocessing as mp
import concurrent.futures
from functools import partial


def run_single_simulation(params, n_steps, L, r, steps_to_record):
    d, s = params
    soil_lattice_data = run_predatorprey(n_steps, L, r, d, s, steps_to_record=steps_to_record)
    return [{"d": d, "s": s, "step": step, "soil_lattice": soil_lattice_data[steps_to_record == step][0]} for step in steps_to_record]


def run_raster_predatorprey(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
    grid = np.meshgrid(d_list, s_list)
    ds_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s

    num_processes = max(1, mp.cpu_count() - 4)  # Leave 4 cores free
    with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
        soil_lattice_list = list(tqdm(executor.map(partial(run_single_simulation, n_steps=n_steps, L=L, r=r, steps_to_record=steps_to_record), ds_pairs), total=len(ds_pairs)))

    # Flatten the list of lists
    soil_lattice_list = [item for sublist in soil_lattice_list for item in sublist]

    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 10_000_000  # number of bacteria moves (more for stochastic)
    L = 100  # side length of the square lattice
    r = 1  # reproduction rate
    # d = np.linspace(0, 0.3, 20)  # death rate
    d = np.linspace(0, 1, 20)  # death rate
    s = np.linspace(0, 1, 20)  # soil filling rate

    # soil_lattice_data = run_raster_predatorprey(n_steps, L, r, d, s, np.linspace(n_steps//2, n_steps, 5, dtype=np.int32))
    soil_lattice_data = run_raster_predatorprey(n_steps, L, r, d, s, np.array([n_steps]))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    # soil_lattice_data.to_json(f"docs/data/single_species/predatorprey_stochastic_{r=}.json", orient="records")
    soil_lattice_data.to_json(f"docs/data/single_species/predatorprey_stochastic_mfcomparison_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
