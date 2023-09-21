import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from single_species_utils import init_lattice

def ode_integrate(L, s, d, r, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the single species model.

    Parameters
    ----------
    L : int
        Side length of the square lattice.
    s : float
        Soil filling rate.
    d : float
        Death rate.
    r : float
        Reproduction rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    
    Returns
    -------
    T : list
        List of times.
    S : list
        List of soil fractions.
    E : list
        List of empty fractions.
    B : list
        List of bacteria fractions.
    """

    N_sites = L**2  # number of sites
    B_0 = int(N_sites / 10)  # initial number of bacteria
    E_0 = int((N_sites - B_0) / 2)  # initial number of empty sites
    S_0 = N_sites - B_0 - E_0  # initial number of soil sites

    dt = stoptime / nsteps

    S = [S_0/N_sites]
    B = [B_0/N_sites]
    E = [E_0/N_sites]
    T = [0]


    for i in range(nsteps):
        S.append(S[i] + dt * (s*E[i] - B[i]*S[i]))
        E.append(E[i] + dt * (B[i]*S[i] + d*B[i] - s*E[i] - r*B[i]*S[i]*E[i]))
        B.append(B[i] + dt * (r*B[i]*S[i]*E[i] - d*B[i]))
        T.append(T[i] + dt)
    
    return T, S, E, B


import numpy as np
from tqdm import tqdm
import pandas as pd
from single_species_utils import run


def run_raster(n_steps, L, r, d_list, s_list, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
        T, S, E, B = ode_integrate(L, s, d, r, stoptime=n_steps, nsteps=100_000)
        for step in steps_to_record:
            soil_lattice_list.append({"d": d, "s": s, "step":step, "vacancy": E[step], "bacteria": B[step], "soil": S[step]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of bacteria moves
    L = 20  # side length of the square lattice
    r = 1  # reproduction rate
    d = np.linspace(0, 0.3, 10)  # death rate
    s = np.linspace(0, 0.8, 10)  # soil filling rate

    soil_lattice_data = run_raster(n_steps, L, r, d, s)

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/single_species/mean_field_data_{r=}.json", orient="records")
    

if __name__ == "__main__":
    main()
