import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import ode_derivatives
from numba import njit

@njit
def ode_integrate_rk4_ICs(W_0, E_0, S_0, N_0, sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method.
    
    Parameters
    ----------
    
    W_0 : float
        Initial fraction of worms.
    E_0 : float
        Initial fraction of empty sites.
    S_0 : float
        Initial fraction of soil sites.
    N_0 : float
        Initial fraction of nutrient sites.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm reproduction rate.
    delta : float
        Nutrient decay rate.
    stoptime : int, optional
        Time to stop the integration. The default is 100_000.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    """

    dt = stoptime / nsteps

    S = np.zeros(nsteps+1)
    W = np.zeros(nsteps+1)
    E = np.zeros(nsteps+1)
    N = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    S[0] = S_0
    W[0] = W_0
    E[0] = E_0
    N[0] = N_0
    T[0] = 0

    for i in range(nsteps):
        k1_S, k1_E, k1_N, k1_W = ode_derivatives(S[i], E[i], N[i], W[i], sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k1_S
        E_temp = E[i] + 0.5 * dt * k1_E
        N_temp = N[i] + 0.5 * dt * k1_N
        W_temp = W[i] + 0.5 * dt * k1_W
        k2_S, k2_E, k2_N, k2_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k2_S
        E_temp = E[i] + 0.5 * dt * k2_E
        N_temp = N[i] + 0.5 * dt * k2_N
        W_temp = W[i] + 0.5 * dt * k2_W
        k3_S, k3_E, k3_N, k3_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + dt * k3_S
        E_temp = E[i] + dt * k3_E
        N_temp = N[i] + dt * k3_N
        W_temp = W[i] + dt * k3_W
        k4_S, k4_E, k4_N, k4_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S[i+1] = S[i] + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E[i+1] = E[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        N[i+1] = N[i] + (dt / 6) * (k1_N + 2 * k2_N + 2 * k3_N + k4_N)
        W[i+1] = W[i] + (dt / 6) * (k1_W + 2 * k2_W + 2 * k3_W + k4_W)
        T[i+1] = T[i] + dt

    return T, S, E, N, W



def run_raster_ICs(W_0, E_0, S_0, N_0, n_steps, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
        T, S, E, N, W = ode_integrate_rk4_ICs(W_0, E_0, S_0, N_0, sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "vacancy": E[step], "nutrient": N[step], "worm":W[step], "soil": S[step]})
    return soil_lattice_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 20)  # death rate
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate

    # W_0, E_0, S_0, N_0 = 0.1, 0, 0.9, 0  # 90% soil ICs
    # soil_lattice_data = run_raster_ICs(W_0, E_0, S_0, N_0, n_steps, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100000]))
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/meanfield_90soil_{rho=}_{delta=}.json", orient="records")


    # W_0, E_0, S_0, N_0 = 0.9, 0, 0.1, 0  # 90% worm ICs
    # soil_lattice_data = run_raster_ICs(W_0, E_0, S_0, N_0, n_steps, rho, theta_list, sigma_list, delta)
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/meanfield_90worm_{rho=}_{delta=}.json", orient="records")
    

    # W_0, E_0, S_0, N_0 = 0.25, 0.25, 0.25, 0.25  # even ICs
    # soil_lattice_data = run_raster_ICs(W_0, E_0, S_0, N_0, n_steps, rho, theta_list, sigma_list, delta)
    # soil_lattice_data = pd.DataFrame(soil_lattice_data)
    # soil_lattice_data.to_json(f"docs/data/nutrient/meanfield_even_{rho=}_{delta=}.json", orient="records")



if __name__ == "__main__":
    main()
