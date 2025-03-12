import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import f_oneway
from tqdm import tqdm
import pandas as pd
from numba import njit


@njit
def ode_derivatives(S, E, N, W, sigma, theta, rho, delta):
    """Calculate the derivatives of S, E, N, W.

    This function is not called directly, but rather through `ode_integrate_rk4`
    
    Parameters
    ----------
    S : float
        Soil fraction.
    E : float
        Empty fraction.
    N : float
        Nutrient fraction.
    W : float
        Worm fraction.
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm eeproduction rate.
    delta : float
        Nutrient decay rate.

    Returns
    -------
    dS : float
        Derivative of soil fraction.
    dE : float
        Derivative of empty fraction.
    dN : float
        Derivative of nutrient fraction.
    dW : float
        Derivative of worm fraction.
    """

    dS = sigma*S*(E+N) - W*S
    dE = (1-rho)*W*N + theta*W - sigma*S*E + delta*N
    dN = W*S - W*N - sigma*S*N - delta*N
    dW = rho*W*N - theta*W

    return dS, dE, dN, dW


@njit
def ode_integrate_rk4(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000, S_0=0.3, E_0=0.3, N_0=0.3, W_0=0.1):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method.
    
    Parameters
    ----------
    
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


def calculate_time_periods(data, prominence=0.01):
    peaks, _ = find_peaks(data, prominence=prominence)
    if len(peaks) < 2:
        return []
    time_periods = np.diff(peaks)
    return time_periods


def check_if_oscillating(df, prominence=0.01):
    oscillating = False
    time_periods_list = []
    sub_df = df[df["step"] > df["step"].max() / 2].reset_index()  # only consider the second half of the time series
    for column in ["emptys", "nutrients", "greens", "soil"]:
        time_periods = calculate_time_periods(sub_df[column], prominence)
        if not len(time_periods):
            return False
        time_periods_list.append(time_periods)
    # check if the time periods are all the same with an ANOVA test
    if all(np.all(time_periods == time_periods[0]) for time_periods in time_periods_list):
        return True
    F, p = f_oneway(*time_periods_list)
    if p > 0.05:  # if p-value > 0.05, we cannot reject the null hypothesis that the means are equal
        oscillating = True
    return oscillating


def run_raster(n_steps, rho, theta_list, sigma_list, delta, S0=0.25, E0=0.25, N0=0.25, W0=0.25, tolerance=1e-6):
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
    state_data = []
    # S0, E0, N0, W0 = 0.25, 0.25, 0.25, 0.25
    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta, sigma = ts_pairs[i]
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps, S_0=S0, E_0=E0, N_0=N0, W_0=W0)
        if W[-1] <= tolerance:
            if S[-1] <= tolerance:
                state = "Empty"
            else:
                state = "Soil"
        else:
            df = pd.DataFrame({"emptys": E, "nutrients": N, "greens": W, "soil": S, "step": np.arange(len(E))})
            is_oscillating = check_if_oscillating(df)
            if is_oscillating:
                state = "Oscillating"
            else:
                state = "Stable"
        state_data.append({"sigma": sigma, "theta": theta, "state": state})
    return state_data


def main():

    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 200)  # death rate
    sigma_list = np.linspace(0, 1, 200)  # soil filling rate

    S0, E0, N0, W0 = 0.25, 0.25, 0.25, 0.25

    state_data = run_raster(n_steps, rho, theta_list, sigma_list, delta, S0, E0, N0, W0)

    df = pd.DataFrame(state_data)
    # df.to_csv("src/IUPAB_abstract/outputs/TimeseriesMeanField/raster.csv", index=False)
    df.to_json(f"docs/data/nutrient/meanfield_attractors/S0_{S0}_E0_{E0}_N0_{N0}_W0_{W0}.json", orient="records")


def run_raster_node_start(n_steps, rho, theta_list, sigma_list, delta, tolerance=1e-6):
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
    state_data = []

    def stable_fixed_point(sigma, theta):
        sqrt_argument = sigma**2 * theta**2 - 2 * sigma * theta - 4 * theta + 1
        if sqrt_argument < 0:  # if the stable fixed point does not exist
            return 0.3, 0.3, 0.1
        sqrt_term = np.sqrt(sqrt_argument)
        E = (sqrt_term - sigma * theta - 2 * theta + 1) / (2 * (sigma + 1))
        S = 0.5 * (-sqrt_term - sigma * theta + 1)
        W = sigma * (sqrt_term + sigma * theta + 1) / (2 * (sigma + 1))
        return E, S, W
    

    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta, sigma = ts_pairs[i]
        E0, S0, W0 = stable_fixed_point(sigma, theta)
        E0, S0, W0 = max(0, E0 - 1e-3), max(0, S0 - 1e-3), max(0, W0 - 1e-3)
        N0 = 1 - E0 - S0 - W0
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps, S_0=S0, E_0=E0, N_0=N0, W_0=W0)
        if W[-1] <= tolerance:
            if S[-1] <= tolerance:
                state = "Empty"
            else:
                state = "Soil"
        else:
            df = pd.DataFrame({"emptys": E, "nutrients": N, "greens": W, "soil": S, "step": np.arange(len(E))})
            is_oscillating = check_if_oscillating(df)
            if is_oscillating:
                state = "Oscillating"
            else:
                state = "Stable"
        state_data.append({"sigma": sigma, "theta": theta, "state": state})
    return state_data


def start_near_node():
    
    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 200)  # death rate
    sigma_list = np.linspace(0, 1, 200)  # soil filling rate

    state_data = run_raster_node_start(n_steps, rho, theta_list, sigma_list, delta)

    df = pd.DataFrame(state_data)
    df.to_json(f"docs/data/nutrient/meanfield_attractors/node.json", orient="records")



if __name__ == "__main__":
    # main()
    start_near_node()
