import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import ode_integrate_rk4, ode_derivatives
from numba import njit


def run_raster(n_steps, rho, theta_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
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
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=100_000)
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

    soil_lattice_data = run_raster(n_steps, rho, theta_list, sigma_list, delta)

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/nutrient/meanfield_{rho=}_{delta=}.json", orient="records")
    

def ode_integrate_fast(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000):
    """Integrate the ODEs for the single species model.

    Parameters
    ----------
    sigma : float
        Soil filling rate.
    theta : float
        Worm death rate.
    rho : float
        Worm eproduction rate.
    delta : float
        Nutrient decay rate.
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
    N : list
        List of nutrient fractions.
    W : list
        List of worm fractions.
    """

    W_0 = 0.1  # initial fraction of worms
    E_0 = (1 - W_0) / 3  # initial number of empty sites
    S_0 = (1 - W_0) / 3 # initial number of soil sites
    N_0 = 1 - W_0 - E_0 - S_0  # initial number of nutrient sites

    dt = stoptime / nsteps

    S = [S_0]
    W = [W_0]
    E = [E_0]
    N = [N_0]
    T = [0]

    f = 0.1

    for i in range(nsteps):
        S.append(S[i] + dt * (sigma*S[i]*(f*E[i]+N[i]) - W[i]*S[i]))
        E.append(E[i] + f * dt * ((1-rho)*W[i]*N[i] + theta*W[i] - sigma*S[i]*E[i] + delta*N[i]))
        N.append(N[i] + dt * (W[i]*S[i] - f*W[i]*N[i] - sigma*S[i]*N[i] - f *delta*N[i]))
        W.append(W[i] + f * dt * (rho*W[i]*N[i] - theta*W[i]))
        T.append(T[i] + dt)
    
    return T, S, E, N, W


def plot_single_run():
    n_steps = 100_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    # theta = 0.08
    # sigma = 1
    
    theta = 0.1
    sigma = 0.39
    # T, S, E, N, W = ode_integrate_fast(sigma, theta, rho, delta, stoptime=n_steps, nsteps=100_000)

    T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps)

    import matplotlib.pyplot as plt

    plt.plot(T, S, label="soil")
    plt.plot(T, E, label="vacancy")
    plt.plot(T, N, label="nutrient")
    plt.plot(T, W, label="worm")
    plt.title(f"{theta=}, {sigma=}, {rho=}, {delta=}")
    # plt.plot(T, np.array(S) + np.array(N) + np.array(W) + np.array(E), label="total")
    plt.legend()
    plt.show()


@njit
def ode_integrate_rk4_wormcap(sigma, theta, rho, delta, wormcap=1e-6, stoptime=100_000, nsteps=100_000):
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
    wormcap : float, optional
        Minimum allowed worm density. The default is 1e-6.
    stoptime : int, optional
        Time to stop the integration. The default is 100_000.
    nsteps : int, optional
        Number of steps to take. The default is 100_000.
    """

    W_0 = 0.1  # initial fraction of worms
    E_0 = (1 - W_0) / 3  # initial number of empty sites
    S_0 = (1 - W_0) / 3 # initial number of soil sites
    N_0 = 1 - W_0 - E_0 - S_0  # initial number of nutrient sites

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

        if W[i+1] < wormcap:
            artificially_added_worms = wormcap - W[i+1]
            W[i+1] = wormcap
            ENS_sum = E[i+1] + N[i+1] + S[i+1]
            E[i+1] -= artificially_added_worms * E[i+1] / ENS_sum
            N[i+1] -= artificially_added_worms * N[i+1] / ENS_sum
            S[i+1] -= artificially_added_worms * S[i+1] / ENS_sum

    return T, S, E, N, W



def run_raster_wormcap(n_steps, rho, theta_list, sigma_list, delta, wormcap=1e-6, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Scans the ODE integrator over s and d for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho : float
        Reproduction rate.
    d_list : ndarray
        List of death rates.
    s_list : ndarray
        List of soil filling rates.
    delta : float
        Nutrient decay rate.
    wormcap : float
        Minimum allowed worm density.
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
        T, S, E, N, W = ode_integrate_rk4_wormcap(sigma, theta, rho, delta, wormcap, stoptime=n_steps, nsteps=100_000)
        for step in steps_to_record:
            soil_lattice_list.append({"theta": theta, "sigma": sigma, "step":step, "vacancy": E[step], "nutrient": N[step], "worm":W[step], "soil": S[step]})
    return soil_lattice_list


def main_wormcap():

    # initialize the parameters
    n_steps = 100_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 20)  # death rate
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate
    wormcap = 1e-1

    soil_lattice_data = run_raster_wormcap(n_steps, rho, theta_list, sigma_list, delta, wormcap, steps_to_record=np.array([100000]))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # plot the data as a heatmap
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    pivoted_data = soil_lattice_data.pivot(index="sigma", columns="theta", values="worm")
    sns.heatmap(np.log10(pivoted_data), ax=ax, cmap="viridis")    
    ax.invert_yaxis()
    plt.suptitle("log10 worm density")
    ax.set_title(f"{rho=}, {delta=}, {wormcap=}")
    xlabels = [round(float(label.get_text()), 2) for label in ax.get_xticklabels()]
    ax.set_xticklabels(xlabels)
    ylabels = [round(float(label.get_text()), 2) for label in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)

    plt.savefig(f"src/nutrient_model/plots/meanfield_wormcap/{wormcap=}.png", dpi=300)
    plt.show()
    




if __name__ == "__main__":
    # main()
    # plot_single_run()
    main_wormcap()
