import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import ode_integrate_rk4


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

    T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=100_000)

    import matplotlib.pyplot as plt

    plt.plot(T, S, label="soil")
    plt.plot(T, E, label="vacancy")
    plt.plot(T, N, label="nutrient")
    plt.plot(T, W, label="worm")
    plt.title(f"{theta=}, {sigma=}, {rho=}, {delta=}")
    # plt.plot(T, np.array(S) + np.array(N) + np.array(W) + np.array(E), label="total")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # main()
    plot_single_run()
