import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_twospec_utils import ode_integrate_rk4


def run_raster(n_steps, rho1, rho2, theta1, theta2_list, sigma_list, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Scans the ODE integrator over sigma and theta2 for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    theta1 : float
        Death rate of green worms.
    theta2_list : ndarray
        Array of death rates of blue worms.
    sigma_list : ndarray
        Array of soil filling rates.
    delta : float
        Nutrient decay rate.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].
    
    Returns
    -------
    soil_fractions_list : list
        List of soil_fraction data for specific timesteps and parameters.
    """
    grid = np.meshgrid(theta2_list, sigma_list)
    ts_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    soil_fractions_list = []
    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta2, sigma = ts_pairs[i]
        T, S, E, NG, NB, WG, WB = ode_integrate_rk4(sigma, theta1, theta2, rho1, rho2, delta, stoptime=n_steps, nsteps=n_steps)
        for step in steps_to_record:
            soil_fractions_list.append({"theta2":theta2, "sigma": sigma, "step":step, "vacancy": E[step], "nutrient_g": NG[step],
                                         "nutrient_b": NB[step], "worm_g":WG[step], "worm_b":WB[step], "soil": S[step]})
    return soil_fractions_list


def main():

    # initialize the parameters
    n_steps = 100_000  # number of worm moves
    rho1 = 1  # green reproduction rate
    rho2 = 1  # blue reproduction rate
    delta = 0  # nutrient decay rate
    theta1 = 0.1  # death rate of green worms
    theta2_list = np.linspace(0, 0.3, 20)  # death rate of blue worms
    sigma_list = np.linspace(0, 1, 20)  # soil filling rate

    soil_lattice_data = run_raster(n_steps, rho1, rho2, theta1, theta2_list, sigma_list, delta, np.array([n_steps]))

    soil_lattice_data = pd.DataFrame(soil_lattice_data)

    # save the data
    soil_lattice_data.to_json(f"docs/data/nutrient_twospec/meanfield_{rho1=}_{rho2=}_{delta=}_{theta1=}.json", orient="records")
    


def plot_single_run():
    n_steps = 10_000  # number of worm moves
    rho1 = 1
    rho2 = 1
    theta1 = 0.05
    delta = 0  # nutrient decay rate   

    theta2 = 0.1
    sigma = 0.8

    T, S, E, NG, NB, WG, WB = ode_integrate_rk4(sigma, theta1, theta2, rho1, rho2, delta, stoptime=n_steps, nsteps=n_steps)

    import matplotlib.pyplot as plt

    plt.grid()
    plt.plot(T, S, label="soil", c='brown')
    plt.plot(T, E, label="vacancy", c='grey')
    plt.plot(T, NG, label="nutrient green", c='lawngreen')
    plt.plot(T, NB, label="nutrient blue", linestyle='--', c='turquoise')
    plt.plot(T, WG, label="worm green", c='green')
    plt.plot(T, WB, label="worm blue", linestyle='--', c='blue')
    plt.title(f"{theta1=}, {theta2=}, {sigma=}, {rho1=}, {rho2=}, {delta=}")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # main()
    plot_single_run()
