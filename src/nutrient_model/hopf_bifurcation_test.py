import numpy as np
from tqdm import tqdm
from nutrient_utils import ode_integrate_rk4
import matplotlib.pyplot as plt


def plot_multiple_runs():
    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0.16, 0.21, 4)
    # theta_list = np.linspace(0.110, 0.135, 5)
    sigma = 0.37
    E_list = []
    for theta in tqdm(theta_list):
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=100_000)
        E_list.append(E)

    for i, E in enumerate(E_list):
        plt.plot(T, E, label=f"theta = {theta_list[i]:.4f}")
    plt.title(f"sigma = {sigma}")
    plt.legend()
    plt.show()



def main():

    plot_multiple_runs()
    



if __name__ == "__main__":
    main()
