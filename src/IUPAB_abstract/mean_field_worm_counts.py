import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import f_oneway
from tqdm import tqdm
import pandas as pd
from numba import njit

@njit
def ode_derivatives(S, E, N, W, sigma, theta, rho, delta):
    """Calculate the derivatives of S, E, N, W."""
    dS = sigma*S*(E+N) - W*S
    dE = (1-rho)*W*N + theta*W - sigma*S*E + delta*N
    dN = W*S - W*N - sigma*S*N - delta*N
    dW = rho*W*N - theta*W
    return dS, dE, dN, dW

@njit
def ode_integrate_rk4(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000, S_0=0.3, E_0=0.3, N_0=0.3, W_0=0.1):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method."""
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

def run_raster(n_steps, rho, theta_list, sigma_list, delta, equilibrium_step_fraction, S0=0.25, E0=0.25, N0=0.25, W0=0.25):
    """Scans the ODE integrator over sigma and theta for n_steps timesteps."""
    grid = np.meshgrid(theta_list, sigma_list)
    ts_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of theta and sigma

    mean_worm_concentrations = []
    mean_nutrient_concentrations = []

    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta, sigma = ts_pairs[i]
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps, S_0=S0, E_0=E0, N_0=N0, W_0=W0)
        
        final_fraction = int(len(T) * (1 - equilibrium_step_fraction))
        mean_worm = np.mean(W[final_fraction:])  # Calculate the mean worm concentration
        mean_nutrient = np.mean(N[final_fraction:])  # Calculate the mean nutrient concentration

        mean_worm_concentrations.append((sigma, theta, mean_worm))
        mean_nutrient_concentrations.append((sigma, theta, mean_nutrient))

    return mean_worm_concentrations, mean_nutrient_concentrations

def main():
    n_steps = 100_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.2, 200)  # death rate
    sigma_list = np.array([0.1, 0.5, 1.0])  # soil filling rate
    equilibrium_step_fraction = 3/4  # Fraction of the final steps to consider for the equilibrium

    S0, E0, N0, W0 = 0.25, 0.25, 0.25, 0.25

    mean_worm_concentrations, mean_nutrient_concentrations = run_raster(
        n_steps, rho, theta_list, sigma_list, delta, equilibrium_step_fraction, S0, E0, N0, W0)

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Reds([0.5, 0.7, 1]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    for sigma in sigma_list:
        worm_data = [(theta, mean_worm) for s, theta, mean_worm in mean_worm_concentrations if s == sigma]
        nutrient_data = [(theta, mean_nutrient) for s, theta, mean_nutrient in mean_nutrient_concentrations if s == sigma]

        worm_data.sort()
        nutrient_data.sort()

        theta_values_worm, mean_worm_values = zip(*worm_data)
        theta_values_nutrient, mean_nutrient_values = zip(*nutrient_data)

        ax1.plot(theta_values_worm, mean_worm_values, label=f'sigma = {sigma}')
        ax2.plot(theta_values_nutrient, mean_nutrient_values, label=f'sigma = {sigma}')

    ax1.set_title('Worm vs Theta')
    ax1.set_xlabel("Theta")
    ax1.set_ylabel("Mean Worm Concentration")
    ax1.grid()
    ax1.legend()

    ax2.set_title('Nutrient vs Theta')
    ax2.set_xlabel("Theta")
    ax2.set_ylabel("Mean Nutrient Concentration")
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig('src/IUPAB_abstract/plots/worm_counts/3sigmas_meanfield.png')
    plt.show()

if __name__ == "__main__":
    main()