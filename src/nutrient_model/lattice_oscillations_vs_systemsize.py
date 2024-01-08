import numpy as np
import matplotlib.pyplot as plt
from nutrient_utils import update_stochastic_3D, ode_integrate_rk4, init_lattice_3D
from scipy.signal import find_peaks
from numba import njit
from tqdm import tqdm



@njit
def run_timeseries_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    rho : float
        Reproduction rate.
    theta : float
        Death rate.
    sigma : float
        Soil filling rate.
    delta : float
        Nutrient decay rate.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    soil_lattice_data : ndarray
        List of soil_lattice data for specific timesteps.
    """
    soil_lattice = init_lattice_3D(L)

    emptys = np.zeros(len(steps_to_record), dtype=np.int32)
    nutrients = np.zeros_like(emptys)
    soil = np.zeros_like(emptys)
    worms = np.zeros_like(emptys)
    i = 0  # indexing for recording steps

    for step in range(1, n_steps+1):
        update_stochastic_3D(soil_lattice, L, rho, theta, sigma, delta)
        if step in steps_to_record:
            flattened = soil_lattice.flatten()
            counts = np.bincount(flattened, minlength=4)
            emptys[i] = counts[0]
            nutrients[i] = counts[1]
            soil[i] = counts[2]
            worms[i] = counts[3]
            i += 1

    emptys = emptys / L**3
    nutrients = nutrients / L**3
    soil = soil / L**3
    worms = worms / L**3

    return emptys, nutrients, soil, worms


def calculate_oscillation_parameters(data, prominence=0.01):
    peaks, _ = find_peaks(data, prominence=prominence)
    time_periods = np.diff(peaks)
    avg_time_period = np.mean(time_periods)
    time_period_error = np.std(time_periods) / np.sqrt(len(time_periods))
    
    troughs, _ = find_peaks(-data, prominence=prominence)
    min_length = min(len(peaks), len(troughs))  # to deal with different number of peaks and troughs
    peaks = peaks[:min_length]
    troughs = troughs[:min_length]
    
    amplitudes = data[peaks] - data[troughs]
    avg_amplitude = np.mean(amplitudes) / 2
    amplitude_error = np.std(amplitudes) / (2 * np.sqrt(len(amplitudes)))

    return avg_time_period, time_period_error, avg_amplitude, amplitude_error


# todo: maybe add 2D in too to compare?

def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    rho = 1  # reproduction rate
    delta = 0
    theta = 0.16
    sigma = 0.47
    L_list = np.arange(10, 75 + 1, 5)  # side lengths of the cubic lattice

    T_E, T_err_E = np.zeros(len(L_list)), np.zeros(len(L_list))
    A_E, A_err_E = np.zeros_like(T_E), np.zeros_like(T_E)
    T_N, T_err_N = np.zeros_like(T_E), np.zeros_like(T_E)
    A_N, A_err_N = np.zeros_like(T_E), np.zeros_like(T_E)
    T_S, T_err_S = np.zeros_like(T_E), np.zeros_like(T_E)
    A_S, A_err_S = np.zeros_like(T_E), np.zeros_like(T_E)
    T_W, T_err_W = np.zeros_like(T_E), np.zeros_like(T_E)
    A_W, A_err_W = np.zeros_like(T_E), np.zeros_like(T_E)


    for i, L in tqdm(enumerate(L_list), total=len(L_list)):
        n_steps = steps_per_latticepoint * L**3  # 3D
        steps_to_record = np.arange(n_steps//4, n_steps+1, L**3, dtype=np.int32)

        emptys, nutrients, soil, worms = run_timeseries_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)

        emptys_period, emptys_period_error, emptys_amplitude, emptys_amplitude_error = calculate_oscillation_parameters(emptys)
        nutrients_period, nutrients_period_error, nutrients_amplitude, nutrients_amplitude_error = calculate_oscillation_parameters(nutrients)
        soil_period, soil_period_error, soil_amplitude, soil_amplitude_error = calculate_oscillation_parameters(soil)
        worms_period, worms_period_error, worms_amplitude, worms_amplitude_error = calculate_oscillation_parameters(worms)

        T_E[i], T_err_E[i] = emptys_period, emptys_period_error
        A_E[i], A_err_E[i] = emptys_amplitude, emptys_amplitude_error
        T_N[i], T_err_N[i] = nutrients_period, nutrients_period_error
        A_N[i], A_err_N[i] = nutrients_amplitude, nutrients_amplitude_error
        T_S[i], T_err_S[i] = soil_period, soil_period_error
        A_S[i], A_err_S[i] = soil_amplitude, soil_amplitude_error
        T_W[i], T_err_W[i] = worms_period, worms_period_error
        A_W[i], A_err_W[i] = worms_amplitude, worms_amplitude_error


    fig, axs = plt.subplots(2, figsize=(8, 8))

    plt.suptitle(f"{rho=}, {theta=}, {sigma=}, {delta=}")

    axs[0].errorbar(L_list, T_E, yerr=T_err_E, capsize=5, linestyle='--', marker='x', label="emptys")
    axs[0].errorbar(L_list, T_N, yerr=T_err_N, capsize=5, linestyle='--', marker='x', label="nutrients")
    axs[0].errorbar(L_list, T_S, yerr=T_err_S, capsize=5, linestyle='--', marker='x', label="soil")
    axs[0].errorbar(L_list, T_W, yerr=T_err_W, capsize=5, linestyle='--', marker='x', label="worms")
    axs[0].set_xlabel("L")
    axs[0].set_ylabel("Period")
    axs[0].legend()
    axs[0].grid()

    axs[1].errorbar(L_list, A_E, yerr=A_err_E, capsize=5, linestyle='--', marker='x', label="emptys")
    axs[1].errorbar(L_list, A_N, yerr=A_err_N, capsize=5, linestyle='--', marker='x', label="nutrients")
    axs[1].errorbar(L_list, A_S, yerr=A_err_S, capsize=5, linestyle='--', marker='x', label="soil")
    axs[1].errorbar(L_list, A_W, yerr=A_err_W, capsize=5, linestyle='--', marker='x', label="worms")
    axs[1].set_xlabel("L")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(f"src/nutrient_model/plots/oscillations_vs_systemsize/{theta=}_{sigma=}.png", dpi=300)

    plt.show()



def plot_single_run():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    L = 20  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    rho = 1  # reproduction rate
    delta = 0
    theta = 0.09
    sigma = 0.16
    # sigma = 0.37
    # theta = 0.16
    # sigma = 0.47

    steps_to_record = np.arange(n_steps//4, n_steps+1, L**3, dtype=np.int32)

    emptys, nutrients, soil, worms = run_timeseries_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)

    emptys_period, emptys_period_error, emptys_amplitude, emptys_amplitude_error = calculate_oscillation_parameters(emptys)
    nutrients_period, nutrients_period_error, nutrients_amplitude, nutrients_amplitude_error = calculate_oscillation_parameters(nutrients)
    soil_period, soil_period_error, soil_amplitude, soil_amplitude_error = calculate_oscillation_parameters(soil)
    worms_period, worms_period_error, worms_amplitude, worms_amplitude_error = calculate_oscillation_parameters(worms)

    print(f"emptys_period = {emptys_period:.2f} ± {emptys_period_error:.2f},\t\temptys_amplitude = {emptys_amplitude:.2f} ± {emptys_amplitude_error:.2f}")
    print(f"nutrients_period = {nutrients_period:.2f} ± {nutrients_period_error:.2f},\tnutrients_amplitude = {nutrients_amplitude:.2f} ± {nutrients_amplitude_error:.2f}")
    print(f"soil_period = {soil_period:.2f} ± {soil_period_error:.2f},\t\tsoil_amplitude = {soil_amplitude:.2f} ± {soil_amplitude_error:.2f}")
    print(f"worms_period = {worms_period:.2f} ± {worms_period_error:.2f},\t\tworms_amplitude = {worms_amplitude:.2f} ± {worms_amplitude_error:.2f}")
    
    fig, axs = plt.subplots()

    steps_to_record = steps_to_record / L**3

    axs.plot(steps_to_record, soil, label="soil")
    axs.plot(steps_to_record, emptys, label="emptys")
    axs.plot(steps_to_record, nutrients, label="nutrients")
    axs.plot(steps_to_record, worms, label="worms")
    axs.set_title(f"{L=}, {rho=}, {theta=}, {sigma=}, {delta=}")
    axs.set_xlabel("Timestep / L^3")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()

    plt.show()


if __name__ == "__main__":
    main()
    # plot_single_run()
