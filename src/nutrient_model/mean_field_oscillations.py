import numpy as np
from tqdm import tqdm
import pandas as pd
from nutrient_utils import ode_integrate_rk4
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


def calculate_oscillation_metrics(array, time_series):
    """Calculates the oscillation period, amplitude, and sensitivity of the oscillating part of the time series.
    
    Parameters
    ----------
    array : ndarray
        Time series of the variable.
    time_series : ndarray
        Time series of the time.
    
    Returns
    -------
    oscillation_period : float
        Mean oscillation period.
    amplitude : float
        Mean oscillation amplitude.
    sensitivity : float
        Mean oscillation sensitivity.
    """
    # find the peaks
    peaks, _ = find_peaks(array, prominence=0.01)
    
    # if no peaks are found, return None
    if len(peaks) == 0:
        return None, None, None

    # find the time at which the peaks occur
    peak_times = time_series[peaks]
    # find the difference between the times
    peak_time_diffs = np.diff(peak_times)
    # calculate the mean difference (oscillation period)
    oscillation_period = np.mean(peak_time_diffs)
    # calculate the mean peak value (amplitude)
    amplitude = np.mean(array[peaks])

    # define the oscillating region as the part of the array between the first and last peak
    oscillating_region = array[peaks[0]:peaks[-1]+1]
    # calculate the oscillation closeness metric for the oscillating region (sensitivity)
    min_value = np.min(oscillating_region)
    max_value = np.max(oscillating_region)
    sensitivity = (max_value - min_value) / min_value

    # return the oscillation period, amplitude, and sensitivity
    return oscillation_period, amplitude, sensitivity


def run_raster(n_steps, rho, theta_list, sigma_list, delta):
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
    oscillation_list = []
    for i in tqdm(range(len(ts_pairs))):  # todo: parallelize
        theta, sigma = ts_pairs[i]
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps)
        time_period, amplitude, sensitivity =  calculate_oscillation_metrics(W[n_steps//2:], T[n_steps//2:])
        oscillation_list.append({"theta": theta, "sigma": sigma, "time_period": time_period, "amplitude": amplitude, "sensitivity": sensitivity})
    return oscillation_list


def main():

    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 100)  # death rate
    sigma_list = np.linspace(0, 1, 100)  # soil filling rate

    oscillation_data = run_raster(n_steps, rho, theta_list, sigma_list, delta)

    oscillation_data = pd.DataFrame(oscillation_data)

    # Assuming oscillation_data is a pandas DataFrame
    period_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='time_period')
    amplitude_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='amplitude')
    sensitivity_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='sensitivity')

    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    plt.suptitle("Worm Oscillations: Period and Amplitude")

    # Period heatmap
    cax0 = axs[0].imshow(np.log10(period_pivot), aspect='auto', origin='lower', extent=[period_pivot.columns.min(), period_pivot.columns.max(), period_pivot.index.min(), period_pivot.index.max()])
    cbar0 = fig.colorbar(cax0, ax=axs[0], orientation='horizontal', pad=0.2)
    axs[0].set_xlabel(r"$\theta$ (Death rate)")
    axs[0].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[0].set_title(r"$\log_{10}(T)$")

    # Amplitude heatmap
    cax1 = axs[1].imshow(amplitude_pivot, aspect='auto', origin='lower', extent=[amplitude_pivot.columns.min(), amplitude_pivot.columns.max(), amplitude_pivot.index.min(), amplitude_pivot.index.max()])
    cbar1 = fig.colorbar(cax1, ax=axs[1], orientation='horizontal', pad=0.2)
    axs[1].set_xlabel(r"$\theta$ (Death rate)")
    axs[1].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[1].set_title(r"$A$")

    # Sensitivity heatmap
    cax2 = axs[2].imshow(np.log10(sensitivity_pivot), aspect='auto', origin='lower', extent=[sensitivity_pivot.columns.min(), sensitivity_pivot.columns.max(), sensitivity_pivot.index.min(), sensitivity_pivot.index.max()])
    cbar2 = fig.colorbar(cax2, ax=axs[2], orientation='horizontal', pad=0.2)
    axs[2].set_xlabel(r"$\theta$ (Death rate)")
    axs[2].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[2].set_title(r"$\log_{10}$((Max - Min) / Min)")

    plt.tight_layout()

    plt.savefig("src/nutrient_model/plots/mean_field_oscillations.png")

    plt.show()


if __name__ == "__main__":
    main()
