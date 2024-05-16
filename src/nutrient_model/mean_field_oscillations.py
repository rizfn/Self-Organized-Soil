import numpy as np
import pandas as pd
from nutrient_utils import ode_integrate_rk4
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib import colors
import multiprocessing as mp
from functools import partial
import concurrent.futures
from tqdm import tqdm


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

def calculate_oscillations(ts_pair, rho, delta, n_steps):
    theta, sigma = ts_pair
    T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=n_steps, nsteps=n_steps)
    time_period, amplitude, sensitivity =  calculate_oscillation_metrics(W[n_steps//2:], T[n_steps//2:])
    return {"theta": theta, "sigma": sigma, "time_period": time_period, "amplitude": amplitude, "sensitivity": sensitivity}

def run_raster(n_steps, rho, theta_list, sigma_list, delta):
    grid = np.meshgrid(theta_list, sigma_list)
    ts_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s

    num_processes = max(1, mp.cpu_count() - 4)  # Leave 4 cores free
    with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
        oscillation_list = list(tqdm(executor.map(partial(calculate_oscillations, rho=rho, delta=delta, n_steps=n_steps), ts_pairs), total=len(ts_pairs)))

    return oscillation_list


def main():

    n_steps = 10_000  # number of worm moves
    rho = 1  # reproduction rate
    delta = 0  # nutrient decay rate
    theta_list = np.linspace(0, 0.3, 400)  # death rate
    sigma_list = np.linspace(0, 1, 400)  # soil filling rate

    oscillation_data = run_raster(n_steps, rho, theta_list, sigma_list, delta)

    oscillation_data = pd.DataFrame(oscillation_data)

    # Assuming oscillation_data is a pandas DataFrame
    period_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='time_period')
    amplitude_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='amplitude')
    sensitivity_pivot = oscillation_data.pivot(index='sigma', columns='theta', values='sensitivity')

    fig, axs = plt.subplots(1, 3, figsize=(16, 8))

    plt.suptitle("Worm Oscillations: Period and Amplitude")

    cmap1 = colors.ListedColormap(plt.get_cmap('hot')(np.arange(0.8*256).astype(int)))
    cmap2 = colors.ListedColormap(plt.get_cmap('gist_heat')(np.arange(0.9*256).astype(int)))
    cmap3 = colors.ListedColormap(plt.get_cmap('afmhot')(np.arange(0.9*256).astype(int)))

    # Period heatmap
    cax0 = axs[0].imshow(np.log10(period_pivot), aspect='auto', origin='lower', extent=[period_pivot.columns.min(), period_pivot.columns.max(), period_pivot.index.min(), period_pivot.index.max()], cmap=cmap1)
    cbar0 = fig.colorbar(cax0, ax=axs[0], orientation='horizontal', pad=0.1)
    axs[0].set_xlabel(r"$\theta$ (Death rate)")
    axs[0].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[0].set_title(r"$\log_{10}(T)$")

    # Amplitude heatmap
    cax1 = axs[1].imshow(amplitude_pivot, aspect='auto', origin='lower', extent=[amplitude_pivot.columns.min(), amplitude_pivot.columns.max(), amplitude_pivot.index.min(), amplitude_pivot.index.max()], cmap=cmap1)
    cbar1 = fig.colorbar(cax1, ax=axs[1], orientation='horizontal', pad=0.1)
    axs[1].set_xlabel(r"$\theta$ (Death rate)")
    axs[1].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[1].set_title(r"$A$")

    # Sensitivity heatmap
    cax2 = axs[2].imshow(np.log10(sensitivity_pivot), aspect='auto', origin='lower', extent=[sensitivity_pivot.columns.min(), sensitivity_pivot.columns.max(), sensitivity_pivot.index.min(), sensitivity_pivot.index.max()], cmap=cmap1)
    cbar2 = fig.colorbar(cax2, ax=axs[2], orientation='horizontal', pad=0.1)
    axs[2].set_xlabel(r"$\theta$ (Death rate)")
    axs[2].set_ylabel(r"$\sigma$ (Soil filling rate)")
    axs[2].set_title(r"$\log_{10}$((Max - Min) / Min)")

    plt.tight_layout()

    plt.savefig("src/nutrient_model/plots/mean_field_oscillations.png")

    plt.show()


if __name__ == "__main__":
    main()
