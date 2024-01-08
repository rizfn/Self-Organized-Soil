import numpy as np
import matplotlib.pyplot as plt
from twospec_samenutrient_utils import update_3D, init_lattice_3D
from scipy.signal import find_peaks
from numba import njit
from tqdm import tqdm



@njit
def run_timeseries_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record=np.array([100, 1000, 10000, 100000])):
    """Run the stochastic simulation for n_steps timesteps.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    L : int
        Side length of the square lattice.
    sigma : float
        Soil filling rate.
    theta : float
        Death rate.
    rho1 : float
        Reproduction rate of green worms.
    rho2 : float
        Reproduction rate of blue worms.
    mu1 : float
        Nutrient creation rate of green worms.
    mu2 : float
        Nutrient creation rate of blue worms.
    steps_to_record : ndarray, optional
        Array of timesteps to record the lattice data for, by default [100, 1000, 10000, 100000].

    Returns
    -------
    emptys : ndarray
        Fraction of empty lattice points at each timestep.
    nutrients : ndarray
        Fraction of nutrient lattice points at each timestep.
    soil : ndarray
        Fraction of soil lattice points at each timestep.
    greens : ndarray
        Fraction of green worm lattice points at each timestep.
    blues : ndarray
        Fraction of blue worm lattice points at each timestep.
    """
    soil_lattice = init_lattice_3D(L)

    emptys = np.zeros(len(steps_to_record), dtype=np.int32)
    nutrients = np.zeros_like(emptys)
    soil = np.zeros_like(emptys)
    greens = np.zeros_like(emptys)
    blues = np.zeros_like(emptys)
    i = 0  # indexing for recording steps

    for step in range(1, n_steps+1):
        update_3D(soil_lattice, L, sigma, theta, rho1, rho2, mu1, mu2)
        if step in steps_to_record:
            flattened = soil_lattice.flatten()
            counts = np.bincount(flattened, minlength=5)
            emptys[i] = counts[0]
            nutrients[i] = counts[1]
            soil[i] = counts[2]
            greens[i] = counts[3]
            blues[i] = counts[4]
            i += 1

    emptys = emptys / L**3
    nutrients = nutrients / L**3
    soil = soil / L**3
    greens = greens / L**3
    blues = blues / L**3

    return emptys, nutrients, soil, greens, blues


def main():

    # initialize the parameters
    steps_per_latticepoint = 1000  # number of bacteria moves per lattice point
    L = 25  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    sigma = 0.2
    theta = 0.05
    rho1 = 1
    rho2 = 0.9
    mu1 = 0.5
    mu2 = 1

    steps_to_record = np.arange(0, n_steps+1, L**3, dtype=np.int32)

    emptys, nutrients, soil, greens, blues = run_timeseries_3D(n_steps, L, sigma, theta, rho1, rho2, mu1, mu2, steps_to_record=steps_to_record)
    
    fig, axs = plt.subplots()

    steps_to_record = steps_to_record / L**3

    axs.plot(steps_to_record, soil, label="soil", c="brown")
    axs.plot(steps_to_record, emptys, label="emptys", c="grey")
    axs.plot(steps_to_record, nutrients, label="nutrients", c="lawngreen")
    axs.plot(steps_to_record, greens, label="green worms", c="green")
    axs.plot(steps_to_record, blues, label="blue worms", c="blue")
    axs.set_title(f"{L=}, {sigma=}, {theta=}, {rho1=}, {rho2=}, {mu1=}, {mu2=}")
    axs.set_xlabel("Timestep / L^3")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    plt.show()


if __name__ == "__main__":
    main()
