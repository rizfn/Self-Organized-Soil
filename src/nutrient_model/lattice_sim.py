import numpy as np
import matplotlib.pyplot as plt
from nutrient_utils import run_stochastic, run_stochastic_3D, ode_integrate_rk4



def main():

    # initialize the parameters
    steps_per_latticepoint = 500  # number of bacteria moves per lattice point
    # L = 100  # side length of the square lattice
    # n_steps = steps_per_latticepoint * L**2  # number of timesteps to run the simulation for
    L = 75  # side length of the cubic lattice
    n_steps = steps_per_latticepoint * L**3  # 3D
    rho = 1  # reproduction rate
    delta = 0
    sigma = 0.37
    theta = 0.06

    # steps_to_record = np.arange(1, n_steps+1, L**2, dtype=np.int32)
    # soil_lattice_data = run_stochastic(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)

    steps_to_record = np.arange(1, n_steps+1, L**3, dtype=np.int32)
    soil_lattice_data = run_stochastic_3D(n_steps, L, rho, theta, sigma, delta, steps_to_record=steps_to_record)

    # emptys = np.sum(soil_lattice_data == 0, axis=(1, 2)) / L**2
    # nutrients = np.sum(soil_lattice_data == 1, axis=(1, 2)) / L**2
    # soil = np.sum(soil_lattice_data == 2, axis=(1, 2)) / L**2
    # worms = np.sum(soil_lattice_data == 3, axis=(1, 2)) / L**2

    def calculate_fractions(matrix):
        flattened = matrix.flatten()
        counts = np.bincount(flattened, minlength=4)
        fractions = counts / counts.sum()
        return fractions

    # Assuming soil_lattice_data is a list of 3D numpy arrays
    fractions = np.array([calculate_fractions(matrix) for matrix in soil_lattice_data])

    emptys = fractions[:, 0]
    nutrients = fractions[:, 1]
    soil = fractions[:, 2]
    worms = fractions[:, 3]


    plt.rcParams['font.family'] = 'monospace'
    # fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,6))
    fig, axs = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(8,5))

    steps_to_record = steps_to_record / L**3
    # First subplot
    axs[0].plot(steps_to_record, soil, label="soil", c="brown")
    axs[0].plot(steps_to_record, emptys, label="empty", c="grey")
    axs[0].plot(steps_to_record, nutrients, label="nutrients", c="turquoise")
    axs[0].plot(steps_to_record, worms, label="worms", c="green")
    axs[0].set_title(f"3D Lattice, {L=}")
    axs[0].set_xlabel("Timestep")
    axs[0].set_ylabel("Fraction")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend(loc="upper right")
    
    # Second subplot
    T, S, E, N, W = ode_integrate_rk4(sigma, theta, rho, delta, stoptime=steps_per_latticepoint, nsteps=steps_per_latticepoint)
    
    axs[1].plot(T, S, label="soil", c="brown")
    axs[1].plot(T, E, label="empty", c="grey")
    axs[1].plot(T, N, label="nutrient", c="turquoise")
    axs[1].plot(T, W, label="worm", c="green")
    axs[1].set_title("Mean-Field")
    axs[1].set_xlabel("Timestep")
    axs[1].set_ylabel("Fraction")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(f'src/nutrient_model/plots/time_series/3D_vs_MF/s_{sigma}_t_{theta}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
