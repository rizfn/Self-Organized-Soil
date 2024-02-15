import matplotlib.pyplot as plt
import numpy as np


def main():
    # 3D
    L = 50
    sigma = 0.5
    theta = 0.06
    rho = 1
    mu = 1

    step, E, N_G, N_B, N_P, S, G, B, P = np.loadtxt(f"src/multi_species_nutrient/outputs/timeseries3D/3spec/sigma_{sigma}_theta_{theta}_rho_{rho}_mu_{mu}.csv", skiprows=1, delimiter=",", unpack=True)

    fig, axs = plt.subplots(figsize=(10, 6))

    axs.plot(step, S, label="soil", c="brown")
    axs.plot(step, E, label="emptys", c="grey")
    axs.plot(step, G, label="green worms", c="green")
    axs.plot(step, B, label="blue worms", c="blue")
    axs.plot(step, P, label="purple worms", c="purple")
    axs.set_title(f"{L=}, {sigma=}, {theta=}")
    axs.set_xlabel(r"Timestep / L$^3$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    plt.savefig(f'src/multi_species_nutrient/plots/timeseries3D/3spec/sigma_{sigma}_theta_{theta}.png', dpi=300)

    plt.show()



if __name__ == "__main__":
    main()
