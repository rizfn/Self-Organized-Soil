import matplotlib.pyplot as plt
import numpy as np


def main():
    # 3D
    L = 50
    sigma = 1
    theta = 0.03

    step, E, N, S, G, B = np.loadtxt(f"src/nutrient_mutations/outputs/timeseries/sigma_{sigma}_theta_{theta}.csv", skiprows=1, delimiter=",", unpack=True)

    fig, axs = plt.subplots(figsize=(10, 6))

    axs.plot(step, S, label="soil", c="brown")
    axs.plot(step, E, label="emptys", c="grey")
    axs.plot(step, N, label="nutrients", c="lawngreen")
    axs.plot(step, G, label="green worms", c="green")
    axs.plot(step, B, label="blue worms", c="blue")
    axs.set_title(f"{L=}, {sigma=}, {theta=}")
    axs.set_xlabel(r"Timestep / L$^3$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    plt.savefig('src/two_species_same_nutrient/plots/lattice_timeseries/parasite_oscillations.png', dpi=300)

    plt.show()



if __name__ == "__main__":
    main()
