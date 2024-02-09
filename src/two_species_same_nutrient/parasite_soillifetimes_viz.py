import matplotlib.pyplot as plt
import numpy as np


def main():
    # 3D
    L = 50
    sigma = 1
    theta = 0.02
    n_steps = L**3 * 2000  # from sim

    step, soil_lifetimes = np.loadtxt(f"src/two_species_same_nutrient/outputs/soil_lifetimes/sigma_{sigma}_theta_{theta}.csv", skiprows=1, delimiter=",", unpack=True)
    # soil_lifetimes = soil_lifetimes[step > n_steps / 4]
    soil_lifetimes = soil_lifetimes / L**3

    plt.suptitle(f"Time distribution for soil\n{L=}, {theta=}, {sigma=}")

    plt.hist(soil_lifetimes, bins=100, alpha=0.8, linewidth=0.5, edgecolor="black")
    plt.xlabel('Soil Lifetime (steps / L^3)')
    plt.ylabel('Frequency')
    plt.yscale("log")
    plt.show()



if __name__ == "__main__":
    main()
