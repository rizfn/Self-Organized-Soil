import matplotlib.pyplot as plt
import pandas as pd

def main():
    L = 500
    sigma = 0.5
    theta = 0.023
    rho = 1
    mu = 1
    N = 3  # Number of species

    data = pd.read_csv(f"src/multi_species_nutrient/outputs/timeseries2D/Nspec/{N}spec_sigma_{sigma}_theta_{theta}.csv")

    fig, axs = plt.subplots(figsize=(10, 6))

    axs.plot(data['step'], data['soil'], label="soil", c="brown")
    axs.plot(data['step'], data['emptys'], label="emptys", c="grey")

    colors = ['green', 'blue', 'purple', 'red', 'orange']  # Add more colors if N > 5

    for i in range(1, N + 1):
        axs.plot(data['step'], data[f'worm{i}'], label=f"worm{i}", c=colors[i % len(colors)])

    axs.set_title(f"{L=}, {sigma=}, {theta=}")
    axs.set_xlabel(r"Timestep / L$^2$")
    axs.set_ylabel("Fraction of lattice points")
    axs.legend()
    axs.grid()

    # plt.savefig(f'src/multi_species_nutrient/plots/timeseries2D/Nspec/{N}spec_sigma_{sigma}_theta_{theta}.png', dpi=300)

    plt.show()


if __name__ == "__main__":
    main()