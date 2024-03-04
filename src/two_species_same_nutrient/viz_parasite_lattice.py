import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def main():
    sigma = 1
    theta = 0.042
    rho1 = 0.25
    rho2 = 1
    mu = 1
    filepath = f"src/two_species_same_nutrient/outputs/confinement/sigma_{sigma}_theta_{theta}_rhofactor_{rho2/rho1:.0f}.tsv"

    colors = ['white', 'turquoise','sienna', 'green', 'blue']

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)
    fig.suptitle(f'Parasite model, $\sigma$={sigma}, $\\theta$={theta}, $\\rho_1$={rho1}, $\\rho_2$={rho2}, $\mu$={mu}')
    lattice = np.loadtxt(filepath)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    plt.savefig(f'src/two_species_same_nutrient/plots/confinement/rhofactor_{rho2/rho1:.0f}/sigma_{sigma}_theta_{theta}.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
