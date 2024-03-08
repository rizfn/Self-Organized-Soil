import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import glob

def main():
    sigma = 0.5
    theta = 0.015
    N = 5  # Number of species
    # filepath = f"src/multi_species_nutrient/outputs/lattice2D/{N}spec/GPU_sigma_{sigma}_theta_{theta}.tsv"
    filepath = f"src/multi_species_nutrient/outputs/lattice2D/{N}spec/GPU_noSoil_theta_{theta}.tsv"

    # Define the color scheme
    light_colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']
    dark_colors = ['green', 'blue', 'purple', 'red', 'darkgoldenrod']
    if N > len(light_colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    # colors = ['white'] + light_colors[:N] + ['sienna'] + dark_colors[:N]
    colors = ['white'] + light_colors[:N] + dark_colors[:N]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    fig, ax = plt.subplots(figsize=(9, 9), dpi=200)
    fig.suptitle(f'{N} species, $\sigma$={sigma}, $\\theta$={theta}')
    lattice = np.loadtxt(filepath)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    # ax.invert_yaxis()  # Invert the y-axis
    plt.show()


def viz_all():
    dirpath = "src/multi_species_nutrient/outputs/confinement/algo_comparison/other2"
    light_colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']
    dark_colors = ['green', 'blue', 'purple', 'red', 'darkgoldenrod']
    N = 5  # Number of species
    if N > len(light_colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    colors = ['white'] + light_colors[:N] + ['sienna'] + dark_colors[:N]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    for filepath in glob.glob(f"{dirpath}/*.tsv"):
        fig, ax = plt.subplots(dpi=200)
        fig.suptitle(filepath.split('\\')[-1])
        lattice = np.loadtxt(filepath)
        ax.imshow(lattice, cmap=cmap, norm=norm)
        plt.savefig(f"{filepath}.png")
        # plt.show()
        plt.close()


if __name__ == "__main__":
    main()
    # viz_all()