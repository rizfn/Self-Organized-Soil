import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import ast
from tqdm import tqdm
import numpy as np

def parse_lattice(s):
    return ast.literal_eval(s)

def update(frame, lines, img, ax):
    step, lattice_str = lines[frame].split('\t')
    lattice = parse_lattice(lattice_str)
    img.set_array(lattice)
    ax.set_title(f'Step: {step}')
    return img,

def main():
    theta = 0.05
    N = 5  # Number of species
    filepath = f"src/multi_species_nutrient/outputs/lattice2D/{N}spec/noSoil_theta_{theta}.tsv"

    # Define the color scheme
    light_colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']
    dark_colors = ['green', 'blue', 'purple', 'red', 'darkgoldenrod']
    if N > len(light_colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    colors = ['white'] + light_colors[:N] + dark_colors[:N]

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.suptitle(f'{N} species, No soil, $\\theta$={theta}')
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    total_frames = len(lines)
    step, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.set_title(f'Step: {step}')
    ax.invert_yaxis()  # Invert the y-axis
    # plt.show(); return
    pbar = tqdm(total=total_frames)
    def update_with_progress(frame, lines, img, ax):
        pbar.update()
        return update(frame, lines, img, ax)
    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), fargs=(lines, img, ax), blit=True)
    # ffmpegwriter = animation.FFMpegWriter(fps=30, bitrate=-1)
    # ani.save(f'src/multi_species_nutrient/plots/lattice2D/{N}spec/noSoil_theta_{theta}.mp4', writer=ffmpegwriter)
    ani.save(f'src/multi_species_nutrient/plots/lattice2D/{N}spec/noSoil_theta_{theta}.gif', writer='ffmpeg', fps=30)
    pbar.close()


def main_nosoil_nonutrient():
    N = 5  # Number of species
    filepath = f"src/multi_species_nutrient/outputs/lattice2D/{N}spec/noSoilnoNutrient.tsv"

    # colors = ['mediumseagreen', 'cornflowerblue', 'mediumorchid', 'salmon', 'goldenrod']
    # cmap = plt.get_cmap('Pastel1')
    # colors = [cmap(i) for i in range(cmap.N)]
    colors = ['lightgreen', 'lightblue', 'violet', 'tomato', 'wheat']

    if N > len(colors):
        raise ValueError(f'N={N} is not supported: add more colours!!')
    
    cmap = mcolors.ListedColormap(colors[:N])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.suptitle(f'{N} species, No soil, No nutrient')
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    total_frames = len(lines)
    step, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)
    img = ax.imshow(lattice, cmap=cmap)
    ax.set_title(f'Step: {step}')
    ax.invert_yaxis()  # Invert the y-axis
    # plt.show(); return
    pbar = tqdm(total=total_frames)
    def update_with_progress(frame, lines, img, ax):
        pbar.update()
        return update(frame, lines, img, ax)
    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), fargs=(lines, img, ax), blit=True)
    # ffmpegwriter = animation.FFMpegWriter(fps=30, bitrate=-1)
    ani.save(f'src/multi_species_nutrient/plots/lattice2D/{N}spec/noSoilNoNutrient.gif', writer='ffmpeg', fps=20)
    pbar.close()


if __name__ == "__main__":
    # main()
    main_nosoil_nonutrient()