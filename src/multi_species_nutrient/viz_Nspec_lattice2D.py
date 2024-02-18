import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import ast
from tqdm import tqdm

def parse_lattice(s):
    return ast.literal_eval(s)

def update(frame, lines, img, ax):
    _, lattice_str = lines[frame].split('\t')
    lattice = parse_lattice(lattice_str)
    img.set_array(lattice)
    ax.set_title(f'Step: {frame}')
    return img,

def main():
    sigma = 0.5
    theta = 0.012
    N = 5  # Number of species
    filepath = f"src/multi_species_nutrient/outputs/lattice2D/{N}spec/sigma_{sigma}_theta_{theta}.tsv"

    # Define the color scheme
    if N == 4:
        colors = ['white', 'lightgreen', 'lightblue', 'violet', 'tomato', 'sienna', 'green', 'blue', 'purple', 'red']
    elif N == 5:
        colors = ['white', 'lightgreen', 'lightblue', 'violet', 'tomato', 'wheat', 'sienna', 'green', 'blue', 'purple', 'red', 'darkgoldenrod']
    else:
        raise ValueError(f'N={N} is not supported: add more colours!!')
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(colors)-1)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
    fig.suptitle(f'{N} species, $\sigma$={sigma}, $\\theta$={theta}')
    with open(filepath, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
    total_frames = len(lines)
    _, lattice_str = lines[0].split('\t')
    lattice = parse_lattice(lattice_str)
    img = ax.imshow(lattice, cmap=cmap, norm=norm)
    ax.invert_yaxis()  # Invert the y-axis
    pbar = tqdm(total=total_frames)
    def update_with_progress(frame, lines, img, ax):
        pbar.update()
        return update(frame, lines, img, ax)
    ani = animation.FuncAnimation(fig, update_with_progress, frames=range(total_frames), fargs=(lines, img, ax), blit=True)
    ani.save(f'src/multi_species_nutrient/plots/lattice2D/{N}spec_sigma_{sigma}_theta_{theta}.gif', writer='pillow', fps=30, dpi=200)
    pbar.close()

if __name__ == "__main__":
    main()