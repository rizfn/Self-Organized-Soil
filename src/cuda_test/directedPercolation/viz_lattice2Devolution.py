import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm


def blocks(file):
    block = []
    for line in file:
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []

def load_lattices(filename):
    lattices = []
    with open(filename, 'r') as file:
        for block in blocks(file):
            lattice = np.genfromtxt(block, delimiter=',')
            lattices.append(lattice)
    return lattices


def main():
    p = 0.2873
    L = 100

    filename = f'src/cuda_test/directedPercolation/outputs/latticeEvolution2D/p_{p}_L_{L}.csv'
    lattices = load_lattices(filename)

    sigma = 1
    theta = 0.605
    filename = f'src/cuda_test/simpleNbrGrowth/outputs/latticeEvolution2D/sigma_{sigma}_theta_{theta}_L_{L}.csv'
    lattices_2 = load_lattices(filename)

    fig, axs = plt.subplots(1, 2, figsize=(14,8))
    plt.tight_layout()

    # Create the initial images
    im1 = axs[0].imshow(lattices[0], cmap='viridis', origin='lower')
    im2 = axs[1].imshow(lattices_2[0], cmap='viridis', origin='lower')

    def update(i):
        # Update the data of the images
        im1.set_array(lattices[i])
        im2.set_array(lattices_2[i])
        fig.suptitle(f'Step: {i}')
        axs[0].title.set_text(f'Directed Percolation, p={p}')
        axs[1].title.set_text(f'Simple Nbr Growth, $\sigma$={sigma}, $\\theta$={theta}')

    pbar = tqdm(total=len(lattices))

    def update_with_progress(i):
        pbar.update()
        return update(i)

    ani = animation.FuncAnimation(fig, update_with_progress, frames=len(lattices))

    ani.save('src/cuda_test/directedPercolation/plots/timeseries/fourActiveInit.mp4', writer='ffmpeg', dpi=300, fps=30)

    pbar.close()

if __name__ == '__main__':
    main()

