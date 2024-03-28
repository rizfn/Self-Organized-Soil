import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

def read_large_file(file_object):
    """A generator to read a large file lazily."""
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

def main():

    sigma = 1
    theta = 0.3
    # Load the CSV file
    with open(f'src/cuda_test/simpleNbrGrowth/outputs/lattice2D/sigma_{sigma}_theta_{theta}.csv', 'r') as file:
        file_gen = read_large_file(file)
        header = next(file_gen)  # Skip the header

        for line in tqdm(file_gen):
            # Extract the lattice data
            lattice = np.fromstring(line, sep=',')[1:]

    # Display the final lattice using imshow
    plt.imshow(lattice.reshape((int(np.sqrt(len(lattice))), int(np.sqrt(len(lattice))))), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()


def animate():
    sigma = 1
    theta = 0.5
    # Load the CSV file
    with open(f'src/cuda_test/simpleNbrGrowth/outputs/lattice2D/sigma_{sigma}_theta_{theta}.csv', 'r') as file:
        file_gen = read_large_file(file)
        header = next(file_gen)  # Skip the header
        data = [np.fromstring(line, sep=',') for line in tqdm(file_gen)]
        steps = [int(d[0]) for d in data]
        lattices = [d[1:] for d in data]
    L = int(np.sqrt(len(lattices[0])))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Initial image
    im = ax.imshow(lattices[0].reshape((L, L)), cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax.set_title(f'Step: {steps[0]}')

    def update(i):
        # Clear the current axes.
        ax.clear()
        # Draw the new image.
        im = ax.imshow(lattices[i].reshape((L, L)), cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'Step: {steps[i]}')
        return [im]

    pbar = tqdm(total=len(lattices))
    
    def update_with_progress(i):
        pbar.update()
        return update(i)
        
    ani = FuncAnimation(fig, update_with_progress, frames=len(lattices), blit=True)
    ani.save(f'src/cuda_test/simpleNbrGrowth/plots/lattice2D/sigma_{sigma}_theta_{theta}.gif', writer='ffmpeg', fps=10, dpi=200)

    pbar.close()
    plt.show()

if __name__ == '__main__':
    # main()
    animate()