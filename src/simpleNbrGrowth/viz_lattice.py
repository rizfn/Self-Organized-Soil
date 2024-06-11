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
    theta = 0.38
    # Load the CSV file
    with open(f'src/simpleNbrGrowth/outputs/lattice2D/sigma_{sigma}_theta_{theta}.csv', 'r') as file:
        file_gen = read_large_file(file)
        header = next(file_gen)  # Skip the header

        for line in tqdm(file_gen):
            # Extract the lattice data
            lattice = np.fromstring(line, sep=',')[1:]

    print(lattice.sum()/len(lattice))

    # Display the final lattice using imshow
    plt.imshow(lattice.reshape((int(np.sqrt(len(lattice))), int(np.sqrt(len(lattice))))), cmap='viridis', origin='lower')
    plt.colorbar()
    plt.show()


def animate():
    sigma = 1
    theta = 1
    # Load the CSV file
    with open(f'src/simpleNbrGrowth/outputs/lattice2DNbrDeath/sigma_{sigma}_theta_{theta}.csv', 'r') as file:
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
    ani.save(f'src/simpleNbrGrowth/plots/lattice2DNbrDeath/sigma_{sigma}_theta_{theta}.mp4', writer='ffmpeg', fps=30, dpi=200)

    pbar.close()
    plt.show()

import cc3d
import matplotlib.colors as mcolors
import numpy as np



def color_clusters(filename):
    with open(filename, 'r') as file:
        file_gen = read_large_file(file)
        header = next(file_gen)  # Skip the header

        for line in tqdm(file_gen):
            # Extract the lattice data
            lattice = np.fromstring(line, sep=',')[1:]

    print(lattice.sum()/len(lattice))
    lattice = lattice.reshape((int(np.sqrt(len(lattice))), int(np.sqrt(len(lattice)))))

    labels_out = cc3d.connected_components(lattice, connectivity=4, periodic_boundary=True)

    # # Count the occurrences of each label to get the cluster sizes
    # cluster_sizes = np.bincount(labels_out.ravel())
    # cluster_sizes = cluster_sizes[1:]
    # counts, bin_edges = np.histogram(cluster_sizes, bins=np.logspace(np.log10(1),np.log10(max(cluster_sizes)), 50))
    # bin_widths = np.diff(bin_edges)
    # normalized_counts = counts / bin_widths
    # plt.bar(bin_edges[:-1], normalized_counts, width=bin_widths, align='edge', log=True)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.title('Cluster size distribution')
    # plt.xlabel('Cluster size')
    # plt.ylabel('Frequency')
    # plt.show()

    cmap = plt.get_cmap('gist_rainbow')
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors[0, :] = [0, 0, 0, 1]
    cmap = mcolors.LinearSegmentedColormap.from_list('Custom', colors, cmap.N)

    labels_out_unique = np.unique(labels_out)
    np.random.shuffle(labels_out_unique[1:])  # Shuffle in-place
    labels_out = np.vectorize(lambda x: labels_out_unique[x])(labels_out)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.logical_not(lattice), origin='lower', cmap='binary')  # not so filled sites are white
    axs[0].set_xticklabels([])  # Remove x-axis labels
    axs[0].set_yticklabels([])  # Remove y-axis labels
    axs[0].tick_params(axis='both', which='both', length=0)
    axs[1].imshow(labels_out, cmap=cmap, origin='lower')
    axs[1].set_xticklabels([])  # Remove x-axis labels
    axs[1].set_yticklabels([])  # Remove y-axis labels
    axs[1].tick_params(axis='both', which='both', length=0)

    plt.savefig(f'src/simpleNbrGrowth/plots/lattice2D/colored_clusters_FSPL.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()




if __name__ == '__main__':
    # main()
    # animate()
    color_clusters('src/simpleNbrGrowth/outputs/lattice2D/sigma_1_theta_0.38.csv')
