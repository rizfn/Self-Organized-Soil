import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cc3d

def main():
    L = 50
    r1 = 20
    r2 = 21
    lattice = np.ones((L, L))
    for i in range(L):
        for j in range(L):
            # put points on circle of radius `20` to 0
            if r1**2 <= (i-25)**2 + (j-25)**2 < r2**2:
                lattice[i, j] = 0

    # Identify clusters of zeros
    labels, N_clusters = cc3d.connected_components(1-lattice, connectivity=4, periodic_boundary=True, return_N=True)

    # Identify clusters of zeros
    labels, N_clusters = cc3d.connected_components(1-lattice, connectivity=4, periodic_boundary=True, return_N=True)

    # Create a colormap: 0 -> white, 1 -> red, 2 -> red, 3 -> blue, 4 -> blue, etc.
    colors = [(1,1,1,1)] + [('#901A1E' if i // 2 % 2 == 0 else '#17BEBB') for i in range(1, N_clusters+1)]
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(labels, cmap=cmap)  # Use `zero_clusters` instead of `lattice`
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig("src/visualizations_simple/plots/closing_off_cluster.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()


def viz_fspl():
    L = 50
    Nsteps = 500
    p = 0.343
    lattice = np.zeros((Nsteps, L, L))
    lattice[0, :, :] = 1

    for step in range(Nsteps-1):
        for i in range(L):
            for j in range(L):
                if lattice[step, i, j] == 1:
                    if lattice[step+1, i, j] == 0:
                        lattice[step+1, i, j] = np.random.rand() < p
                    if lattice[step+1, (i+1)%L, j] == 0:
                        lattice[step+1, (i+1)%L, j] = np.random.rand() < p
                    if lattice[step+1, i, (j+1)%L] == 0:
                        lattice[step+1, i, (j+1)%L] = np.random.rand() < p
                    if lattice[step+1, (i+1)%L, (j+1)%L] == 0:
                        lattice[step+1, (i+1)%L, (j+1)%L] = np.random.rand() < p

    cmap = ListedColormap([(1,1,1,1), '#901A1E'])

    labels, n_clusters = cc3d.connected_components(lattice[-1], connectivity=4, periodic_boundary=True, return_N=True)

    # randomly shuffly the label numbering, except for 0s
    label_map = np.arange(n_clusters+1)
    np.random.shuffle(label_map[1:])
    labels = label_map[labels]
    
    # Define the start and end colors
    start_color = (1, 0.75, 0.75, 1)  # Pink
    end_color = (0.5647, 0.1019, 0.1176, 1)  # Dark red
    
    # Create a colormap that linearly interpolates between the start and end colors
    cmap = LinearSegmentedColormap.from_list("my_colormap", [start_color, end_color], N=n_clusters)
    
    # Generate the colors for each cluster
    colors = [(1,1,1,1)] + [cmap(i) for i in range(n_clusters)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # ax.imshow(lattice[-1], cmap=cmap)
    ax.imshow(labels, cmap=ListedColormap(colors))
    ax.set_xticklabels([])  # Remove x-axis labels
    ax.set_yticklabels([])  # Remove y-axis labels
    ax.tick_params(axis='both', which='both', length=0)
    plt.savefig("src/visualizations_simple/plots/fspl_2D_lattice.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()


if __name__ == '__main__':
    # main()
    viz_fspl()