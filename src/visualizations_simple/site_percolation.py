import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cc3d

def main():
    L = 100
    p1 = 0.55
    p2 = 0.592746
    p3 = 0.63

    lattice1 = (np.random.rand(L, L) < p1).astype(int)
    lattice2 = (np.random.rand(L, L) < p2).astype(int)
    lattice3 = (np.random.rand(L, L) < p3).astype(int)

    # use cc3d to find the clusters
    labels1 = cc3d.connected_components(lattice1, connectivity=4, periodic_boundary=True)
    labels2 = cc3d.connected_components(lattice2, connectivity=4, periodic_boundary=True)
    labels3 = cc3d.connected_components(lattice3, connectivity=4, periodic_boundary=True)

    # find the largest non-zero cluster
    largest_cluster1 = np.argmax(np.bincount(labels1[labels1 > 0]))
    largest_cluster2 = np.argmax(np.bincount(labels2[labels2 > 0]))
    largest_cluster3 = np.argmax(np.bincount(labels3[labels3 > 0]))

    # set the largest cluster's value to 2
    lattice1[labels1 == largest_cluster1] = 2
    lattice2[labels2 == largest_cluster2] = 2
    lattice3[labels3 == largest_cluster3] = 2

    cmap = ListedColormap([(1,1,1,1), '#666666', '#901A1E'])

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(lattice1, cmap=cmap)
    axs[0].set_title(f"p = {p1}")   
    axs[0].set_xticklabels([])  # Remove x-axis labels
    axs[0].set_yticklabels([])  # Remove y-axis labels
    axs[1].imshow(lattice2, cmap=cmap)
    axs[1].set_title(f"p = {p2}")
    axs[1].set_xticklabels([])  # Remove x-axis labels
    axs[1].set_yticklabels([])  # Remove y-axis labels
    axs[2].imshow(lattice3, cmap=cmap)
    axs[2].set_title(f"p = {p3}")
    axs[2].set_xticklabels([])  # Remove x-axis labels
    axs[2].set_yticklabels([])  # Remove y-axis labels

    plt.savefig("src/visualizations_simple/plots/site_percolation.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()


if __name__ == '__main__':
    main()