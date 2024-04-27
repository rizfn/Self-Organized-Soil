import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from tqdm import tqdm


def blocks(file):
    block = []
    for line in file:
        if line.strip():
            block.append(line)
        elif block:
            yield block
            block = []

def get(filename):
    lattices = []
    with open(filename, 'r') as file:
        for block in blocks(file):
            lattice = np.genfromtxt(block, delimiter=',')
            lattices.append(lattice)
    return lattices

def periodic_distance(x, y, L):
    dx = np.abs(x - y)
    dx = np.where(dx > 0.5 * L, L - dx, dx)
    return np.sqrt(np.sum(dx**2, axis=-1))

from itertools import product
from scipy.spatial import cKDTree

def periodic_delaunay(points, L):
    # Create the supercell
    super_cell = []
    translations = []
    for dx, dy in product([-L, 0, L], repeat=2):
        super_cell.append(points + [dx, dy])
        translations.append([dx, dy])
    super_cell = np.concatenate(super_cell)

    # Compute the Delaunay triangulation for the supercell
    tri = Delaunay(super_cell)

    # Map the triangles back to the original lattice
    tree = cKDTree(points)
    _, idx = tree.query(tri.points)

    # Get the links between points only in the original cell
    delaunay_simplices = []
    for simplex in tri.simplices:
        mapped_simplex = [idx[simplex[i]] for i in range(3)]
        if len(set(mapped_simplex)) == 3:  # Ensure that each simplex consists of unique points
            delaunay_simplices.append(mapped_simplex)

    # Inverse map the simplices to the original cell
    delaunay_simplices = np.mod(delaunay_simplices, len(points))

    return np.array(delaunay_simplices)


def main():
    p = 0.2873
    L = 512

    filename = f'src/cuda_test/directedPercolation/outputs/latticeEvolution2D/nbrDist_p_{p}_L_{L}.csv'

    distances_list = []

    with open(filename, 'r') as file:
        for block in tqdm(blocks(file)):
            lattice = np.genfromtxt(block, delimiter=',')
            true_points = np.argwhere(lattice)
    
            # If there's not enough diversity in the x and y coordinates, calculate the distances directly
            if np.unique(true_points[:, 0]).size < 2:
                sorted_points = np.sort(true_points[:, 1])  # Sort based on y-coordinates
                distances = np.diff(sorted_points)
                # Add the distance between the first and last point for periodic boundary conditions
                distances = np.append(distances, sorted_points[0] + L - sorted_points[-1])
            elif np.unique(true_points[:, 1]).size < 2:
                sorted_points = np.sort(true_points[:, 0])  # Sort based on x-coordinates
                distances = np.diff(sorted_points)
                # Add the distance between the first and last point for periodic boundary conditions
                distances = np.append(distances, sorted_points[0] + L - sorted_points[-1])
            else:
                # vor = Voronoi(true_points)
                # voronoi_neighbors = vor.ridge_points
                # distances = np.linalg.norm(true_points[voronoi_neighbors[:, 0]] - true_points[voronoi_neighbors[:, 1]], axis=1)
                # tri = Delaunay(true_points)
                # delaunay_simplices = tri.simplices
                # distances = np.linalg.norm(true_points[delaunay_simplices[:, 0]] - true_points[delaunay_simplices[:, 1]], axis=1)   
                delaunay_simplices = periodic_delaunay(true_points, L)         
                distances = periodic_distance(true_points[delaunay_simplices[:, 0]], true_points[delaunay_simplices[:, 1]], L)
            
            distances_list.append(distances)
            
    print(f"{distances=}")
    distances = np.concatenate(distances_list)

    bins = np.logspace(np.log10(np.min(distances)), np.log10(np.max(distances)), num=50)
    counts, bin_edges = np.histogram(distances, bins=bins)
    widths = np.diff(bin_edges)
    normalized_counts = counts / widths
    
    plt.plot(bin_edges[1:], normalized_counts, 'x', label='Data')
    plt.title(f'Distances between voronoi neighbours, {L=}')
    plt.xlabel('Distance')
    plt.ylabel('Probability density')
    plt.grid()
    plt.xscale('log')
    plt.yscale('log')
    ylim = plt.ylim()
    tau1, tau2 = 2,2.5
    plt.plot(bin_edges, normalized_counts[0]*bin_edges**-tau1, label=f'$\\tau$={tau1} power law', linestyle='--', zorder=-1, alpha=0.8)
    plt.plot(bin_edges, 0.3e1*normalized_counts[0]*bin_edges**-tau2, label=f'$\\tau$={tau2} power law', linestyle='--', zorder=-1, alpha=0.8)
    plt.ylim(ylim)
    plt.legend()
    plt.savefig(f'src/cuda_test/directedPercolation/plots/nbrDist/{p=}_{L=}.png', dpi=300)
    plt.show()

def plot_voronoi(filename):
    lattice = np.loadtxt(filename)

    true_points = np.argwhere(lattice)
    vor = Voronoi(true_points)

    fig, axs = plt.subplots(1, 2)
    voronoi_plot_2d(vor, ax=axs[1], show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
    xlim, ylim = axs[1].get_xlim(), axs[1].get_ylim()
    axs[1].set_title('Voronoi diagram')
    axs[0].imshow((lattice.T), cmap='binary')
    axs[0].set_xlim(xlim)
    axs[0].set_title('Lattice')
    axs[0].set_ylim(ylim)
    plt.show()



if __name__ == '__main__':
    main()
    # plot_voronoi('src/cuda_test/directedPercolation/outputs/lattice2D/survival_p_0.2873_L_2048_steps_2000.csv')
    