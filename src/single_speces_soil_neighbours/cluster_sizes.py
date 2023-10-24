import numpy as np
from tqdm import tqdm
from scipy import ndimage
import matplotlib.pyplot as plt
from soil_neighbour_utils import run_stochastic


def calculate_cluster_sizes(soil_lattice_data):
    """Calculate the cluster sizes for each timestep.

    Parameters
    ----------
    soil_lattice_data : ndarray
        Array of soil_lattice data for each timestep.
    
    Returns
    -------
    cluster_sizes : ndarray
        Array of cluster sizes for each timestep.
    """
    cluster_sizes = []
    for i in range(len(soil_lattice_data)):
        m = soil_lattice_data[i] == 1
        lw, num = ndimage.label(m)
        sizes = ndimage.sum(m, lw, index=np.arange(num + 1))
        cluster_sizes.append(sizes)
    return cluster_sizes

def main():

    # initialize the parameters
    steps_per_latticepoint = 10_000  # number of time steps for each lattice point
    L = 100  # side length of the square lattice
    n_steps = steps_per_latticepoint * L**2  # number of bacteria moves
    N = int(L**2 / 10)  # initial number of bacteria
    r = 1  # reproduction rate
    d = 0.03  # death rate
    s = 0.75  # soil filling rate

    steps_to_record = np.arange(n_steps//2, n_steps, 10*steps_per_latticepoint, dtype=np.int32)

    # run the simulation
    soil_lattice_data = run_stochastic(n_steps, L, r, d, s, steps_to_record)

    # calculate the cluster sizes
    cluster_sizes = calculate_cluster_sizes(soil_lattice_data)

    combined_cluster_sizes = np.concatenate(cluster_sizes)

    hist, edges = np.histogram(combined_cluster_sizes, bins=100, density=True)
    plt.title(f'{L=}, {r=}, {d=}, {s=}')
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.plot(edges[:-1], hist, 'x', label='data')
    # plot power law with exponent -2
    x = np.array(edges[:-1])
    plt.plot(x, 5*x**-2, label=r'$\tau=2$ power law')
    plt.plot(x, x**-1, label=r'$\tau=1$ power law')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

