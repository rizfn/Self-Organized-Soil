import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cc3d

def main():
    sigma = 1
    # theta = 0.404
    # fname = 'FSPL'
    theta = 0.566
    fname = 'ESPL'
    np.random.seed(0)
    L = 200
    N_steps = 1000
    timeskip = 2
    scaling_factor = 4

    while True:
        array = np.zeros((N_steps+1, L, L))
        array[0, L//2, L//2] = 1
    
        for i in tqdm(range(0, N_steps)):
            # Create a mask for the active sites
            active_sites = array[i] == 1
        
            # Handle the case where the site does not die
            array[i+1][active_sites] = (np.random.random(active_sites.sum()) > theta).astype(int)
        
            # Handle the case where the site becomes active
            inactive_sites = np.logical_not(active_sites)
            # Create a padded version of the array for easier neighbor checking
            padded = np.pad(array[i], ((1, 1), (1, 1)), mode='constant')
            # Count the number of active neighbors for each site
            active_neighbors = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
            # Calculate the probability of becoming active based on the number of active neighbors
            activation_prob = active_neighbors / 4.0
            # Update the sites that become active
            array[i+1][inactive_sites] = (np.random.random(inactive_sites.sum()) < activation_prob[inactive_sites]).astype(int)
    
            if np.sum(array[i+1]) == 0:
                break
        else:
            break

    total_cluster_data = []
    for i in range(-100, -1):
        lattice = array[i]
        if fname == 'ESPL':
            lattice = np.logical_not(lattice)
        labels = cc3d.connected_components(lattice, connectivity=4)
        cluster_sizes = np.bincount(labels.flat)
        cluster_sizes = cluster_sizes[1:]
        total_cluster_data.extend(cluster_sizes)

    total_cluster_data = np.array(total_cluster_data)

    hist, bin_edges = np.histogram(total_cluster_data, bins=np.geomspace(1, np.max(total_cluster_data), 50))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.title('$\sigma$ = {}, $\\theta$ = {}'.format(sigma, theta))
    plt.plot(bin_centers, hist, 'o')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
        
    



if __name__ == "__main__":
    main()