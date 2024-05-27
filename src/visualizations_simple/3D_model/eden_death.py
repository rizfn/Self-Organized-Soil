import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
from stl import mesh
import cc3d

def main():
    sigma = 1
    # theta = 0.404
    # fname = 'FSPL'
    theta = 0.566
    fname = 'ESPL'
    np.random.seed(0)
    L = 100
    N_steps = 200
    timeskip = 2
    scaling_factor = 4

    while True:
        array = np.zeros((N_steps+1, L, L))
        array[0, L//2, L//2] = 1
    
        # for i in tqdm(range(0, N_steps)):
        #     for j in range(L):
        #         for k in range(L):
        #             if array[i, j, k] == 1:
        #                 if np.random.random() > theta:  # if it does not die
        #                     array[i+1, j, k] = 1
        #             else:
        #                 # choose a random neighbour of the site (with solid boundary conditions) and if it's `true`, the site becomes `true` with `sigma`
        #                 nbr = np.random.choice([array[i, j, k] for j, k in [(j-1, k), (j+1, k), (j, k-1), (j, k+1)] if 0 <= j < L and 0 <= k < L])
        #                 if nbr:
        #                     array[i+1, j, k] = np.random.random() < sigma

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

    # for i in range(10):
    #     plt.cla()
    #     plt.xlim(90, 110)
    #     plt.ylim(90, 110)
    #     plt.imshow(array[i], cmap='gray')
    #     plt.show()

    array = array[::timeskip, :, :]

    array = np.repeat(np.repeat(np.repeat(array, scaling_factor, axis=0), scaling_factor, axis=1), scaling_factor, axis=2)
    # add a border of zeros to the condensed array to avoid artifacts in the 3D model
    array = np.pad(array, 1, mode='constant', constant_values=0)

    # run ccd3d to remove isolated clusters (besides the main one)
    labels = cc3d.connected_components(array, connectivity=26)
    cluster_sizes = np.bincount(labels.flat)
    cluster_sizes = cluster_sizes[1:]
    main_cluster_label = np.argmax(cluster_sizes) + 1
    array[labels != main_cluster_label] = 0

    verts, faces, normals, values = measure.marching_cubes(array, level=0)

    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]
    
    obj_3d.save(f'src/visualizations_simple/3D_model/eden_{fname}.stl')



if __name__ == "__main__":
    main()