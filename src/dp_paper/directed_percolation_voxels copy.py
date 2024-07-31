import numpy as np
from mayavi import mlab
from tqdm import tqdm
import matplotlib.pyplot as plt
import cc3d

def main():
    p = 0.27
    np.random.seed(1)
    L = 150
    N_steps = 500
    timeskip = 2

    while True:
        array = np.zeros((N_steps+1, L, L))
        array[0, L//2, L//2] = 1
    
        for i in tqdm(range(0, N_steps)):
            for j in range(L):
                for k in range(L):
                    if array[i, j, k] == 1:
                        if array[i+1, j, k] == 0:
                            array[i+1, j, k] = np.random.rand() < p
                        if j > 0 and array[i+1, j-1, k] == 0:
                            array[i+1, j-1, k] = np.random.rand() < p
                        if j < L-1 and array[i+1, j+1, k] == 0:
                            array[i+1, j+1, k] = np.random.rand() < p
                        if k > 0 and array[i+1, j, k-1] == 0:
                            array[i+1, j, k-1] = np.random.rand() < p
                        if k < L-1 and array[i+1, j, k+1] == 0:
                            array[i+1, j, k+1] = np.random.rand() < p
    
            if np.sum(array[i+1]) == 0:
                break
        else:
            break

    condensed_array = array[::timeskip, :, :]

    labels, n_clusters = cc3d.connected_components(condensed_array[-1], connectivity=4, periodic_boundary=True, return_N=True)
    labels = np.ravel(labels)

    cluster_sizes = np.bincount(labels)
    bin_edges = np.logspace(np.log10(cluster_sizes.min()), np.log10(cluster_sizes.max()), num=20)
    hist, _ = np.histogram(cluster_sizes, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    normalized_hist = hist / bin_widths

    plt.plot(bin_edges[:-1], normalized_hist, linestyle='--', marker='x')
    plt.title(f'{p = }')
    plt.xlabel('Cluster Size')
    plt.ylabel('Normalized Frequency')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()



    # condensed_array = np.repeat(np.repeat(np.repeat(condensed_array, scaling_factor, axis=0), scaling_factor, axis=1), scaling_factor, axis=2)

    # # Adjust the figure size for better visibility
    # mlab.figure(size=(1200, 1200))

    # # Create the 3D voxel plot
    # src = mlab.pipeline.scalar_field(condensed_array)
    # mlab.pipeline.iso_surface(src, contours=[0.5], opacity=0.3, color=(0.2, 0.2, 0.2))  # Adjusted opacity for transparency

    # # Extract the 2D cross-section at the maximum i index
    # max_i = condensed_array.shape[0] - 1

    # # Add the 2D cross-section as a plane in the 3D plot
    # plane = mlab.pipeline.image_plane_widget(src,
    #                                          plane_orientation='x_axes',
    #                                          slice_index=max_i,
    #                                          colormap='Greys')

    # # Set the opacity of the plane's actors
    # for actor in plane.module_manager.scalar_lut_manager.lut.table.to_array():
    #     actor[-1] = int(0.7 * 255)  # Set the alpha value to 30% opacity

    # # Draw bounding box
    # Lx, Ly, Lz = condensed_array.shape
    # mlab.plot3d([0, Lx], [0, 0], [0, 0], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, Lx], [Ly, Ly], [0, 0], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, Lx], [0, 0], [Lz, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, Lx], [Ly, Ly], [Lz, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, 0], [0, Ly], [0, 0], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([Lx, Lx], [0, Ly], [0, 0], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, 0], [0, Ly], [Lz, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([Lx, Lx], [0, Ly], [Lz, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, 0], [0, 0], [0, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([Lx, Lx], [0, 0], [0, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([0, 0], [Ly, Ly], [0, Lz], color=(1, 1, 1), tube_radius=None)
    # mlab.plot3d([Lx, Lx], [Ly, Ly], [0, Lz], color=(1, 1, 1), tube_radius=None)

    # mlab.show()

if __name__ == "__main__":
    main()