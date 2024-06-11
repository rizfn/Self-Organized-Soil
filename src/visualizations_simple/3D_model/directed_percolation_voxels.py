import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import measure
from stl import mesh

def main():
    # p = 0.28734
    # fname = 'DP'
    # p = 0.3435
    # fname = 'FSPL'
    p = 0.3185
    fname = 'ESPL'
    np.random.seed(0)
    L = 150
    N_steps = 230
    timeskip = 2
    scaling_factor = 4

    while True:
        array = np.zeros((N_steps+1, L, L))
        array[0, L//2, L//2] = 1
    
        for i in tqdm(range(0, N_steps)):
            for j in range(L):
                for k in range(L):
                    if array[i, j, k] == 1:
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

    # for i in range(10):
    #     plt.cla()
    #     plt.xlim(90, 110)
    #     plt.ylim(90, 110)
    #     plt.imshow(array[i], cmap='gray')
    #     plt.show()

    condensed_array = array[::2*timeskip, ::2, ::2]

    # for i in range(10):
    #     plt.cla()
    #     plt.imshow(condensed_array[i], cmap='gray')
    #     plt.show()

    condensed_array = np.repeat(np.repeat(np.repeat(condensed_array, scaling_factor, axis=0), scaling_factor, axis=1), scaling_factor, axis=2)
    # add a border of zeros to the condensed array to avoid artifacts in the 3D model
    condensed_array = np.pad(condensed_array, 1, mode='constant', constant_values=0)

    verts, faces, normals, values = measure.marching_cubes(condensed_array, level=0)

    obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        obj_3d.vectors[i] = verts[f]
    
    obj_3d.save(f'src/visualizations_simple/3D_model/{fname}.stl')



if __name__ == "__main__":
    main()