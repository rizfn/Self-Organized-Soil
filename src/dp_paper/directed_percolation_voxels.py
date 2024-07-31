import numpy as np
from tqdm import tqdm
import json

def main():
    p = 0.271
    np.random.seed(1)
    L = 150
    N_steps = 350
    timeskip = 1

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

    if timeskip != 1:
        condensed_array = array[::timeskip, :, :]
    else:
        condensed_array = array

    # Save the array to a JSON file
    with open(f'src/dp_paper/outputs/DP_3D_lattice/{p=}_{L=}_{N_steps=}.json', 'w') as f:
        json.dump(condensed_array.tolist(), f)


if __name__ == "__main__":
    main()