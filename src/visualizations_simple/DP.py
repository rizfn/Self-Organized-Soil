import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main():
    p = 0.6447
    L = 200
    N_steps = 400
    array = np.zeros((N_steps+1, L))
    array[0, L//2] = 1

    plt.figure(figsize=(8, 16))

    for i in range(0, N_steps):
        for j in range(L):
            if array[i, j] == 1:
                if array[i+1, (j-1+L)%L] == 0:
                    array[i+1, (j-1+L)%L] = np.random.rand() < p
                if array[i+1, (j+L)%L] == 0:
                    array[i+1, (j+1+L)%L] = np.random.rand() < p

    cmap = ListedColormap([(1,1,1,0), '#901A1E'])  # Create a custom colormap
    plt.imshow(np.flip(array), cmap=cmap)
    plt.axis('off')
    plt.savefig("src/visualizations_simple/plots/DP.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

if __name__ == "__main__":
    main()