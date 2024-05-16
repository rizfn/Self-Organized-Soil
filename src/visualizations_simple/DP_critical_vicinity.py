import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def main():
    p_values = [0.62, 0.6447, 0.67]  # values below, at, and above the critical point
    L = 200
    N_steps = 500
    array = np.zeros((N_steps+1, L))

    fig, axs = plt.subplots(1, 3, figsize=(8, 16))

    for idx, p in enumerate(p_values):
        array[0, L//2] = 1
        for i in range(0, N_steps):
            for j in range(L):
                if array[i, j] == 1:
                    if array[i+1, (j-1+L)%L] == 0:
                        array[i+1, (j-1+L)%L] = np.random.rand() < p
                    if array[i+1, (j+L)%L] == 0:
                        array[i+1, (j+1+L)%L] = np.random.rand() < p

        cmap = ListedColormap([(1,1,1,0), '#901A1E'])  # Create a custom colormap
        axs[idx].imshow(np.flip(array[::2, ::2]), cmap=cmap)
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])
        axs[idx].tick_params(axis='both', which='both', length=0)
        axs[idx].set_title(f"p = {p}")
        # axs[idx].axis('off')

    plt.tight_layout()
    plt.savefig("src/visualizations_simple/plots/DP_critical_vicinity.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.show()

if __name__ == "__main__":
    main()