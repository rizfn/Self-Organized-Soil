import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

def main():
    sigma = 1
    theta = 0.5
    L = 51
    N_steps = 40
    lattice = np.zeros((L, L))
    lattice[L//2, L//2] = 1

    steps_to_record = [0, 5, 20, 40]
    lattice_snapshots = [lattice.copy()]

    for i in tqdm(range(N_steps)):
        for _ in range(L**2):
            site = np.random.randint(0, L, 2)
            if lattice[site[0], site[1]]:
                if np.random.rand() < theta:
                    lattice[site[0], site[1]] = 0
            else: # If the site is empty
                neighbour = [(-1, 0), (1, 0), (0, -1), (0, 1)][np.random.randint(0, 4)]
                neighbour = np.mod(neighbour + site, L)
                if lattice[neighbour[0], neighbour[1]]:
                    if np.random.rand() < sigma:
                        lattice[site[0], site[1]] = 1
        if i+1 in steps_to_record:
            lattice_snapshots.append(lattice.copy())
    
    fig, axs = plt.subplots(1, len(steps_to_record), figsize=(6*len(steps_to_record), 6))
    cmap = ListedColormap([(1,1,1,0), '#901A1E'])  # Create a custom colormap
    for i, step in enumerate(steps_to_record):
        axs[i].imshow(lattice_snapshots[i], cmap=cmap, vmin=0, vmax=1)
        axs[i].set_xticklabels([])  # Remove x-axis labels
        axs[i].set_yticklabels([])  # Remove y-axis labels
        axs[i].grid(True)
        # axs[i].axis('off')

    plt.savefig(f"src/visualizations_simple/plots/eden_theta_{theta}.png", dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()




if __name__ == "__main__":
    main()
