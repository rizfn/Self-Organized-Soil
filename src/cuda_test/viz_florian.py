import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def main():
    sigma = 1
    theta = 0.5
    # find the number of files in the directory
    files = glob.glob(f"src/cuda_test/outputs/florian/*")

    L_list = []
    bl_list = []
    occupied_data = []

    for i, file in enumerate(files):
        L = int(file.split("\\")[-1].split("_")[1])
        bl = int(file.split("\\")[-1].split("_")[3].split(".")[0])
        L_list.append(L)
        bl_list.append(bl)

        occupied_fracs = np.genfromtxt(file, delimiter=',', skip_header=1, unpack=True)

        # append the occupied data to the list
        occupied_data.append(occupied_fracs)

    # Get the indices that would sort L_list in ascending order
    sort_indices = np.argsort(L_list)

    # Use these indices to sort L_list and the occupied_data list
    L_list = np.array(L_list)[sort_indices]
    bl_list = np.array(bl_list)[sort_indices]
    occupied_data = np.array(occupied_data)[sort_indices]

    # List of colormaps
    colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']
    line_styles = ['-', '--']

    fig, axs = plt.subplots(figsize=(12, 8))

    # plot the occupied timeseries for each L
    for i, L in enumerate(np.unique(L_list)):
        # Get the indices for this Ls
        indices = np.where(L_list == L)

        # Get the occupied data and bl values for this L
        occupied_data_L = occupied_data[indices]
        bl_list_L = bl_list[indices]

        # Sort the occupied data and bl values by bl
        sort_indices = np.argsort(bl_list_L)
        occupied_data_L = occupied_data_L[sort_indices]
        bl_list_L = bl_list_L[sort_indices]

        # Create a custom colormap for this L
        cmap_L = cm.get_cmap(colormaps[i % len(colormaps)])

        # Normalize the colormap to avoid extreme light and dark values
        norm = Normalize(vmin=-1, vmax=0.8)

        # Plot the occupied timeseries for each bl, using a different shade of the color for this L
        for j, bl in enumerate(bl_list_L):
            sliced_data = occupied_data_L[j][2000:]
            mean, sd = np.mean(sliced_data), np.std(sliced_data)
            axs.plot(occupied_data_L[j], label=f"L={L}, bl={bl}, $\mu$={mean:.4f}, $\sigma$={sd:.4f}", color=cmap_L(norm(j / len(bl_list_L))), linestyle=line_styles[j % len(line_styles)], alpha=0.8)


    axs.set_title(f"Florian Algorithm")
    axs.set_xlabel("Time")
    axs.set_ylabel("Soil")
    axs.legend(ncol=3)
    axs.grid()

    # plt.savefig(f'src/cuda_test/2D_bl_sigma_{sigma}_theta_{theta}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
