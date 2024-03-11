import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def main():
    # find the number of files in the directory
    files = glob.glob("src/two_species_same_nutrient/outputs/timeseries/*.csv")

    L_list = []
    soil_data = []

    for i, file in enumerate(files):
        L = int(file.split("\\")[-1].split("_")[1])
        L_list.append(L)

        _, _, soil, _, _ = np.genfromtxt(file, delimiter=',', skip_header=1, unpack=True)

        # append the soil data to the list
        soil_data.append(soil)

    # Get the indices that would sort L_list in ascending order
    sort_indices = np.argsort(L_list)

    # Use these indices to sort L_list and the soil_data list
    L_list = np.array(L_list)[sort_indices]
    soil_data = np.array(soil_data)[sort_indices]

    fig, axs = plt.subplots(figsize=(12, 8))

    # plot the soil timeseries for each L
    for i, L in enumerate(L_list):
        axs.plot(soil_data[i], label=f"L={L}")

    _, _, _, sigma, _, theta, _, rhofactor = file.split("\\")[-1].split("_")

    axs.set_title(f"$\\sigma=${sigma}, $\\theta=${theta}, $\\rho_2/\\rho_1=${rhofactor}")
    axs.set_xlabel("Time")
    axs.set_ylabel("Soil")
    axs.legend()
    axs.grid()

    plt.savefig(f'src/two_species_same_nutrient/plots/lattice_timeseries/confinementtest/2D_long_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.png', dpi=300)
    plt.show()


def bl_test():
    sigma = 0.5
    theta = 0.03
    rhofactor = 4
    # find the number of files in the directory
    files = glob.glob(f"src/two_species_same_nutrient/outputs/timeseries/bl_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}/*")

    L_list = []
    bl_list = []
    soil_data = []

    for i, file in enumerate(files):
        L = int(file.split("\\")[-1].split("_")[1])
        bl = int(file.split("\\")[-1].split("_")[3].split(".")[0])
        L_list.append(L)
        bl_list.append(bl)

        _, _, soil, _, _ = np.genfromtxt(file, delimiter=',', skip_header=1, unpack=True)

        # append the soil data to the list
        soil_data.append(soil)

    # Get the indices that would sort L_list in ascending order
    sort_indices = np.argsort(L_list)

    # Use these indices to sort L_list and the soil_data list
    L_list = np.array(L_list)[sort_indices]
    bl_list = np.array(bl_list)[sort_indices]
    soil_data = np.array(soil_data)[sort_indices]

    # List of colormaps
    colormaps = ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges', 'Greys']
    line_styles = ['-', '--']

    fig, axs = plt.subplots(figsize=(12, 8))

    # plot the soil timeseries for each L
    for i, L in enumerate(np.unique(L_list)):
        # Get the indices for this L
        indices = np.where(L_list == L)

        # Get the soil data and bl values for this L
        soil_data_L = soil_data[indices]
        bl_list_L = bl_list[indices]

        # Sort the soil data and bl values by bl
        sort_indices = np.argsort(bl_list_L)
        soil_data_L = soil_data_L[sort_indices]
        bl_list_L = bl_list_L[sort_indices]

        # Create a custom colormap for this L
        cmap_L = cm.get_cmap(colormaps[i % len(colormaps)])

        # Normalize the colormap to avoid extreme light and dark values
        norm = Normalize(vmin=-1, vmax=0.8)

        # Plot the soil timeseries for each bl, using a different shade of the color for this L
        for j, bl in enumerate(bl_list_L):
            sliced_data = soil_data_L[j][2000:]
            mean, sd = np.mean(sliced_data), np.std(sliced_data)
            axs.plot(soil_data_L[j], label=f"L={L}, bl={bl}, $\mu$={mean:.3f}, $\sigma$={sd:.3f}", color=cmap_L(norm(j / len(bl_list_L))), linestyle=line_styles[j % len(line_styles)], alpha=0.8)


    axs.set_title(f"$\\sigma=${sigma}, $\\theta=${theta}, $\\rho_2/\\rho_1=${rhofactor}")
    axs.set_xlabel("Time")
    axs.set_ylabel("Soil")
    axs.legend(ncol=3)
    axs.grid()

    plt.savefig(f'src/two_species_same_nutrient/plots/lattice_timeseries/confinementtest/2D_bl_sigma_{sigma}_theta_{theta}_rhofactor_{rhofactor}.png', dpi=300)
    plt.show()



if __name__ == "__main__":
    # main()
    bl_test()
