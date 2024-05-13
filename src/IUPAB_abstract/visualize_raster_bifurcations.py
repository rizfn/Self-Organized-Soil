import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def main():
    df_meanfield = pandas.read_csv("src/IUPAB_abstract/outputs/TimeseriesMeanField/raster.csv")

    # map states to numbers
    state_dict = {"Soil": 0, "Empty": 1, "Oscillating": 2, "Stable": 3}
    df_meanfield['state_num'] = df_meanfield['state'].map(state_dict)

    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                            np.array([232, 233, 243])/255, 
                            np.array([66, 158, 166])/255, 
                            np.array([215, 207, 7])/255])

    # Create pivot tables
    pivot_meanfield = df_meanfield.pivot(index="sigma", columns="theta", values="state_num")

    # Create a single figure with subplots
    fig, ax0 = plt.subplots(1, 1, figsize=(9, 7))  # Adjust the figure size to accommodate the new plot
    plt.rcParams['font.family'] = 'monospace'
    
    # Plot meanfield data on the first subplot
    im_meanfield = ax0.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)  # todo: remove hardcode extent
    ylim = ax0.get_ylim()
    x = np.linspace(0.1, 0.3, 100)
    ax0.plot(x, (1-2*np.sqrt(x))/x, c='k', linestyle='--')
    ax0.set_ylim(ylim)
    ax0.set_xlabel(r"Death rate ($\theta$)")
    ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax0.set_title("Meanfield", fontweight='bold')
    ax0.invert_yaxis()
    
    # Create a single vertical colorbar for all plots
    cbar = fig.colorbar(im_meanfield, ax=ax0, orientation='vertical', ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])  # set the state names

    plt.tight_layout()

    # S0, E0, N0, W0 = 0.2, 0.4, 0.2, 0.2
    # plt.savefig(f"src/IUPAB_abstract/plots/meanfield_bifurcations/S0_{S0}_E0_{E0}_N0_{N0}_W0_{W0}.png", dpi=300)
    plt.show()

    
if __name__ == "__main__":
    main()