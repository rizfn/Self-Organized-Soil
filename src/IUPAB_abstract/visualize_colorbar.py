import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_colorbar():
    plt.rcParams['font.size'] = 20
    # Define the colormap and state dictionary
    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                           np.array([232, 233, 243])/255, 
                           np.array([66, 158, 166])/255, 
                           np.array([215, 207, 7])/255])

    # Create a figure for the colorbar
    fig, ax = plt.subplots(figsize=(7, 1.3))  # Adjust the figure size as needed

    # Create a colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')
    
    # Set the ticks and labels
    cbar.set_ticks([0.125, 0.375, 0.625, 0.875])
    cbar.set_ticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])

    plt.tight_layout()

    # Save the colorbar
    plt.savefig("src/IUPAB_abstract/plots/colorbar/horizontal.png", dpi=300)
    plt.show()


def plot_colorbar_nooscillations():
    plt.rcParams['font.size'] = 25
    # Define the colormap and state dictionary
    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                           np.array([232, 233, 243])/255,
                           np.array([215, 207, 7])/255])

    # Create a figure for the colorbar
    fig, ax = plt.subplots(figsize=(7, 1.5))  # Adjust the figure size as needed

    # Create a colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')
    
    # Set the ticks and labels
    cbar.set_ticks([1/6, 3/6, 5/6])
    cbar.set_ticklabels(['Soil', 'Empty', 'Stable'])

    plt.tight_layout()

    # Save the colorbar
    plt.savefig("src/IUPAB_abstract/plots/colorbar/horizontal_nooscillations.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_colorbar()
    plot_colorbar_nooscillations()