from glob import glob
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import f_oneway

def calculate_time_periods(data, prominence=0.01):
    peaks, _ = find_peaks(data, prominence=prominence)
    if len(peaks) < 2:
        return []
    time_periods = np.diff(peaks)
    return time_periods


def check_if_oscillating(df, prominence=0.01):
    oscillating = False
    time_periods_list = []
    sub_df = df[df["step"] > df["step"].max() / 2].reset_index()  # only consider the second half of the time series
    for column in ["emptys", "nutrients", "greens", "soil"]:
        time_periods = calculate_time_periods(sub_df[column], prominence)
        if not len(time_periods):
            return False
        time_periods_list.append(time_periods)
    # check if the time periods are all the same with an ANOVA test
    F, p = f_oneway(*time_periods_list)
    if p > 0.05:  # if p-value > 0.05, we cannot reject the null hypothesis that the means are equal
        oscillating = True
    return oscillating



def get_highlight_coordinates(pivot_table, pairs):
    # Assume pivot_table.index and pivot_table.columns are the sigma and theta values
    sigma_values = pivot_table.index.values.astype(float)
    theta_values = pivot_table.columns.values.astype(float)

    ellipses = []

    for sigma, theta in pairs:
        # Find the closest sigma and theta values in the pivot table
        sigma_idx = (np.abs(sigma_values - sigma)).argmin()
        theta_idx = (np.abs(theta_values - theta)).argmin()

        # Calculate the x and y coordinates of the ellipse's center
        x = theta_values[theta_idx]
        y = sigma_values[sigma_idx]

        # Calculate the width and height of the ellipse
        width = 0.01  # Adjust this value as needed
        height = width * (1.0 / 0.3)

        ellipses.append((x, y, width, height))

    return ellipses


def custom_tick_labels(ax):
    for label in ax.get_xticklabels()[0:-1]:
        label.set_visible(False)
    for label in ax.get_yticklabels()[0:-1]:
        label.set_visible(False)



def main():

    data_list_3D = []
    for filename in glob("src/IUPAB_abstract/outputs/timeseries3D/*.csv"):
        df = pandas.read_csv(filename)
        sigma = filename.split("_")[2]
        theta = filename.split("_")[4].rsplit(".", 1)[0]
        df["sigma"] = sigma
        df["theta"] = theta

        if df.iloc[-1]["greens"] == 0:
            if df.iloc[-1]["soil"] == 0:
                state = "Empty"
            else:
                state = "Soil"
        else:
            is_oscillating = check_if_oscillating(df)
            if is_oscillating:
                state = "Oscillating"
            else:
                state = "Stable"
                
        data_list_3D.append({"sigma": sigma, "theta": theta, "state": state})
    df_3D = pandas.DataFrame(data_list_3D)


    data_list_2D = []
    for filename in glob("src/IUPAB_abstract/outputs/timeseries2D/*.csv"):
        df = pandas.read_csv(filename)
        sigma = filename.split("_")[2]
        theta = filename.split("_")[4].rsplit(".", 1)[0]
        df["sigma"] = sigma
        df["theta"] = theta
        if df.iloc[-1]["greens"] == 0:
            if df.iloc[-1]["soil"] == 0:
                state = "Empty"
            else:
                state = "Soil"
        else:
            is_oscillating = check_if_oscillating(df)
            if is_oscillating:
                state = "Oscillating"
            else:
                state = "Stable"                
        data_list_2D.append({"sigma": sigma, "theta": theta, "state": state})
    df_2D = pandas.DataFrame(data_list_2D)

    df_meanfield = pandas.read_csv("src/IUPAB_abstract/outputs/TimeseriesMeanField/raster.csv")

    # map states to numbers
    state_dict = {"Soil": 0, "Empty": 1, "Oscillating": 2, "Stable": 3}
    df_meanfield['state_num'] = df_meanfield['state'].map(state_dict)
    df_2D['state_num'] = df_2D['state'].map(state_dict)
    df_3D['state_num'] = df_3D['state'].map(state_dict)

    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                            np.array([232, 233, 243])/255, 
                            np.array([66, 158, 166])/255, 
                            np.array([215, 207, 7])/255])

    # Create pivot tables
    pivot_meanfield = df_meanfield.pivot(index="sigma", columns="theta", values="state_num")
    pivot_2D = df_2D.pivot(index="sigma", columns="theta", values="state_num")
    pivot_3D = df_3D.pivot(index="sigma", columns="theta", values="state_num")

    plt.rcParams.update({
        'font.size': 20,
        'axes.labelpad': -25,  # Move the axes labels closer to the plot
        # reduce margins
        'figure.autolayout': True,
        'axes.xmargin': 0,
        'axes.ymargin': 0,
    })

    # Create a gridspec instance with 3 rows and 1 column
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.1])

    # Create a single figure
    fig = plt.figure(figsize=(7, 14))  # Adjust the figure size to make the plots square

    # highlighted = [(0.16, 0.09), (0.37, 0.06), (0.37, 0.09), (0.37, 0.16), (0.47, 0.16)]
    highlighted = [(0.37, 0.06), (0.37, 0.16)]
    
    # Plot 3D data first
    ax0 = plt.subplot(gs[0, 0])
    im_3D = ax0.imshow(pivot_3D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect='auto')
    ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax0.set_title("3D", fontweight='bold')
    ax0.invert_yaxis()
    
    custom_tick_labels(ax0)
    ax0.annotate('0.0', xy=(0, 0), xytext=(-15, -15), textcoords='offset points', ha='center', va='center')

    # Highlight specified sigma, theta pairs
    for ellipse in get_highlight_coordinates(pivot_3D, highlighted):
        x, y, width, height = ellipse
        ax0.add_patch(Ellipse((x, y), width=width, height=height, fill=False, edgecolor='#901A1E', lw=2))

    
    # Plot meanfield data second
    ax1 = plt.subplot(gs[1, 0], sharex=ax0)
    im_meanfield = ax1.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect='auto')
    ax1.xaxis.labelpad = -15
    ax1.set_xlabel(r"Death rate ($\theta$)")
    ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax1.set_title("Mean-field", fontweight='bold')
    ax1.invert_yaxis()

    custom_tick_labels(ax1)
    ax1.annotate('0.0', xy=(0, 0), xytext=(-15, -15), textcoords='offset points', ha='center', va='center')

    # Highlight specified sigma, theta pairs
    for ellipse in get_highlight_coordinates(pivot_3D, highlighted):
        x, y, width, height = ellipse
        ax1.add_patch(Ellipse((x, y), width=width, height=height, fill=False, edgecolor='#901A1E', lw=2))

    # Create a single horizontal colorbar for all plots
    cbar_ax = fig.add_axes([0.12, 0.05, 0.8, 0.02])  # Adjust the position and size of the colorbar
    cbar = fig.colorbar(im_3D, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 3])
    cbar.ax.set_xticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'], fontsize=17)  # set the state names

    # plt.tight_layout()
    plt.savefig("src/IUPAB_abstract/plots/raster_highlighted/snic.png", dpi=300)
    # plt.show()


def paper_2D_3highlights():
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelpad': -25,  # Move the axes labels closer to the plot
        # reduce margins
        'figure.autolayout': True,
        'axes.xmargin': 0,
        'axes.ymargin': 0,
    })

    data_list_2D = []
    for filename in glob("src/IUPAB_abstract/outputs/timeseries2D/*.csv"):
        df = pandas.read_csv(filename)
        sigma = filename.split("_")[2]
        theta = filename.split("_")[4].rsplit(".", 1)[0]
        df["sigma"] = sigma
        df["theta"] = theta
        if df.iloc[-1]["greens"] == 0:
            if df.iloc[-1]["soil"] == 0:
                state = "Empty"
            else:
                state = "Soil"
        else:
            is_oscillating = check_if_oscillating(df)
            if is_oscillating:
                state = "Oscillating"
            else:
                state = "Stable"                
        data_list_2D.append({"sigma": sigma, "theta": theta, "state": state})
    df_2D = pandas.DataFrame(data_list_2D)

    # map states to numbers
    state_dict = {"Soil": 0, "Empty": 1, "Oscillating": 2, "Stable": 3}
    df_2D['state_num'] = df_2D['state'].map(state_dict)

    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                            np.array([232, 233, 243])/255, 
                            np.array([66, 158, 166])/255, 
                            np.array([215, 207, 7])/255])

    # Create pivot tables
    pivot_2D = df_2D.pivot(index="sigma", columns="theta", values="state_num")

    # Create a single figure
    fig, ax = plt.subplots(figsize=(7, 7))  # Adjust the figure size to make the plot square

    highlighted = [(0.21, 0.14), (0.21, 0.09), (0.89, 0.09)]
    
    # Plot 2D data
    im_2D = ax.imshow(pivot_2D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect='auto')
    ax.xaxis.labelpad = -15
    ax.set_xlabel(r"Death rate ($\theta$)")
    ax.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax.set_title("2D", fontweight='bold')
    ax.invert_yaxis()
    
    # Hide all tick labels except the first and last
    def custom_tick_labels(ax):
        for label in ax.get_xticklabels()[0:-1]:
            label.set_visible(False)
        for label in ax.get_yticklabels()[0:-1]:
            label.set_visible(False)

    custom_tick_labels(ax)

    ax.annotate('0.0', xy=(0, 0), xytext=(-15, -15),
                textcoords='offset points', ha='center', va='center')

    # Highlight specified sigma, theta pairs
    for ellipse in get_highlight_coordinates(pivot_2D, highlighted):
        x, y, width, height = ellipse
        ax.add_patch(Ellipse((x, y), width=width, height=height, fill=False, edgecolor='#901A1E', lw=2))

    plt.tight_layout()

    plt.savefig("src/IUPAB_abstract/plots/raster_highlighted/2D_3params.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
    # paper_2D_3highlights()