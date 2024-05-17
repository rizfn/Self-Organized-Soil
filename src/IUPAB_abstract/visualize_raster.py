from glob import glob
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
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

        # if state == "Oscillating" or state == "Stable":
        #     plt.suptitle(state)
        #     plt.title(f"3D: {sigma=}, {theta=}, L=75")
        #     plt.xlabel("Step / L^3")
        #     plt.ylabel("Number of sites")
        #     plt.plot(df["step"], df['emptys'], label='emptys', c='grey')
        #     plt.plot(df["step"], df['nutrients'], label='nutrients', c='lightblue')
        #     plt.plot(df["step"], df['greens'], label='greens', c='green')
        #     plt.plot(df["step"], df['soil'], label='soil', c='brown')
        #     plt.legend()
        #     plt.show()
                
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

    # # Create a gridspec instance with 3 rows and 2 columns
    # gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.05])

    # # Create a single figure
    # fig = plt.figure(figsize=(7, 16))  # Adjust the figure size to accommodate the new plot
    # plt.rcParams['font.family'] = 'monospace'

    # # Plot meanfield data
    # ax0 = plt.subplot(gs[0, 0])
    # im_meanfield = ax0.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)  # todo: remove hardcode extent
    # ax0.set_xlabel(r"Death rate ($\theta$)")
    # ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax0.set_title("Meanfield", fontweight='bold')
    # ax0.invert_yaxis()

    # # Plot 2D data
    # ax1 = plt.subplot(gs[1, 0])
    # im_2D = ax1.imshow(pivot_2D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    # ax1.set_xlabel(r"Death rate ($\theta$)")
    # ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax1.set_title("2D: L=500", fontweight='bold')
    # ax1.invert_yaxis()

    # # Plot 3D data
    # ax2 = plt.subplot(gs[2, 0])
    # im_3D = ax2.imshow(pivot_3D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    # ax2.set_xlabel(r"Death rate ($\theta$)")
    # ax2.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax2.set_title("3D: L=75", fontweight='bold')
    # ax2.invert_yaxis()

    # # Create a single vertical colorbar for all plots
    # cbar_ax = fig.add_subplot(gs[:, 1])
    # cbar = fig.colorbar(im_2D, cax=cbar_ax, orientation='vertical', ticks=[0, 1, 2, 3])
    # cbar.ax.set_yticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])  # set the state names

    # plt.tight_layout()

    # plt.savefig("src/IUPAB_abstract/plots/raster_column.png", dpi=300)
    # plt.show()

    # # Create a gridspec instance with 2 rows and 2 columns
    # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.05])

    # # Create a single figure
    # fig = plt.figure(figsize=(7, 11))  # Adjust the figure size to accommodate the new plot
    # plt.rcParams['font.family'] = 'monospace'

    # # Plot 3D data first
    # ax0 = plt.subplot(gs[0, 0])
    # im_3D = ax0.imshow(pivot_3D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    # ax0.set_xlabel(r"Death rate ($\theta$)")
    # ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax0.set_title("3D: L=75", fontweight='bold')
    # ax0.invert_yaxis()

    # # Plot meanfield data second
    # ax1 = plt.subplot(gs[1, 0])
    # im_meanfield = ax1.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)  # todo: remove hardcode extent
    # ax1.set_xlabel(r"Death rate ($\theta$)")
    # ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    # ax1.set_title("Meanfield", fontweight='bold')
    # ax1.invert_yaxis()
    
    # # Create a single vertical colorbar for all plots
    # cbar_ax = fig.add_subplot(gs[:, 1])
    # cbar = fig.colorbar(im_3D, cax=cbar_ax, orientation='vertical', ticks=[0, 1, 2, 3])
    # cbar.ax.set_yticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])  # set the state names

    # plt.tight_layout()

    # plt.savefig("src/IUPAB_abstract/plots/raster_column_2plots.png", dpi=300)
    # plt.show()


    # Create a gridspec instance with 2 rows and 3 columns
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05])

    # Create a single figure
    fig = plt.figure(figsize=(16, 7))  # Adjust the figure size to accommodate the new plot
    plt.rcParams['font.family'] = 'monospace'

    # Plot meanfield data
    ax0 = plt.subplot(gs[0, 0])
    im_meanfield = ax0.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)  # todo: remove hardcode extent
    ax0.set_xlabel(r"Death rate ($\theta$)")
    ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax0.set_title("Meanfield", fontweight='bold')
    ax0.invert_yaxis()

    # Plot 3D data
    ax1 = plt.subplot(gs[0, 1])
    im_2D = ax1.imshow(pivot_3D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    ax1.set_xlabel(r"Death rate ($\theta$)")
    ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax1.set_title("3D: L=75", fontweight='bold')
    ax1.invert_yaxis()

    # Plot 2D data
    ax2 = plt.subplot(gs[0, 2])
    im_3D = ax2.imshow(pivot_2D, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    ax2.set_xlabel(r"Death rate ($\theta$)")
    ax2.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax2.set_title("2D: L=500", fontweight='bold')
    ax2.invert_yaxis()

    # Create a single horizontal colorbar for all plots
    cbar_ax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(im_2D, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 3])
    cbar.ax.set_xticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])  # set the state names

    plt.tight_layout()

    plt.savefig("src/IUPAB_abstract/plots/raster_row.png", dpi=300)
    plt.show()

    
if __name__ == "__main__":
    main()