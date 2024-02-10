from glob import glob
import pandas
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

        # plt.suptitle(result)
        # plt.title(f"3D: {sigma=}, {theta=}, L=50")
        # plt.xlabel("Step / L^3")
        # plt.ylabel("Number of sites")
        # plt.plot(df["step"], df['emptys'], label='emptys', c='grey')
        # plt.plot(df["step"], df['nutrients'], label='nutrients', c='lightblue')
        # plt.plot(df["step"], df['greens'], label='greens', c='green')
        # plt.plot(df["step"], df['soil'], label='soil', c='brown')
        # plt.legend()
        # plt.show()
                
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

    
    from matplotlib.colors import ListedColormap

    # map states to numbers
    state_dict = {"Soil": 0, "Empty": 1, "Oscillating": 2, "Stable": 3}
    df_3D['state_num'] = df_3D['state'].map(state_dict)
    df_2D['state_num'] = df_2D['state'].map(state_dict)

    # create a colormap
    cmap = ListedColormap(['brown', 'white', 'blue', 'green'])

    # Create pivot tables
    pivot_2D = df_2D.pivot(index="sigma", columns="theta", values="state_num")
    pivot_3D = df_3D.pivot(index="sigma", columns="theta", values="state_num")

    # Create a gridspec instance
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05])

    # Create a single figure
    fig = plt.figure(figsize=(12, 8))

    # Plot 2D data
    ax0 = plt.subplot(gs[0])
    im_2D = ax0.imshow(pivot_2D, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax0.set_xlabel(r"Death rate ($\theta$)")
    ax0.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax0.set_title("2D: L=500")
    ax0.invert_yaxis()

    # Plot 3D data
    ax1 = plt.subplot(gs[1])
    im_3D = ax1.imshow(pivot_3D, cmap=cmap, vmin=-0.5, vmax=3.5)
    ax1.set_xlabel(r"Death rate ($\theta$)")
    ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax1.set_title("3D: L=50")
    ax1.invert_yaxis()

    # Create a single horizontal colorbar for both plots
    cbar_ax = fig.add_subplot(gs[2:])
    cbar = fig.colorbar(im_2D, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 3])
    cbar.ax.set_xticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'])  # set the state names

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()