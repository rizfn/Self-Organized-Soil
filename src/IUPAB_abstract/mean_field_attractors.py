import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numba import njit
import pandas as pd
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec
from matplotlib.markers import MarkerStyle

@njit
def ode_derivatives(S, E, N, W, sigma, theta, rho, delta):
    """Calculate the derivatives of S, E, N, W."""
    dS = sigma*S*(E+N) - W*S
    dE = (1-rho)*W*N + theta*W - sigma*S*E + delta*N
    dN = W*S - W*N - sigma*S*N - delta*N
    dW = rho*W*N - theta*W
    return dS, dE, dN, dW

@njit
def ode_integrate_rk4(sigma, theta, rho, delta, stoptime=100_000, nsteps=100_000, S_0=0.25, E_0=0.25, N_0=0.25, W_0=0.25):
    """Integrate the ODEs for the nutrient model using Runge-Kutta 4th order method."""
    dt = stoptime / nsteps

    S = np.zeros(nsteps+1)
    W = np.zeros(nsteps+1)
    E = np.zeros(nsteps+1)
    N = np.zeros(nsteps+1)
    T = np.zeros(nsteps+1)

    S[0] = S_0
    W[0] = W_0
    E[0] = E_0
    N[0] = N_0
    T[0] = 0

    for i in range(nsteps):
        k1_S, k1_E, k1_N, k1_W = ode_derivatives(S[i], E[i], N[i], W[i], sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k1_S
        E_temp = E[i] + 0.5 * dt * k1_E
        N_temp = N[i] + 0.5 * dt * k1_N
        W_temp = W[i] + 0.5 * dt * k1_W
        k2_S, k2_E, k2_N, k2_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + 0.5 * dt * k2_S
        E_temp = E[i] + 0.5 * dt * k2_E
        N_temp = N[i] + 0.5 * dt * k2_N
        W_temp = W[i] + 0.5 * dt * k2_W
        k3_S, k3_E, k3_N, k3_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S_temp = S[i] + dt * k3_S
        E_temp = E[i] + dt * k3_E
        N_temp = N[i] + dt * k3_N
        W_temp = W[i] + dt * k3_W
        k4_S, k4_E, k4_N, k4_W = ode_derivatives(S_temp, E_temp, N_temp, W_temp, sigma, theta, rho, delta)

        S[i+1] = S[i] + (dt / 6) * (k1_S + 2 * k2_S + 2 * k3_S + k4_S)
        E[i+1] = E[i] + (dt / 6) * (k1_E + 2 * k2_E + 2 * k3_E + k4_E)
        N[i+1] = N[i] + (dt / 6) * (k1_N + 2 * k2_N + 2 * k3_N + k4_N)
        W[i+1] = W[i] + (dt / 6) * (k1_W + 2 * k2_W + 2 * k3_W + k4_W)
        T[i+1] = T[i] + dt

    return T, S, E, N, W

@njit
def sigma_on_line(theta, slope, intercept):
    return slope * theta + intercept

def plot_combined():
    df_meanfield = pd.read_csv("src/IUPAB_abstract/outputs/TimeseriesMeanField/raster.csv")

    # map states to numbers
    state_dict = {"Soil": 0, "Empty": 1, "Oscillating": 2, "Stable": 3}
    df_meanfield['state_num'] = df_meanfield['state'].map(state_dict)

    cmap = ListedColormap([np.array([153, 98, 30])/255, 
                            np.array([232, 233, 243])/255, 
                            np.array([66, 158, 166])/255, 
                            np.array([215, 207, 7])/255])

    # Create pivot tables
    pivot_meanfield = df_meanfield.pivot(index="sigma", columns="theta", values="state_num")

    plt.rcParams.update({
        'font.size': 20,
        'axes.labelpad': -20,  # Move the axes labels closer to the plot
        'axes.xmargin': 0,
        'axes.ymargin': 0,
    })

    # Create a single figure with two subplots using gridspec
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])

    # Left subplot: raster with line
    im_meanfield = ax1.imshow(pivot_meanfield, cmap=cmap, vmin=-0.5, vmax=3.5, extent=[0, 0.3, 1, 0], aspect=1/3)
    ax1.set_xlabel(r"Microbe death rate ($\theta$)")
    ax1.set_ylabel(r"Soil filling rate ($\sigma$)")
    ax1.invert_yaxis()

    theta_list = np.linspace(0.125, 0.20, 6)
    color_list = plt.cm.Reds(np.linspace(0.4, 1, len(theta_list)))
    sigma_list = sigma_on_line(theta_list, 3, 0)

    for theta, sigma, color in zip(theta_list, sigma_list, color_list):
        ax1.plot(theta, sigma, 'o', color=color, markersize=10)

    # Customize tick labels
    for label in ax1.get_xticklabels()[0:-1]:
        label.set_visible(False)
    for label in ax1.get_yticklabels()[0:-1]:
        label.set_visible(False)
    ax1.annotate('0.0', xy=(0, 0), xytext=(-15, -15), textcoords='offset points', ha='center', va='center')

    # Right subplot: 3D plot of final attractors
    plt.rcParams.update({
        'axes.labelpad': 0,
    })
    ax2 = fig.add_subplot(gs[:, 1], projection='3d')

    ax2.set_xlabel('E')
    ax2.set_ylabel('S')
    ax2.set_zlabel('M')
    ax2.view_init(elev=30, azim=45)  # Change these values to set the desired orientation    

    for theta, sigma, color in zip(theta_list, sigma_list, color_list):
        T, S, E, N, W = ode_integrate_rk4(sigma, theta, 1, 0)

        # Select the last 10% of the data points
        start_index = int(0.9 * len(T))

        if theta < 0.13:
            marker = MarkerStyle('o', fillstyle = 'left')
        elif 0.13 <= theta < 0.169:
            marker = ''
        elif 0.169 <= theta < 0.196:
            marker = 'o'
        else:
            marker = MarkerStyle('o', fillstyle = 'right')

        ax2.plot(E[start_index:], S[start_index:], W[start_index:], label=f'{sigma}, {theta}', c=color, marker=marker, markersize=20, markeredgewidth=0)

    # Customize tick labels for 3D plot
    for axis in [ax2.xaxis, ax2.yaxis, ax2.zaxis]:
        ticks = axis.get_major_ticks()
        for tick in ticks:
            tick.label1.set_visible(False)
        ticks[0].label1.set_visible(True)
        if axis == ax2.yaxis:  # hacky
            ticks[-1].label1.set_visible(True)
            ticks[-1].label1.set_ha('right')
        else:
            ticks[-2].label1.set_visible(True)
            if axis == ax2.xaxis:
                ticks[-2].label1.set_ha('left')

    # Create a single horizontal colorbar for the heatmap
    cbar_ax = fig.add_subplot(gs[1, 0])
    cbar = fig.colorbar(im_meanfield, cax=cbar_ax, orientation='horizontal', ticks=[0, 1, 2, 3])
    cbar.ax.set_xticklabels(['Soil', 'Empty', 'Oscillating', 'Stable'], fontsize=18)  # set the state names

    plt.tight_layout(pad=1)

    plt.savefig("src/IUPAB_abstract/plots/meanfield_trajectories/meanfield_raster_attractors.png", dpi=500)
    plt.show()

if __name__ == "__main__":
    plot_combined()