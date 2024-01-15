import numpy as np
from tqdm import tqdm
import pandas as pd
from twospec_samenutrient_utils import ode_integrate_rk4
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter


def run_raster_alive(n_steps, sigma, theta, rho1, mu1, rho2_list, mu2_list, tolerance=1e-6):
    """Scans the ODE integrator over rho2 and mu2 for n_steps timesteps.
    
    Parameters
    ----------
    n_steps : int
        Number of timesteps to run the simulation for.
    sigma : float
        Soil filling rate.
    theta : float
        Death rate.
    rho1 : float
        Reproduction rate of green worms.
    mu1 : float
        Nutrient creation rate of green worms.
    rho2_list : list
        List of reproduction rates of blue worms.
    mu2_list : list
        List of nutrient creation rates of blue worms.
    tolerance : float
        Tolerance for the worms to be considered alive.
    
    Returns
    -------
    alive_information : list
        List of soil_fraction data for the final timesteps and parameters.
    """
    grid = np.meshgrid(rho2_list, mu2_list)
    rho_mu_pairs = np.reshape(grid, (2, -1)).T  # all possible pairs of d and s
    alive_information = []
    for i in tqdm(range(len(rho_mu_pairs))):  # todo: parallelize
        rho2, mu2 = rho_mu_pairs[i]
        T, S, E, N, WG, WB = ode_integrate_rk4(sigma, theta, rho1, rho2, mu1, mu2, stoptime=n_steps, nsteps=n_steps)
        alive_information.append({"rho1": rho1, "rho2":rho2, "mu1": mu1, "mu2":mu2, "soil_alive":S[-1]>tolerance, "green_alive": WG[-1]>tolerance, "blue_alive": WB[-1]>tolerance})
    return alive_information


def main():  # TODO: fix

    n_steps = 100_000  # number of worm moves
    sigma = 0.5
    theta = 0.05
    rho1 = 0.5
    mu1 = 0.5
    rho2_list = np.linspace(0, 1, 40)
    mu2_list = np.linspace(0, 1, 40)

    alive_information = run_raster_alive(n_steps, sigma, theta, rho1, mu1, rho2_list, mu2_list)

    alive_information = pd.DataFrame(alive_information)

    def map_colors(row):
        if row['green_alive'] and row['blue_alive']:
            return 0  # Yellow
        elif row['green_alive']:
            return 1  # Green
        elif row['blue_alive']:
            return 2  # Blue
        elif row['soil_alive']:
            return 3  # Brown
        else:
            return 4  # Empty

    alive_information['color'] = alive_information.apply(map_colors, axis=1)
    pivot_df = alive_information.pivot(index='rho2', columns='mu2', values='color')
    cmap = colors.ListedColormap(['yellow', 'green', 'blue', 'brown', 'white'])

    data = pivot_df.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title(f"Alive worms\n{n_steps=:}, {sigma=}, {theta=}, {rho1=}, {mu1=}")
    ax.imshow(data, cmap=cmap, origin='lower', vmin=0, vmax=4)  # Use origin='lower' to start the plot from the bottom left
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, t:round (mu2_list[int(v)],2) if v<len(mu2_list) else ''))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, t: round(rho2_list[int(v)],2) if v<len(rho2_list) else ''))
    ax.set_xlabel('mu2')
    ax.set_ylabel('rho2')
    plt.savefig(f'src/two_species_same_nutrient/plots/alive_raster/meanfield_{sigma=}_{theta=}.png', dpi=300)
    plt.show()




def plot_single_run():
    n_steps = 10_000  # number of worm moves
    rho1 = 0.5
    rho2 = 1
    theta = 0.025
    mu1 = 0.5
    mu2 = 0
    sigma = 0.5

    T, S, E, N, WG, WB = ode_integrate_rk4(sigma, theta, rho1, rho2, mu1, mu2, stoptime=n_steps, nsteps=n_steps)

    plt.grid()
    plt.plot(T, S, label="soil", c='brown')
    plt.plot(T, E, label="vacancy", c='grey')
    plt.plot(T, N, label="nutrient", c='lawngreen')
    plt.plot(T, WG, label="worm green", c='green')
    plt.plot(T, WB, label="worm blue", linestyle='--', c='blue')
    plt.title(f"{sigma=}, {theta=}, {rho1=}, {rho2=}, {mu1=}, {mu2=}")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # main()
    plot_single_run()
