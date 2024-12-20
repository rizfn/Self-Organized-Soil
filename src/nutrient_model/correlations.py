import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numba import njit

#  NOTE: data updated so it should be too slow.

@njit(parallel=True)
def pairwise_manhattan_distances(lattice, L, source_site, target_site):
    source_indices = np.argwhere(lattice == source_site)
    target_indices = np.argwhere(lattice == target_site)
    num_sources = len(source_indices)
    num_targets = len(target_indices)
    distances = np.zeros(num_sources*num_targets, dtype=np.int64)

    for i in range(num_sources):
        for j in range(num_targets):
            dx = np.abs(source_indices[i][0] - target_indices[j][0])
            dy = np.abs(source_indices[i][1] - target_indices[j][1])
            dx = min(dx, L - dx)  # Account for periodic boundary conditions
            dy = min(dy, L - dy)  # Account for periodic boundary conditions
            distance = dx + dy
            distances[i*num_targets + j] = distance

    return distances


@njit
def distance_histogram(lattice, source_site=3, target_site=3):
    L = lattice.shape[0]
    distances = pairwise_manhattan_distances(lattice, L, source_site=source_site, target_site=target_site)
    num_sources = np.count_nonzero(lattice == source_site)
    num_targets = np.count_nonzero(lattice == target_site)
    histogram = np.bincount(distances, minlength=L+1)
    histogram = histogram / num_sources
    d = np.arange(L+1)
    expected_counts = np.where(d < L/2, 4*d, 4*(L - d))
    histogram = histogram / (expected_counts * num_targets/L**2)
    return histogram - 1


@njit
def distance_histogram_logbinned(lattice, source_site=3, target_site=3, num_bins=20):
    L = lattice.shape[0]
    distances = pairwise_manhattan_distances(lattice, L, source_site=source_site, target_site=target_site)
    num_sources = np.count_nonzero(lattice == source_site)
    num_targets = np.count_nonzero(lattice == target_site)
    bins = np.logspace(np.log10(1), np.log10(L), num=num_bins)
    hist, _ = np.histogram(distances, bins=bins)
    hist = hist / num_sources
    d = np.arange(L+1)
    expected_counts = np.where(d < L/2, 4*d, 4*(L - d))
    expected_counts = np.repeat(np.arange(len(expected_counts)), expected_counts)
    expected_counts, _ = np.histogram(expected_counts, bins=bins)
    hist = hist / (expected_counts * num_targets/L**2)
    return hist




def main_fulldata():
    data = pd.read_json("docs/data/nutrient/lattice_rho=1_delta=0.json")

    data = data[data.step == np.max(data.step)]

    source_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm
    target_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm

    data['correlation_hist'] = data.soil_lattice.apply(lambda x: distance_histogram(np.array(x), source_site, target_site))

    sigma_value1, theta_value1 = 0.2, 0.14
    sigma_value2, theta_value2 = 0.8, 0.14
    sigma_value3, theta_value3 = 0.55, 0.1

    L = len(data.soil_lattice.iloc[0])

    # Find the rows with the closest sigma and theta values
    closest1 = data.iloc[((data['sigma']-sigma_value1).abs() + (data['theta']-theta_value1).abs()).argsort()]
    closest2 = data.iloc[((data['sigma']-sigma_value2).abs() + (data['theta']-theta_value2).abs()).argsort()]
    closest3 = data.iloc[((data['sigma']-sigma_value3).abs() + (data['theta']-theta_value3).abs()).argsort()]

    plt.plot(np.arange(L+1), closest1.correlation_hist.iloc[0], label=f"sigma={round(closest1.sigma.iloc[0], 2)}, theta={round(closest1.theta.iloc[0], 2)}", marker='x', linestyle=' ')
    plt.plot(np.arange(L+1), closest2.correlation_hist.iloc[0], label=f"sigma={round(closest2.sigma.iloc[0], 2)}, theta={round(closest2.theta.iloc[0], 2)}", marker='x', linestyle=' ')
    plt.plot(np.arange(L+1), closest3.correlation_hist.iloc[0], label=f"sigma={round(closest3.sigma.iloc[0], 2)}, theta={round(closest3.theta.iloc[0], 2)}", marker='x', linestyle=' ')

    # plot powerlaw of slope -k
    k = 0.5
    x = np.linspace(1, L+1, 1000)
    y = 100 * x**(-k)
    plt.plot(x, y, label=f"powerlaw of slope -{k}", linestyle='--')

    plt.title(f"Correlation histogram for source={source_site}, target={target_site}, {L=}")
    # plt.xscale('log')
    # plt.yscale('log')

    plt.legend()  # Add a legend
    plt.show()  # Display the plot



def main():
    data = pd.read_json("docs/data/nutrient/large_lattice_rho=1_delta=0.json")

    source_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm
    target_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm

    data['correlation_hist'] = data.soil_lattice.apply(lambda x: distance_histogram(np.array(x), source_site, target_site))

    L = len(data.soil_lattice.iloc[0])


    # For each row in dataframe, plot the correlation histogram 
    # for i in range(len(data)):
    #     plt.plot(np.arange(L+1), data.correlation_hist.iloc[i], label=f"sigma={round(data.sigma.iloc[i], 2)}, theta={round(data.theta.iloc[i], 2)}", marker='x', linestyle=' ')
    

    # Group the data by theta
    grouped = data.groupby('theta')

    # For each group, plot the correlation histogram and the exponential with the same color
    for theta, group in grouped:
        color = next(plt.gca()._get_lines.prop_cycler)['color']
        for i in range(len(group)):
            plt.plot(np.arange(L+1), group.correlation_hist.iloc[i], color=color, marker='x', linestyle=' ')
            # k = np.sqrt(1/theta) + 0.12/theta  # NOTE: ARBITRARY
            k = np.sqrt(2/theta)
            x = np.linspace(1, 20, 100)
            # Set the prefactor to the value of the correlation histogram at x=1
            prefactor = 2*group.correlation_hist.iloc[i][4] / np.exp(-1/k)
            y = prefactor * np.exp(-x / k)
            plt.plot(x, y, color=color, label=f"theta={theta:.2f}, exponential of slope -{1/k:.2f}", linestyle='--')


    # # plot exponential of slope -k
    # k = 0.3
    # x = np.linspace(1, 20, 100)
    # y = np.exp(-k*x)
    # plt.plot(x, y, label=f"exponential of slope -{k}", linestyle='--')

    plt.title(f"Correlation histogram for source={source_site}, target={target_site}, {L=}")
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Manhattan distance")
    plt.ylabel("(Observed / Expected) - 1")
    plt.grid()
    plt.xlim(-0.5, 20.5)
    plt.ylim(1e-3, 2)

    plt.legend()  # Add a legend
    plt.savefig("src/nutrient_model/plots/correlation_histogram.png", dpi=300)
    plt.show()  # Display the plot




def main_logbinned():
    data = pd.read_json("docs/data/nutrient/large_lattice_rho=1_delta=0.json")
    source_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm
    target_site = 3  # 0 : empty, 1 : nutrient, 2 : soil, 3 : worm
    n_bins = 100
    data['correlation_hist'] = data.soil_lattice.apply(lambda x: distance_histogram_logbinned(np.array(x), source_site, target_site, n_bins))
    L = len(data.soil_lattice.iloc[0])
    bin_edges = np.geomspace(1, L, n_bins)
    bin_midpoints = bin_edges[:-1]
    for i in range(len(data)):
        plt.plot(bin_midpoints, data.correlation_hist.iloc[i], label=f"sigma={round(data.sigma.iloc[i], 2)}, theta={round(data.theta.iloc[i], 2)}", marker='x', linestyle=' ')
    
    # plot powerlaw of slope -k
    k = 1
    x = np.linspace(1, L+1, 1000)
    # y = x**(-k)
    y = 10**1 * x**(-k)
    # plt.plot(x, y, label=f"powerlaw of slope -{k}", linestyle='--')
    plt.title(f"Logbinned correlation histogram for source={source_site}, target={target_site}, {L=}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Manhattan distance")
    plt.grid()
    plt.legend()  # Add a legend
    plt.savefig("src/nutrient_model/plots/correlation_histogram.png", dpi=300)
    plt.show()  # Display the plot



if __name__ == "__main__":
    main()
    # main_fulldata()
    # main_logbinned()
