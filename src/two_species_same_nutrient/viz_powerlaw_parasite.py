import numpy as np
import matplotlib.pyplot as plt
import csv

def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        steps = []
        cluster_sizes = []
        for row in reader:
            steps.append(int(row[0]))  # Convert to int and add to steps
            cluster_sizes.append([int(x) for x in row[1:]])  # Convert to int and add to cluster_sizes
    return steps, cluster_sizes


def main():
    # steps, cluster_sizes = load_csv('src/nutrient_mutations/outputs/parasite_CSD.csv')
    steps, cluster_sizes = load_csv('src/nutrient_mutations/outputs/parasite_CSD/sigma_0.6_theta_0.04.csv')
    # histogram and plot all the cluster sizes
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(max(sublist) for sublist in cluster_sizes[500:])
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

    # Calculate histograms and plot
    flat_cluster_sizes = [item for sublist in cluster_sizes[500:] for item in sublist]
    hist, edges = np.histogram(flat_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    plt.plot(edges[:-1], hist, 'x', label='data')
    plt.grid()
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')

    tau1, tau2 = 2, 2.3
    x = np.array(edges[:-1])
    plt.plot(x, 3e7*x**-tau1, label=r'$\tau=$' + f'{tau1} power law', linestyle='--', alpha=0.5)
    # plt.plot(x, 1e8*x**-tau2, label=r'$\tau=$' + f'{tau2} power law', linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()



    max_size = max(max(sublist) for sublist in cluster_sizes)
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)
    flat_cluster_sizes = [item for sublist in cluster_sizes for item in sublist]
    hist, edges = np.histogram(flat_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    plt.plot(edges[:-1], hist, 'x', label='data')
    plt.grid()
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.title("All data, including transients")

    tau1, tau2 = 2, 2.3
    x = np.array(edges[:-1])
    plt.plot(x, 1e7*x**-tau1, label=r'$\tau=$' + f'{tau1} power law', linestyle='--', alpha=0.5)
    plt.plot(x, 1e8*x**-tau2, label=r'$\tau=$' + f'{tau2} power law', linestyle='--', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


def histogram_key_points():
    steps, cluster_sizes = load_csv('src/nutrient_mutations/outputs/parasite_CSD.csv')
    plt.figure(figsize=(12, 8))
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(max(cluster_sizes[key_time]) for key_time in [0, 26, 160, 999] if cluster_sizes[key_time])
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)
    key_times = [0, 26, 160, 999]
    for key_time in key_times:
        selected_cluster_sizes = cluster_sizes[key_time]
        hist, edges = np.histogram(selected_cluster_sizes, bins=bins, density=False)
        bin_widths = np.diff(edges)
        hist = hist / bin_widths  # Normalize by bin width
        plt.plot(edges[:-1], hist, 'x', label=f'data at time {key_time}')
    plt.grid()
    plt.xlabel('Cluster size')
    plt.ylabel('Probability density')
    plt.title("Data at key times")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    # histogram_key_points()

