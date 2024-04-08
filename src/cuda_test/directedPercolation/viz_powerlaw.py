import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import sys


def load_csv(filename):
    maxInt = sys.maxsize
    decrement = True

    while decrement:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header
        steps = []
        filled_cluster_sizes = []
        empty_cluster_sizes = []
        for row in reader:
            steps.append(int(row[0]))  # Convert to int and add to steps
            # Check if the row is empty before splitting and converting to int
            filled_cluster_sizes.append([int(x) for x in row[1].split(',')] if row[1] else [0])
            empty_cluster_sizes.append([int(x) for x in row[2].split(',')] if row[2] else [0])
    return steps, filled_cluster_sizes, empty_cluster_sizes


def plot_one(filename):
    fig, ax = plt.subplots(figsize=(6, 4))

    steps, filled_cluster_sizes, empty_cluster_sizes = load_csv(filename)
    # histogram and plot all the cluster sizes
    num_bins = 100
    min_size = 1  # smallest cluster size
    max_size = max(max(max(sublist) for sublist in filled_cluster_sizes), max(max(sublist) for sublist in empty_cluster_sizes))
    bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

    # Calculate histograms and plot for filled clusters
    flat_filled_cluster_sizes = [item for sublist in filled_cluster_sizes for item in sublist]
    print(min(flat_filled_cluster_sizes), max(flat_filled_cluster_sizes), len(np.unique(flat_filled_cluster_sizes)))
    hist, edges = np.histogram(flat_filled_cluster_sizes, bins=bins, density=False)
    bin_widths = np.diff(edges)
    hist = hist / bin_widths  # Normalize by bin width
    ax.plot(edges[:-1], hist, 'x', label='Filled Clusters')

    # Calculate histograms and plot for empty clusters
    flat_empty_cluster_sizes = [item for sublist in empty_cluster_sizes for item in sublist]
    print(min(flat_empty_cluster_sizes), max(flat_empty_cluster_sizes), len(np.unique(flat_empty_cluster_sizes)))
    hist, edges = np.histogram(flat_empty_cluster_sizes, bins=bins, density=False)
    hist = hist / bin_widths  # Normalize by bin width
    ax.plot(edges[:-1], hist, 'x', label='Empty Clusters')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid()
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Probability density')
    ylim = ax.get_ylim()

    tau1, tau2 = 2, 3
    x = np.array(edges[:-1])
    ax.plot(x, 1e7*x**-tau1, label=r'$\tau=$' + f'{tau1} power law', linestyle='--', alpha=0.5)
    ax.plot(x, 6e6*x**-tau2, label=r'$\tau=$' + f'{tau2} power law', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_ylim(ylim)

    # Extract p and L from filename and set as title
    match = re.search(r'p_\d+(\.\d+)?_L_\d+(\.\d+)?', filename)
    if match:
        p_L_part = match.group()
        p, L = p_L_part.split('_')[1], p_L_part.split('_')[3]
    ax.set_title(f'p: {p},  L: {L}')

    plt.tight_layout()
    # plt.savefig(filename.replace('outputs', 'plots').replace('.tsv', '.png'), dpi=300)
    plt.show()



if __name__ == '__main__':
    # main()
    # plot_one('src/cuda_test/directedPercolation/outputs/CSD2D/p_0.2873_L_1024.tsv')
    # plot_one('src/cuda_test/directedPercolation/outputs/CSD2D/p_0.3_L_1024.tsv')
    # plot_one('src/cuda_test/directedPercolation/outputs/CSD2D/p_0.1_L_1024.tsv')
    plot_one('src/cuda_test/directedPercolation/outputs/CSD2D/p_0.6_L_1024.tsv')
