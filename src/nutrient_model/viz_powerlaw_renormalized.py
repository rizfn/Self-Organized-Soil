import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import glob
import re
import csv
import sys

def load_csv(filename):
    maxInt = sys.maxsize
    decrement = True

    while decrement:
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
        for row in reader:
            steps.append(int(row[0]))  # Convert to int and add to steps
            # Check if the row is empty before splitting and converting to int
            filled_cluster_sizes.append([int(x) for x in row[1].split(',')] if row[1] else [0])
    return steps, filled_cluster_sizes


def plot_thesis(directory, outputfilename, tau):
    csv_files = glob.glob(f'{directory}/*.tsv')
    num_files = len(csv_files)
    
    fig, axs = plt.subplots(1, num_files, figsize=(6*num_files, 5))

    for ax, filename in zip(axs, csv_files):
        match = re.search(r'sigma_\d+(\.\d+)?_theta_\d+(\.\d+)?', filename)
        if match:
            sigma_theta_part = match.group()
            sigma, theta = sigma_theta_part.split('_')[1], sigma_theta_part.split('_')[3]

        steps, filled_cluster_sizes = load_csv(filename)
        # histogram and plot all the cluster sizes
        num_bins = 100
        min_size = 1  # smallest cluster size
        max_size = max(max(sublist) for sublist in filled_cluster_sizes)
        bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

        # Calculate histograms and plot for filled clusters
        flat_filled_cluster_sizes = [item for sublist in filled_cluster_sizes for item in sublist]
        hist, edges = np.histogram(flat_filled_cluster_sizes, bins=bins, density=False)
        bin_widths = np.diff(edges)
        hist = hist / bin_widths  # Normalize by bin width
        renormalized_hist = hist * (edges[:-1] ** tau)  # Renormalize

        ax.plot(edges[:-1], renormalized_hist, 'x', label='Filled Clusters')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel(f'Cluster size $\\times x^{{{tau}}}$')
        ylim = ax.get_ylim()

        # Plot power-law line for comparison
        x = np.geomspace(min_size, max_size, num=100)
        y = renormalized_hist[0] * x**(-2.055) * x**tau
        ax.plot(x, y, linestyle='--', label='Power-law (exponent: 2.055)', alpha=0.5)
        ax.legend()
        ax.set_ylim(ylim)

        ax.set_title(f'$\sigma$: {sigma}, $\\theta$: {theta}')

    plt.tight_layout()
    plt.savefig('src/nutrient_model/plots/csd/' + outputfilename + f'_{tau}' + '.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    plot_thesis('src/nutrient_model/outputs/csd2D_paper', 'soil_clusters_paper', 2.05)

