import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import sys
import glob


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
        filled_cluster_sizes = []
        empty_cluster_sizes = []
        for row in reader:
            # Check if the row is empty before splitting and converting to int
            filled_cluster_sizes.append([int(x) for x in row[0].split(',') if x] if row[0] else [0])
            empty_cluster_sizes.append([int(x) for x in row[1].split(',') if x] if row[1] else [0])

    return filled_cluster_sizes, empty_cluster_sizes

def main(directory, outputfilename, tau1=1.8, tau2=1.9):
    csv_files = glob.glob(f'{directory}/*.tsv')
    num_files = len(csv_files)
    
    # Calculate the number of rows and columns for the subplot grid
    num_cols = int(np.ceil((num_files)**0.5))
    num_rows = int(np.ceil(num_files / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 4*num_rows))

    # Ensure axs is always a 2D array
    if num_rows == 1 and num_cols == 1:
        axs = np.array([[axs]])
    elif num_rows == 1 or num_cols == 1:
        axs = axs.reshape((num_rows, num_cols))

    # Flatten the axs array and remove any extra subplots
    axs = axs.flatten()[:num_files]

    for ax, filename in zip(axs, csv_files):
        filled_cluster_sizes, empty_cluster_sizes = load_csv(filename)

        print('-'*50)
        print("Filename: ", filename)
        print("Largest filled cluster size: ", max(max(sublist) for sublist in filled_cluster_sizes))
        print("Largest empty cluster size: ", max(max(sublist) for sublist in empty_cluster_sizes))
        print("Total number of sites", sum(sum(sublist) for sublist in filled_cluster_sizes) + sum(sum(sublist) for sublist in empty_cluster_sizes))
        # histogram and plot all the cluster sizes
        num_bins = 100
        min_size = 1  # smallest cluster size
        max_size = max(max(max(sublist) for sublist in filled_cluster_sizes), max(max(sublist) for sublist in empty_cluster_sizes))
        bins = np.logspace(np.log10(min_size), np.log10(max_size + 1), num=num_bins)

        # Calculate histograms and plot for filled clusters
        flat_filled_cluster_sizes = [item for sublist in filled_cluster_sizes for item in sublist]
        hist, edges = np.histogram(flat_filled_cluster_sizes, bins=bins, density=False)
        bin_widths = np.diff(edges)
        hist = hist / bin_widths  # Normalize by bin width
        ax.plot(edges[:-1], hist, 'x', label='Filled Clusters')

        # Calculate histograms and plot for empty clusters
        flat_empty_cluster_sizes = [item for sublist in empty_cluster_sizes for item in sublist]
        hist, edges = np.histogram(flat_empty_cluster_sizes, bins=bins, density=False)
        hist = hist / bin_widths  # Normalize by bin width
        ax.plot(edges[:-1], hist, 'x', label='Empty Clusters')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Probability density')
        ylim = ax.get_ylim()

        x = np.array(edges[:-1])
        ax.plot(x, 1e6*x**-tau1, label=r'$\tau=$' + f'{tau1} power law', linestyle='--', alpha=0.5)
        ax.plot(x, 1e6*x**-tau2, label=r'$\tau=$' + f'{tau2} power law', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_ylim(ylim)

        # Extract p and L from filename and set as title
        match = re.search(r'p_\d+(\.\d+)?_L_\d+(\.\d+)?', filename)
        if match:
            p_L_part = match.group()
            p, L = p_L_part.split('_')[1], p_L_part.split('_')[3]
        ax.set_title(f'p: {p},  L: {L}')

    plt.tight_layout()
    plt.savefig('src/sitePercolation/plots/csd/' + outputfilename + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # main('src/sitePercolation/outputs/CSD2D/', '2D', tau1=1.8, tau2=1.9)
    main('src/sitePercolation/outputs/CSD2D_ntrials/', '2D_ntrials', tau1=1.9, tau2=2.05)
