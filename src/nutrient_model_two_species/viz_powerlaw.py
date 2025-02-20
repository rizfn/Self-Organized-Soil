import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import glob
import re

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


def load_csv_checkEmpty(filename):
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
            # Check if the row has at least two elements before accessing row[1]
            if len(row) > 1 and row[1]:
                filled_cluster_sizes.append([int(x) for x in row[1].split(',') if x.isdigit()])
            else:
                filled_cluster_sizes.append([0])
    return steps, filled_cluster_sizes


def main(directory, outputfilename):
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
        steps, filled_cluster_sizes = load_csv_checkEmpty(filename)
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
        ax.plot(edges[:-1], hist, 'x', label='Filled Clusters')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid()
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Probability density')
        ylim = ax.get_ylim()

        # tau1, tau2 = 2, 3
        tau1, tau2 = 1.8, 1.9
        x = np.array(edges[:-1])
        ax.plot(x, 5e6*x**-tau1, label=r'$\tau=$' + f'{tau1} power law', linestyle='--', alpha=0.5)
        ax.plot(x, 5e6*x**-tau2, label=r'$\tau=$' + f'{tau2} power law', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_ylim(ylim)

        # Extract sigma and theta from filename and set as title
        match = re.search(r'sigma_\d+(\.\d+)?_theta_\d+(\.\d+)?', filename)
        if match:
            sigma_theta_part = match.group()
            sigma, theta = sigma_theta_part.split('_')[1], sigma_theta_part.split('_')[3]

        ax.set_title(f'$\sigma$: {sigma}, $\\theta$: {theta}')

    plt.tight_layout()
    plt.savefig('src/nutrient_model_two_species/plots/csd/' + outputfilename + '.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main('src/nutrient_model_two_species/outputs/sametheta/CSD2D', '2D_L_2048')