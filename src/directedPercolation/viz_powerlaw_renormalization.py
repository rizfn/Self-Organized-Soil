import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import sys
import glob
import os
import re

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
            filled_cluster_sizes.append([int(x) for x in row[1].split(',') if x] if row[1] else [0])
            empty_cluster_sizes.append([int(x) for x in row[2].split(',') if x] if row[2] else [0])

    return steps, filled_cluster_sizes, empty_cluster_sizes


def main(directory, outputfilename, tau):
    csv_files = sorted(glob.glob(f'{directory}/*.tsv'))
    dimension = int(directory.rsplit('/', 3)[1][3])  # super hacky to get the dimension from fname

    n_files = len(csv_files)

    # Separate files into filled and empty based on dimension (because fspl and espl order changes at D>2)
    if dimension == 2:
        filled_files = csv_files[n_files//2:]
        empty_files = csv_files[:n_files//2]
    else:
        filled_files = csv_files[:n_files//2]
        empty_files = csv_files[n_files//2:]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # Create 2 subplots

    for i, (files, label) in enumerate(zip([filled_files, empty_files], ['Filled Clusters', 'Empty Clusters'])):
        for filename in files:
            steps, filled_cluster_sizes, empty_cluster_sizes = load_csv(filename)
            # histogram and plot all the cluster sizes
            num_bins = 100
            min_size = 1  # smallest cluster size
            max_size = max(max(max(sublist) for sublist in filled_cluster_sizes), max(max(sublist) for sublist in empty_cluster_sizes))
            bins = np.logspace(np.log10(min_size), np.log10(max_size)+1, num=num_bins)

            # Calculate histograms and plot for clusters
            flat_cluster_sizes = [item for sublist in (filled_cluster_sizes if i == 0 else empty_cluster_sizes) for item in sublist]
            hist, edges = np.histogram(flat_cluster_sizes, bins=bins, density=False)
            bin_widths = np.diff(edges)
            hist = hist / bin_widths  # Normalize by bin width
            renormalized_hist = hist * (edges[:-1] ** tau)  # Renormalize

            # Extract p and L from filename
            match = re.search(r'p_\d+(\.\d+)?_L_\d+(\.\d+)?', filename)
            if match:
                p_L_part = match.group()
                p, L = p_L_part.split('_')[1], p_L_part.split('_')[3]

            axs[i].plot(edges[:-1], renormalized_hist, label=f'{label} p: {p}')

        x = np.geomspace(min_size, max_size+1, num=100)
        y = 1e8 * x**(-2.055) * x**tau
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].grid()
        axs[i].set_xlabel('Cluster size')
        axs[i].set_ylabel(f'Cluster size $\\times x^{{{tau}}}$')
        # ylim = axs[i].get_ylim()
        # Plot power-law line
        axs[i].plot(x, y, linestyle='--', label='Power-law (exponent: 2.055)')
        # axs[i].set_ylim(ylim)
        axs[i].legend()
        axs[i].set_title(f'L: {L}')




    plt.tight_layout()
    plt.savefig('src/directedPercolation/plots/csdRenormalized/' + outputfilename + f'_{tau}' + '.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # main('src/directedPercolation/outputs/CSD2D/criticalPointsLarge/', 'criticalPoints2DLarge2', 1.87)
    main('src/directedPercolation/outputs/CSD2D/criticalPointsVLarge/', 'criticalPoints2DVLarge2', 1.85)
    # main('src/directedPercolation/outputs/CSD3D/criticalPointsCPUBCC/', 'criticalPoints3DCPUBCC', 2.25)
    # main('src/directedPercolation/outputs/CSD4D/criticalPointsCPUBCC/', 'criticalPoints4DCPUBCC', 2.7)
    # main('src/directedPercolation/outputs/CSD5D/criticalPointsCPUBCC/', 'criticalPoints5DCPUBCC', 2.8)
