import numpy as np
import matplotlib.pyplot as plt
import csv
import glob

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
    csv_files = glob.glob('src/cuda_test/simpleNbrGrowth/outputs/csd/*.csv')
    num_files = len(csv_files)
    
    # Calculate the number of rows and columns for the subplot grid
    num_cols = int(np.ceil((num_files)**0.5))
    num_rows = int(np.ceil(num_files / num_cols))

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 4*num_rows))

    # Flatten the axs array and remove any extra subplots
    axs = axs.flatten()[:num_files]

    for ax, filename in zip(axs, csv_files):
        steps, cluster_sizes = load_csv(filename)
        # histogram and plot all the cluster sizes
        num_bins = 100
        min_size = 1  # smallest cluster size
        max_size = max(max(sublist) for sublist in cluster_sizes)
        bins = np.logspace(np.log10(min_size), np.log10(max_size), num=num_bins)

        # Calculate histograms and plot
        flat_cluster_sizes = [item for sublist in cluster_sizes for item in sublist]
        hist, edges = np.histogram(flat_cluster_sizes, bins=bins, density=False)
        bin_widths = np.diff(edges)
        hist = hist / bin_widths  # Normalize by bin width
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(edges[:-1], hist, 'x', label='Simulation')
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

        # Extract sigma and theta from filename and set as title
        sigma, theta = filename.split('_')[2], filename.split('_')[4].rsplit('.', 1)[0]
        ax.set_title(f'$\sigma$: {sigma}, $\\theta$: {theta}')

    plt.tight_layout()
    plt.savefig('src/cuda_test/simpleNbrGrowth/plots/csd_powerlaw.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

