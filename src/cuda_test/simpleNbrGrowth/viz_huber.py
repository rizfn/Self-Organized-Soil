import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    sigma = 1
    theta = 0.4
    target = 0
    df = pd.read_csv(f'src/cuda_test/simpleNbrGrowth/outputs/huber/sigma_{sigma}_theta_{theta}_target_{target}.tsv', sep='\t')

    print(df.iloc[0])

    # Convert comma-separated strings to lists of integers
    df['cluster_sizes'] = df['cluster_sizes'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['cluster_lineardim'] = df['cluster_lineardim'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['box_sizes'] = df['box_sizes'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['fractal_dim'] = df['fractal_dim'].apply(lambda x: np.array([float(i) for i in x.split(',') if i]))
    df['point_fractal_dim'] = df['point_fractal_dim'].apply(lambda x: np.array([float(i) for i in x.split(',') if i]))

    # Concatenate 'cluster_sizes' and 'cluster_lineardim'
    cluster_sizes = np.concatenate(df['cluster_sizes'])
    cluster_lineardim = np.concatenate(df['cluster_lineardim'])
    box_sizes = df['box_sizes'].iloc[0]
    fractal_dim = np.mean(np.array(df['fractal_dim'].tolist()), axis=0)
    point_fractal_dim = np.mean(np.array(df['point_fractal_dim'].tolist()), axis=0)

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1: Cluster size distribution
    counts, bin_edges = np.histogram(cluster_sizes, bins=np.logspace(np.log10(1),np.log10(max(cluster_sizes)), 50))
    bin_widths = np.diff(bin_edges)
    normalized_counts = counts / bin_widths
    axs[0, 0].bar(bin_edges[:-1], normalized_counts, width=bin_widths, align='edge', log=True)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Cluster size distribution')
    axs[0, 0].set_xlabel('Cluster size')
    axs[0, 0].set_ylabel('Frequency')
    
    # Plot 2: Fractal dimension of each cluster
    axs[0, 1].plot(cluster_lineardim, cluster_sizes, '.', linestyle='None', alpha=0.1)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Fractal dimension of each cluster')
    axs[0, 1].set_xlabel('Linear dimension')
    axs[0, 1].set_ylabel('Cluster size')

    # Plot 3: Sum of fractal dimensions vs box sizes
    axs[1, 0].plot(box_sizes, fractal_dim, 'x', linestyle='None')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Fractal dimension $d_{tot}$')
    axs[1, 0].set_xlabel('Box size')
    axs[1, 0].set_ylabel('Number of intersecting boxes')

    # Plot 4: Sum of point fractal dimensions vs box sizes
    axs[1, 1].plot(box_sizes, point_fractal_dim, 'x', linestyle='None')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Point fractal dimension (each cluster is a point)')
    axs[1, 1].set_xlabel('Box size')
    axs[1, 1].set_ylabel('Number of intersecting boxes')

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'src/cuda_test/simpleNbrGrowth/outputs/huber/sigma_{sigma}_theta_{theta}_target_{target}.png')
    plt.show()


def plot_single_timestep(filename):
    df = pd.read_csv(filename, sep='\t')

    # Convert comma-separated strings to lists of integers
    df['cluster_sizes'] = df['cluster_sizes'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['cluster_lineardim'] = df['cluster_lineardim'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['box_sizes'] = df['box_sizes'].apply(lambda x: np.array([int(i) for i in x.split(',') if i]))
    df['fractal_dim'] = df['fractal_dim'].apply(lambda x: np.array([float(i) for i in x.split(',') if i]))
    df['point_fractal_dim'] = df['point_fractal_dim'].apply(lambda x: np.array([float(i) for i in x.split(',') if i]))

    # Use data for the final timestep
    final_row = df.iloc[-11]
    cluster_sizes = final_row['cluster_sizes']
    cluster_lineardim = final_row['cluster_lineardim']
    box_sizes = final_row['box_sizes']
    fractal_dim = final_row['fractal_dim']
    point_fractal_dim = final_row['point_fractal_dim']

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot 1: Cluster size distribution
    counts, bin_edges = np.histogram(cluster_sizes, bins=np.logspace(np.log10(1),np.log10(max(cluster_sizes)), 50))
    bin_widths = np.diff(bin_edges)
    normalized_counts = counts / bin_widths
    axs[0, 0].bar(bin_edges[:-1], normalized_counts, width=bin_widths, align='edge', log=True)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_title('Cluster size distribution')
    axs[0, 0].set_xlabel('Cluster size')
    axs[0, 0].set_ylabel('Frequency')
    tau = -1.8
    ylim = axs[0, 0].set_ylim()
    axs[0, 0].plot(bin_edges[:-1], normalized_counts[0]*np.power(bin_edges[:-1].astype(float), tau), label=f"$\\tau$={tau} power law", linestyle='--')
    axs[0, 0].set_ylim(ylim)
    axs[0, 0].legend()

    # Plot 2: Fractal dimension of each cluster
    axs[0, 1].plot(cluster_lineardim, cluster_sizes, '.', linestyle='None', alpha=0.1)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')
    axs[0, 1].set_title('Fractal dimension of each cluster')
    axs[0, 1].set_xlabel('Linear dimension')
    axs[0, 1].set_ylabel('Cluster size')
    D = 1.8
    ylim = axs[0, 1].set_ylim()
    axs[0, 1].plot(cluster_lineardim, np.power(cluster_lineardim.astype(float), D), label=f"$D$={D} power law", linestyle='--')
    axs[0, 1].set_ylim(ylim)
    axs[0, 1].legend()


    # Plot 3: Sum of fractal dimensions vs box sizes
    axs[1, 0].plot(box_sizes, fractal_dim, 'x', linestyle='None')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')
    axs[1, 0].set_title('Fractal dimension $d_{tot}$')
    axs[1, 0].set_xlabel('Box size')
    axs[1, 0].set_ylabel('Number of intersecting boxes')
    D_tot = 2
    ylim = axs[1, 0].set_ylim()
    axs[1, 0].plot(box_sizes, 1e6*np.power(box_sizes.astype(float), -D_tot), label=f"$D_{{tot}}$={D_tot} power law", linestyle='--')
    axs[1, 0].set_ylim(ylim)
    axs[1, 0].legend()

    # Plot 4: Sum of point fractal dimensions vs box sizes
    axs[1, 1].plot(box_sizes, point_fractal_dim, 'x', linestyle='None')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')
    axs[1, 1].set_title('Point fractal dimension (each cluster is a point)')
    axs[1, 1].set_xlabel('Box size')
    axs[1, 1].set_ylabel('Number of intersecting boxes')
    D_num = -2
    ylim = axs[1, 1].set_ylim()
    axs[1, 1].plot(box_sizes, 1e6*np.power(box_sizes.astype(float), D_num), label=f"$D_{{num}}$={D_num} power law", linestyle='--')
    axs[1, 1].set_ylim(ylim)
    axs[1, 1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    plot_single_timestep('src/cuda_test/simpleNbrGrowth/outputs/huber/sigma_1_theta_0.38_target_1.tsv')
