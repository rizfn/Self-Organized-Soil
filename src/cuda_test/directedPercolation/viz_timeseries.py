import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import re

def main():
    # Get all .csv files in the directory
    files = glob.glob('src/cuda_test/directedPercolation/outputs/timeseries2D/randomIC/*.csv')

    for file in files:
        # Use regex to get the p value from the filename
        match = re.search(r'p_(\d+\.?\d*)\.csv', file)
        if match:
            p = match.group(1)

            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = np.mean(latter_half)
            std = np.std(latter_half)
            plt.plot(occupiedfracs, label=f'p={p}, $\mu$={mean:.4f} $\pm$ {std:.4f}')

    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title('Directed Percolation, random ICs')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
    plt.ylim(0, 0.75)
    plt.savefig('src/cuda_test/directedPercolation/plots/timeseries/criticalPointsRandomICs.png', dpi=300)
    plt.show()

def seedIC():
    # Get all .csv files in the directory
    files = glob.glob('src/cuda_test/directedPercolation/outputs/timeseries2D/seedIC/*.csv')

    for file in files:
        # Use regex to get the p value from the filename
        match = re.search(r'p_(\d+\.?\d*)\.csv', file)
        if match:
            p = match.group(1)

            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = np.mean(latter_half)
            std = np.std(latter_half)
            plt.plot(occupiedfracs, label=f'p={p}, $\mu$={mean:.4f} $\pm$ {std:.4f}')

    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title('Directed Percolation, 1 seed IC')
    plt.legend()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
    plt.ylim(0, 0.75)
    plt.savefig('src/cuda_test/directedPercolation/plots/timeseries/criticalPointsSeedIC.png', dpi=300)
    plt.show()

def variable_density():
    # Get all .csv files in the directory
    files = glob.glob('src/cuda_test/directedPercolation/outputs/timeseries2D/variableDensity/*.csv')

    means = []

    for file in files:
        # Use regex to get the rho and p values from the filename
        match = re.search(r'rho_(\d+\.?\d*(?:e-\d+)?)_p_(\d+\.?\d*)_[\d]+\.csv', file)
        if match:
            rho = float(match.group(1))
            p = float(match.group(2))

            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = np.mean(latter_half)
            std = np.std(latter_half)
            means.append(mean)

    min_mean = min(means)
    max_mean = max(means)

    for i, file in enumerate(files):
        match = re.search(r'rho_(\d+\.?\d*(?:e-\d+)?)_p_(\d+\.?\d*)_[\d]+\.csv', file)
        if match:
            rho = float(match.group(1))
            p = float(match.group(2))

            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = means[i]
            std = np.std(latter_half)
            color = cm.viridis((mean - min_mean) / (max_mean - min_mean))  # Normalize mean to [0, 1] and get color from colormap
            plt.plot(occupiedfracs, color=color, label=f'ρ={rho}, μ={mean:.4f} ± {std:.4f}')

    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title(f'Directed Percolation, variable density ICs, P={p}')
    plt.legend()
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    # main()
    # seedIC()
    variable_density()
