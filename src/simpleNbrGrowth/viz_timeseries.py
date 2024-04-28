import numpy as np
import matplotlib.pyplot as plt
import re
import glob

def main():
    sigma = 1
    theta = 1
    filename = f"src/simpleNbrGrowth/outputs/timeseriesNbrDeath/sigma_{sigma}_theta_{theta}.csv"
    fig, ax = plt.subplots(figsize=(12, 8))

    step, occupied_fracs = np.genfromtxt(filename, delimiter=',', skip_header=1, unpack=True)

    mean, sd = np.mean(occupied_fracs), np.std(occupied_fracs)
    ax.set_title(f"$\sigma$={sigma}, $\\theta$={theta}")
    ax.plot(step, occupied_fracs, label=f"$\mu$={mean:.4f}, $\sigma$={sd:.4f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Occupied fraction")
    ax.legend()
    ax.grid()
    plt.savefig(f"src/simpleNbrGrowth/plots/timeseriesNbrDeath/sigma_{sigma}_theta_{theta}.png", dpi=300)

    plt.show()


def random_ICs():
    # Get all .csv files in the directory
    files = glob.glob('src/simpleNbrGrowth/outputs/timeseries/randomICs/*.csv')

    for file in files:
        # Use regex to get the sigma and theta values from the filename
        match = re.search(r'sigma_(\d+\.?\d*)_theta_(\d+\.?\d*)\.csv', file)
        if match:
            sigma = match.group(1)
            theta = match.group(2)
            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = np.mean(latter_half)
            std = np.std(latter_half)
            plt.plot(occupiedfracs, label=f'$\sigma$={sigma}, $\\theta$={theta}, $\mu$={mean:.4f} $\pm$ {std:.4f}')
            
    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title('Our model, random ICs')
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
    plt.ylim(0, 0.75)
    plt.savefig('src/simpleNbrGrowth/plots/timeseries/criticalPointsRandomICs.png', dpi=300)
    plt.show()


def seedIC():
    # Get all .csv files in the directory
    files = glob.glob('src/simpleNbrGrowth/outputs/timeseries/seedIC/*.csv')

    for file in files:
        # Use regex to get the sigma and theta values from the filename
        match = re.search(r'sigma_(\d+\.?\d*)_theta_(\d+\.?\d*)\.csv', file)
        if match:
            sigma = match.group(1)
            theta = match.group(2)
            occupiedfracs = np.loadtxt(file)
            latter_half = occupiedfracs[len(occupiedfracs)//2:]  # Select the latter half of the array
            mean = np.mean(latter_half)
            std = np.std(latter_half)
            plt.plot(occupiedfracs, label=f'$\sigma$={sigma}, $\\theta$={theta}, $\mu$={mean:.4f} $\pm$ {std:.4f}')

    plt.xlabel('time')
    plt.ylabel('occupied fraction')
    plt.title('Our model, 1 seed IC')
    plt.legend()
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7))
    plt.ylim(0, 0.75)
    plt.savefig('src/simpleNbrGrowth/plots/timeseries/criticalPointsSeedIC.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # main()
    random_ICs()
    # seedIC()