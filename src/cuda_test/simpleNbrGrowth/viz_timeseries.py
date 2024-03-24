import numpy as np
import matplotlib.pyplot as plt

def main():
    sigma = 1
    theta = 1
    filename = f"src/cuda_test/simpleNbrGrowth/outputs/timeseriesNbrDeath/sigma_{sigma}_theta_{theta}.csv"
    fig, ax = plt.subplots(figsize=(12, 8))

    step, occupied_fracs = np.genfromtxt(filename, delimiter=',', skip_header=1, unpack=True)

    mean, sd = np.mean(occupied_fracs), np.std(occupied_fracs)
    ax.set_title(f"$\sigma$={sigma}, $\\theta$={theta}")
    ax.plot(step, occupied_fracs, label=f"$\mu$={mean:.4f}, $\sigma$={sd:.4f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Occupied fraction")
    ax.legend()
    ax.grid()
    plt.savefig(f"src/cuda_test/simpleNbrGrowth/plots/timeseriesNbrDeath/sigma_{sigma}_theta_{theta}.png", dpi=300)

    plt.show()

if __name__ == "__main__":
    main()