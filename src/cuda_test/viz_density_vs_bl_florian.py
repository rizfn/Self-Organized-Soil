import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter


def main():
    sigma = 1
    theta = 0.5

    files = glob.glob(f"src/cuda_test/outputs/density_vs_bl/*")

    bl_list = []
    mean_list = []
    std_list = []

    for i, file in enumerate(files):
        L = int(file.split("\\")[-1].split("_")[1])
        bl = int(file.split("\\")[-1].split("_")[3].split(".")[0])

        if bl == 0:
            occupied_fracs = np.genfromtxt(file, delimiter=',', skip_header=1, unpack=True)
            stochastic_value = np.mean(occupied_fracs[1000:])
            stochastic_error = np.std(occupied_fracs[1000:] / np.sqrt(len(occupied_fracs[1000:])))
            continue

        bl_list.append(bl)

        occupied_fracs = np.genfromtxt(file, delimiter=',', skip_header=1, unpack=True)

        mean_list.append(np.mean(occupied_fracs[1000:]))
        std_list.append(np.std(occupied_fracs[1000:]) / np.sqrt(len(occupied_fracs[1000:])))

    print(f"{stochastic_value} +- {stochastic_error}")
    for i in range(len(bl_list)):
        print(f"{bl_list[i]}: {mean_list[i]} +- {std_list[i]}")

    plt.rcParams['font.family'] = 'monospace'

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot([0, 40], [stochastic_value, stochastic_value], label="Stochastic density", linestyle='--', alpha=0.8, c='#666666')
    axs[0].fill_between([0, 40], [stochastic_value-stochastic_error, stochastic_value-stochastic_error], [stochastic_value+stochastic_error, stochastic_value+stochastic_error], color='#666666', alpha=0.2)
    axs[0].errorbar(bl_list, mean_list, yerr=std_list, fmt='x', elinewidth=1, label="Mean density at equilibrium", c='#901A1E')
    axs[0].set_xlabel("Block Length $L_b$")
    axs[0].set_ylabel("Equilibrium occupied fraction")
    axs[0].set_xscale("log")
    # axs[0].set_yscale("log")
    axs[0].grid()
    axs[0].legend()

    total_error = np.sqrt(np.array(std_list)**2 + stochastic_error**2)

    axs[1].errorbar(bl_list, np.array(mean_list)-stochastic_value, yerr=total_error, fmt='x', linestyle='', elinewidth=1, c='#901A1E')
    axs[1].set_xlabel("Block Length $L_b$")
    axs[1].set_ylabel("Deviation from stochastic value")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(f"src/cuda_test/plots/density_vs_bl/florian_sigma_{sigma}_theta_{theta}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
