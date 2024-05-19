import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler


def main():
    p = 0.271
    N_steps = 1000

    # Load data
    steps, survival_prob, avg_filled, mean_squared_distance = np.loadtxt(f'src/directedPercolation/outputs/bethe/p_{p}_steps_{N_steps}.csv', delimiter=',', unpack=True, skiprows=1)

    color_cycle = cycler(color=['#901A1E', '#666666', '#17BEBB'])
    plt.rcParams['axes.prop_cycle'] = color_cycle
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot survival probability
    axs[0].loglog(steps, survival_prob, label='$P_s(t)$')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Survival Probability')
    # axs[0].set_title('Survival Probability as a function of time')
    axs[0].loglog(steps, 1/steps, label='$\delta=-1$ power law', linestyle='--')
    axs[0].grid()
    axs[0].legend()
    
    # Plot average number of filled nodes
    axs[1].loglog(steps, avg_filled, label='$N_A(t)$')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Average Number of Filled Nodes')
    # axs[1].set_title('Average Number of Filled Nodes as a function of time')
    axs[1].grid()
    axs[1].legend()

    # Plot mean squared distance
    axs[2].loglog(steps, mean_squared_distance, label='$R^2(t)$')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Mean Squared Distance from Origin')
    # axs[2].set_title('Mean Squared Distance from Origin as a function of time')
    axs[2].loglog(steps, 1/steps, label='$2/z=1$ power law', linestyle='--')
    axs[2].grid()
    axs[2].legend()

    
    plt.tight_layout()
    # plt.savefig(f'src/directedPercolation/plots/bethe/p_{p}_R2.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()