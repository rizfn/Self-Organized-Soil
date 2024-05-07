import numpy as np
import matplotlib.pyplot as plt


def main():
    p = 0.271
    N_steps = 1000

    # Load data
    steps, survival_prob, avg_filled, mean_squared_distance = np.loadtxt(f'src/directedPercolation/outputs/bethe/p_{p}_steps_{N_steps}.csv', delimiter=',', unpack=True, skiprows=1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot survival probability
    axs[0].loglog(steps, survival_prob)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Survival Probability')
    axs[0].set_title('Survival Probability as a function of time')
    axs[0].grid()
    
    # Plot average number of filled nodes
    axs[1].loglog(steps, avg_filled)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Average Number of Filled Nodes')
    axs[1].set_title('Average Number of Filled Nodes as a function of time')
    axs[1].grid()

    # Plot mean squared distance
    axs[2].loglog(steps, mean_squared_distance)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Mean Squared Distance from Origin')
    axs[2].set_title('Mean Squared Distance from Origin as a function of time')
    axs[2].grid()

    
    plt.tight_layout()
    # plt.savefig(f'src/directedPercolation/plots/bethe/p_{p}_R2.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()