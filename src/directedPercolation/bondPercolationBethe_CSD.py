import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_distance_to_origin(node, adjacency_list):
    distance = 0
    while node != 0:
        node = min(adjacency_list[node])  # Always choose the neighbor with the lowest index
        distance += 1
    return distance

def get_cluster_size_distribution(node_values, adjacency_list):
    cluster_sizes = []
    visited = set()
    for node, value in enumerate(node_values):
        if value and node not in visited:
            cluster_size = 0
            stack = [node]
            while stack:
                current_node = stack.pop()
                if current_node not in visited:
                    visited.add(current_node)
                    cluster_size += 1
                    stack.extend(neighbour for neighbour in adjacency_list[current_node] if node_values[neighbour])
            cluster_sizes.append(cluster_size)
    return cluster_sizes

def main():
    coordination_number = 3
    p = 0.272
    N_steps = 2000
    N_simulations = 1000
    mean_squared_distance = np.zeros((N_steps, N_simulations))
    n_filled = np.zeros((N_steps, N_simulations))
    n_filled[0, :] = 1
    cluster_sizes = []
    threshold_n_active_nodes = 100
    
    for sim in tqdm(range(N_simulations)):
        node_values = [1] + [0] * coordination_number
        
        adjacency_list = {0: list(range(0, coordination_number + 1))}
        for i in range(1, coordination_number + 1):
            adjacency_list[i] = [0, i]  # Each node is a neighbour of itself
            
        last_node = coordination_number

        for i in range(1, N_steps):
            new_node_values = [0] * len(node_values)  # Initialize new_node_values as 0s
            adjacency_list_copy = adjacency_list.copy()
            previous_last_node = last_node  # Save the value of last_node at the start of the iteration
            for node, neighbours in adjacency_list.items():
                if node_values[node]:
                    for neighbour in neighbours:
                        neighbourInfected = np.random.rand() < p
                        if neighbourInfected:
                            new_node_values[neighbour] = True  # Set new_node_values[neighbour] to True
                            if len(adjacency_list[neighbour]) == 2:  # Check if the node only has one neighbour (excluding itself)
                                for _ in range(coordination_number - 1):  # Add coordination_number - 1 neighbours
                                    adjacency_list_copy[neighbour].append(last_node+1)
                                    adjacency_list_copy[last_node+1] = [neighbour, last_node+1]  # New node is a neighbour of itself
                                    last_node += 1
        
            # Append 0s to new_node_values for the new nodes added
            new_node_values.extend([0] * (last_node - previous_last_node))
        
            adjacency_list = adjacency_list_copy
            node_values = new_node_values

            n_active_nodes = sum(node_values)
            n_filled[i, sim] = n_active_nodes
            if n_active_nodes == 0:
                break
            if n_active_nodes > threshold_n_active_nodes:
                cluster_sizes.extend(get_cluster_size_distribution(node_values, adjacency_list))
            active_nodes = [node for node, value in enumerate(node_values) if value]
            total_squared_distance = sum(get_distance_to_origin(node, adjacency_list)**2 for node in active_nodes)
            mean_squared_distance[i, sim] = total_squared_distance / len(active_nodes)


    # Calculate survival probability
    survival_prob = np.sum(n_filled > 0, axis=1) / N_simulations
    
    # Calculate average number of filled nodes
    avg_filled = np.mean(n_filled, axis=1)
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()
    
    # Plot survival probability
    axs[0].loglog(range(N_steps), survival_prob, label='$P_s$')
    ylim = axs[0].get_ylim()
    axs[0].loglog(range(N_steps), 4e0*np.power(np.arange(N_steps).astype(float), -1), label='$\delta$=1 power law', linestyle='--', alpha=0.5)
    axs[0].set_ylim(ylim)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Survival Probability')
    axs[0].set_title('Survival Probability as a function of time')
    axs[0].grid()
    axs[0].legend()
    
    # Plot average number of filled nodes
    axs[1].loglog(range(N_steps), avg_filled)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Average Number of Filled Nodes')
    axs[1].set_title('Average Number of Filled Nodes as a function of time')
    axs[1].grid()

    # Plot mean squared distance
    axs[2].loglog(range(N_steps), np.mean(mean_squared_distance, axis=1), label='$R^2$')
    ylim = axs[2].get_ylim()
    axs[2].loglog(range(N_steps), 7e-1*np.power(np.arange(N_steps).astype(float), 1), label='$2/z$=1 power law', linestyle='--', alpha=0.5)
    axs[2].set_ylim(ylim)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Mean Squared Distance from Origin')
    axs[2].set_title('Mean Squared Distance from Origin as a function of time')
    axs[2].grid()
    axs[2].legend()

    # Plot cluster size distribution
    cluster_sizes = np.array(cluster_sizes)
    hist, bins = np.histogram(cluster_sizes, bins=np.geomspace(1, cluster_sizes.max(), 50), density=True)
    hist /= np.diff(bins)
    axs[3].loglog(bins[:-1], hist, marker='x', linestyle='', label='Cluster Size Distribution')
    axs[3].set_xlabel('Cluster Size')
    axs[3].set_ylabel('Normalized Frequency')
    axs[3].set_title('Log-Log Cluster Size Distribution')
    axs[3].grid()
    axs[3].legend()

    plt.tight_layout()
    plt.savefig(f'src/directedPercolation/plots/bethe_CSD/p_{p}_steps_{N_steps}_sims_{N_simulations}.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()