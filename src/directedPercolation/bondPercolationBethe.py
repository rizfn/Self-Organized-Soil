import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from cycler import cycler

def plot_each_timestep():
    coordination_number = 3
    p = 0.271
    # p = 1  # debugging
    N_steps = 5

    def plot_graph(adjacency_list, node_values, i):
        G = nx.Graph()
        for node, neighbours in adjacency_list.items():
            for neighbour in neighbours:
                if node != neighbour:  # Skip adding an edge if the node is the same as the neighbour
                    G.add_edge(node, neighbour)
    
        # Create a dictionary mapping each node to its color
        node_color_dict = {node: ('#901A1E' if node_value else 'white') for node, node_value in enumerate(node_values)}
    
        # Create a list of colors in the order of the nodes in the graph
        color_map = [node_color_dict[node] for node in G.nodes]
    
        pos = nx.spring_layout(G, k=0.1, iterations=1000)  # Use spring layout
    
        # Draw nodes with increased size
        nx.draw_networkx_nodes(G, pos, node_color=color_map, edgecolors='black', node_size=500)
    
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='#666666')
    
        # Create a dictionary mapping each node to its label color
        label_color_dict = {node: ('white' if node_value else 'black') for node, node_value in enumerate(node_values)}
    
        # Create a dictionary mapping each node to its label
        labels = {node: node for node in G.nodes}
    
        # Draw labels with increased font size
        for node, (x, y) in pos.items():
            plt.text(x, y, labels[node], fontsize=16, ha='center', va='center',
                     color=label_color_dict[node])
        plt.savefig(f'src/directedPercolation/plots/bethe/schematic/{i}.png', dpi=300)
        plt.show()

    
    node_values = [1] + [0] * coordination_number
    
    adjacency_list = {0: list(range(0, coordination_number + 1))}
    for i in range(1, coordination_number + 1):
        adjacency_list[i] = [0, i]  # Each node is a neighbour of itself
        
    last_node = coordination_number

    plot_graph(adjacency_list, node_values, 0)

    for i in range(N_steps):
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

        plot_graph(adjacency_list, node_values, i+1)


def get_distance_to_origin(node, adjacency_list):
    distance = 0
    while node != 0:
        node = min(adjacency_list[node])  # Always choose the neighbor with the lowest index
        distance += 1
    return distance

def main():
    coordination_number = 3
    p = 0.270
    N_steps = 1000
    N_simulations = 100000
    mean_squared_distance = np.zeros((N_steps, N_simulations))
    n_filled = np.zeros((N_steps, N_simulations))
    n_filled[0, :] = 1
    
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
            active_nodes = [node for node, value in enumerate(node_values) if value]
            total_squared_distance = sum(get_distance_to_origin(node, adjacency_list)**2 for node in active_nodes)
            mean_squared_distance[i, sim] = total_squared_distance / len(active_nodes)


    # Calculate survival probability
    survival_prob = np.sum(n_filled > 0, axis=1) / N_simulations
    
    # Calculate average number of filled nodes
    avg_filled = np.mean(n_filled, axis=1)

    color_cycle = cycler(color=['#901A1E', '#666666', '#17BEBB'])
    plt.rcParams['axes.prop_cycle'] = color_cycle
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot survival probability
    axs[0].loglog(range(N_steps), survival_prob, label='$P_s(t)$')
    ylim = axs[0].get_ylim()
    axs[0].loglog(range(N_steps), 4e0*np.power(np.arange(N_steps).astype(float), -1), label='$\delta$=1 power law', linestyle='--', alpha=0.5)
    axs[0].set_ylim(ylim)
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Survival Probability')
    # axs[0].set_title('Survival Probability as a function of time')
    axs[0].grid()
    axs[0].legend()
    
    # Plot average number of filled nodes
    axs[1].loglog(range(N_steps), avg_filled, label='$N_A(t)$')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Average Number of Filled Nodes')
    # axs[1].set_title('Average Number of Filled Nodes as a function of time')
    axs[1].grid()
    axs[1].legend()

    # Plot mean squared distance
    axs[2].loglog(range(N_steps), np.mean(mean_squared_distance, axis=1), label='$R^2(t)$')
    ylim = axs[2].get_ylim()
    axs[2].loglog(range(N_steps), 7e-1*np.power(np.arange(N_steps).astype(float), 1), label='$2/z$=1 power law', linestyle='--', alpha=0.5)
    axs[2].set_ylim(ylim)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Mean Squared Distance from Origin')
    # axs[2].set_title('Mean Squared Distance from Origin as a function of time')
    axs[2].grid()
    axs[2].legend()

    
    plt.tight_layout()
    plt.savefig(f'src/directedPercolation/plots/bethe/p_{p}_R2.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_each_timestep()
    # main()