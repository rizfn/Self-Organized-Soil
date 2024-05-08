import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_bethe_lattice(n_generations, coordination_number=3):
    G = nx.Graph()
    G.add_node(0)
    last_gen = [0]
    for i in range(n_generations):
        new_gen = []
        for node in last_gen:
            for _ in range(coordination_number if i == 0 else coordination_number - 1):  # Root node has 3 neighbors, others have 2
                new_node = max(G.nodes()) + 1
                G.add_node(new_node)
                G.add_edge(node, new_node)
                new_gen.append(new_node)
        last_gen = new_gen
    return G

def visualize_bethe_lattice(G):
    pos = nx.spring_layout(G, k=0.1, iterations=100)
    nx.draw(G, pos, node_color='#901A1E', edge_color='#666666')
    plt.savefig('src/visualizations_simple/plots/bethe_lattice.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

def main():
    n_generations = 3
    coordination_number = 3
    # bethe_lattice = create_bethe_lattice(n_generations, coordination_number)
    bethe_lattice = create_bethe_lattice(n_generations)
    visualize_bethe_lattice(bethe_lattice, n_generations)

if __name__ == "__main__":
    main()