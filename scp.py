import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import pickle
from itertools import combinations
from torch_geometric.data import Data

# Function 1: Load color-mapped graph data
def load_color_mapped_graphs(graph_dir):
    """
    Load color-mapped graph data from specified directory.
    
    Parameters:
        graph_dir (str): Directory containing the graph data files.
        
    Returns:
        List of networkx Graph objects.
    """
    graphs = []
    for file_name in os.listdir(graph_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(graph_dir, file_name)
            with open(file_path, 'rb') as f:
                graph = pickle.load(f)
                if isinstance(graph, Data):
                    g = nx.from_edgelist(graph.edge_index.t().tolist())
                    for i in range(graph.num_nodes):
                        # Assuming the atom label is stored in graph.x
                        atom_label = graph.x[i].numpy().tolist()[0] if graph.x is not None else str(i)
                        g.nodes[i]['atom'] = atom_label  # Assign atom label as node attribute
                    graphs.append(g)
                elif isinstance(graph, nx.Graph):
                    graphs.append(graph)
                else:
                    print(f"Unsupported graph type: {type(graph)}")
    return graphs

# Function 2: Generate subgraph patterns
def generate_subgraph_patterns(graphs, subgraph_size=3):
    """
    Generate the most frequently occurring subgraph patterns from color-mapped graphs.
    
    Parameters:
        graphs (list): List of networkx Graph objects.
        subgraph_size (int): Size of subgraphs to consider.
        
    Returns:
        Counter of subgraph patterns.
    """
    subgraph_counter = Counter()

    for g in graphs:
        for nodes in combinations(g.nodes(), subgraph_size):
            subgraph = g.subgraph(nodes)
            if nx.is_connected(subgraph):
                subgraph_counter[frozenset(subgraph.nodes())] += 1

    return subgraph_counter

# Function 3: Rank patterns and draw top structures
def rank_and_draw_patterns(subgraph_counter, top_n=10):
    """
    Rank the patterns and draw the top N structures.
    
    Parameters:
        subgraph_counter (Counter): Counter of subgraph patterns.
        top_n (int): Number of top patterns to visualize.
    """
    os.makedirs('graph_patterns', exist_ok=True)

    most_common_subgraphs = subgraph_counter.most_common(top_n)

    for idx, (subgraph, count) in enumerate(most_common_subgraphs):
        g = nx.Graph()
        g.add_nodes_from(subgraph)
        for u in subgraph:
            for v in subgraph:
                if g.has_edge(u, v):
                    g.add_edge(u, v)

        # Add debugging information
        print(f"Subgraph {idx + 1}: Nodes - {subgraph}")
        
        # Check for atom labels
        labels = {}
        for node in g.nodes():
            if 'atom' in g.nodes[node]:
                labels[node] = g.nodes[node]['atom']
            else:
                labels[node] = "Unknown"  # Fallback for missing atom labels

        pos = nx.spring_layout(g)
        nx.draw(g, pos, labels=labels, with_labels=True, node_color='blue', node_size=500)
        plt.title(f'Top Subgraph Pattern {idx + 1} (Count: {count})')
        plt.savefig(f'graph_patterns/subgraph_pattern_{idx + 1}.png')
        plt.close()

# Main execution block (Example usage)
if __name__ == "__main__":
    graph_dir = 'graph_pickle'
    graphs = load_color_mapped_graphs(graph_dir)
    
    # Generate subgraph patterns
    subgraph_counter = generate_subgraph_patterns(graphs, subgraph_size=3)
    
    # Rank and draw top patterns
    rank_and_draw_patterns(subgraph_counter)
