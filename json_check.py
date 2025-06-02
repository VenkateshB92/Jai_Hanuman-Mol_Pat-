import os
import json
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data

def load_graph_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Ensure required keys are present
    if 'edges' not in data or 'node_features' not in data or 'node_labels' not in data:
        raise KeyError("Missing one of the required keys: 'edges', 'node_features', or 'node_labels'.")

    # Create a PyTorch Geometric graph from the JSON data
    edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    x = torch.tensor(data['node_features'], dtype=torch.float)

    # Extract node labels from the dictionary format
    node_labels = [data['node_labels'][str(i)] for i in range(len(data['node_labels']))]

    graph = Data(x=x, edge_index=edge_index)

    return graph, node_labels

def draw_graph(graph, node_labels, output_path):
    # Convert to NetworkX graph for visualization
    edge_index = graph.edge_index.numpy()
    edges = [(edge_index[0][i], edge_index[1][i]) for i in range(edge_index.shape[1])]
    g = nx.Graph(edges)

    # Assign node labels
    labels = {i: node_labels[i] for i in range(len(node_labels))}

    # Use the first feature for node coloring
    node_values = graph.x.numpy()[:, 0]  # Assuming the first feature is used for coloring

    pos = nx.spring_layout(g)

    # Draw the graph
    plt.figure(figsize=(10, 8))
    node_colors = plt.cm.Blues(node_values)  # Using Blues colormap
    nx.draw(g, pos, labels=labels, with_labels=True, node_size=700, 
            node_color=node_values, edge_color='gray')

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues)
    sm.set_array(node_values)
    # Create colorbar and attach it to the current axes
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('Node Feature Value')

    plt.title('Molecular Structure from JSON')
    plt.savefig(output_path)
    plt.close()

def convert_json_to_images(json_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for json_file in os.listdir(json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_folder, json_file)
            try:
                graph, node_labels = load_graph_from_json(json_path)
                output_image_path = os.path.join(output_folder, f"{os.path.splitext(json_file)[0]}.png")
                draw_graph(graph, node_labels, output_image_path)
                print(f"Converted {json_file} to {output_image_path}")
            except Exception as e:
                print(f"Failed to process {json_file}: {e}")

# Example usage
json_folder = 'graph_json'  # Folder where JSON files are stored
output_folder = 'graph_images'  # Folder to save generated images

convert_json_to_images(json_folder, output_folder)
