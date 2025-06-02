import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class ExplainableGeneDiseasePipeline:
    def __init__(self):
        self.gnn_model = GeneGNN()
        self.explainer = GraphExplainer(self.gnn_model)
        
    class GeneGNN(torch.nn.Module):
        def __init__(self, num_features=64, hidden_channels=32):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.classifier = torch.nn.Linear(hidden_channels, 1)
            
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return self.classifier(x)

    class GraphExplainer:
        def __init__(self, model):
            self.model = model
            
        def explain_prediction(self, node_idx, x, edge_index, threshold=0.1):
            """Extract subgraph that explains prediction for a specific node"""
            # Get attention weights for edges
            edge_mask = self._compute_edge_importance(node_idx, x, edge_index)
            
            # Filter important edges
            important_edges = edge_mask > threshold
            explanatory_edges = edge_index[:, important_edges]
            
            return explanatory_edges, edge_mask[important_edges]
            
        def _compute_edge_importance(self, node_idx, x, edge_index):
            """Compute importance scores for edges using gradient-based attribution"""
            x.requires_grad_(True)
            
            # Forward pass
            logits = self.model(x, edge_index)
            pred = logits[node_idx]
            
            # Compute gradients
            grad = torch.autograd.grad(pred, x)[0]
            edge_grad = torch.norm(grad[edge_index[0]] + grad[edge_index[1]], p=1, dim=1)
            
            return edge_grad / edge_grad.sum()

    def visualize_explanation(self, node_features, edge_index, target_node, 
                            node_names=None, edge_weights=None):
        """Visualize explanatory subgraph"""
        # Convert to networkx graph
        G = nx.Graph()
        
        # Add nodes
        if node_names is None:
            node_names = {i: f"Node {i}" for i in range(node_features.shape[0])}
            
        for i in range(node_features.shape[0]):
            G.add_node(i, name=node_names[i])
            
        # Add edges with weights
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            weight = edge_weights[i] if edge_weights is not None else 1.0
            G.add_edge(src, dst, weight=weight)
            
        # Draw graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color=['red' if n == target_node else 'lightblue' 
                                       for n in G.nodes()],
                             node_size=1000)
        
        # Draw edges with width proportional to importance
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=[w * 5 for w in edge_weights])
        
        # Add labels
        nx.draw_networkx_labels(G, pos, 
                              {n: G.nodes[n]['name'] for n in G.nodes()})
        
        plt.title("Explanatory Subgraph for Gene-Disease Association")
        plt.axis('off')
        return plt.gcf()

# Example usage
def example_explanation():
    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, 64)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                              [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)
    
    # Initialize pipeline
    pipeline = ExplainableGeneDiseasePipeline()
    
    # Get explanation for target node
    target_node = 0
    explanatory_edges, edge_weights = pipeline.explainer.explain_prediction(
        target_node, x, edge_index
    )
    
    # Visualize
    node_names = {
        0: "Target Disease",
        1: "Gene A",
        2: "Gene B",
        3: "Protein X",
        4: "Pathway Y",
        5: "Biological Process Z"
    }
    
    pipeline.visualize_explanation(x, explanatory_edges, target_node, 
                                 node_names, edge_weights)