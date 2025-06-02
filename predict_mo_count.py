import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from gen import GEN
from models.hgc_gcn import HGC_GCN
from utils import *
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
import networkx as nx
import matplotlib.pyplot as plt
import sys

# Create the target subgraph and graph
# G = nx.Graph()  # Main graph
# H = nx.Graph()  # Subgraph pattern

# # Check for subgraph isomorphism
# matcher = nx.algorithms.isomorphism.GraphMatcher(G, H)
# if matcher.subgraph_is_isomorphic():
#     print("Subgraph found")

def set_labels(x):
    labels = []
    for i in range(len(x)):
        if (x[i][0] != 0): labels.append("C")
        elif (x[i][1] != 0): labels.append("N")
        elif (x[i][2] != 0): labels.append("O")
        elif (x[i][3] != 0): labels.append("S")
        elif (x[i][4] != 0): labels.append("F")
        elif (x[i][5] != 0): labels.append("Si")
        elif (x[i][6] != 0): labels.append("P")
        elif (x[i][7] != 0): labels.append("Cl")
        elif (x[i][8] != 0): labels.append("Br")
        elif (x[i][9] != 0): labels.append("Mg")
        elif (x[i][11] != 0): labels.append("Na")
        elif (x[i][12] != 0): labels.append("Ca")
        elif (x[i][13] != 0): labels.append("Fe")
        elif (x[i][14] != 0): labels.append("As")
        elif (x[i][15] != 0): labels.append("Al")
        elif (x[i][16] != 0): labels.append("I")
        elif (x[i][17] != 0): labels.append("B")
        elif (x[i][18] != 0): labels.append("V")
        elif (x[i][19] != 0): labels.append("K")
        elif (x[i][20] != 0): labels.append("Tl")
        elif (x[i][21] != 0): labels.append("Yb")
        elif (x[i][22] != 0): labels.append("Sb")
        elif (x[i][23] != 0): labels.append("Sn")
        elif (x[i][24] != 0): labels.append("Ag")
        elif (x[i][25] != 0): labels.append("Pd")
        elif (x[i][26] != 0): labels.append("Co")
        elif (x[i][27] != 0): labels.append("Se")
        elif (x[i][28] != 0): labels.append("Ti")
        elif (x[i][29] != 0): labels.append("Zn")
        elif (x[i][30] != 0): labels.append("H")
        elif (x[i][31] != 0): labels.append("Li")
        elif (x[i][32] != 0): labels.append("Ge")
        elif (x[i][33] != 0): labels.append("Cu")
        elif (x[i][34] != 0): labels.append("Au")
        elif (x[i][35] != 0): labels.append("Ni")
        elif (x[i][36] != 0): labels.append("Cd")
        elif (x[i][37] != 0): labels.append("In")
        elif (x[i][38] != 0): labels.append("Mn")
        elif (x[i][39] != 0): labels.append("Zr")
        elif (x[i][40] != 0): labels.append("Cr")
        elif (x[i][40] != 0): labels.append("Pt")
        elif (x[i][40] != 0): labels.append("Hg")
        elif (x[i][40] != 0): labels.append("Pb")
        else: labels.append("X")
    
    label_dict = {i: value for i, value in enumerate(labels)}
    return label_dict
        
def draw_graph(data, explanation, word, word1):
    # Dictionary to store subgraph pattern counts
    global subgraph_patterns
    if 'subgraph_patterns' not in globals():
        subgraph_patterns = {}
    
    # Create initial graph
    edges = []
    for i in range(len(data.edge_index[0])):
        edges.append([data.edge_index[0][i].item(), data.edge_index[1][i].item()])
    g = nx.Graph(edges)

    # Process node importance
    node_mask = explanation.node_mask.squeeze().tolist()
    node_mask_values = {i: value for i, value in enumerate(node_mask)}
    nx.set_node_attributes(g, node_mask_values, 'value')

    # Process edge importance
    edge_values = explanation.edge_mask.numpy()
    for i, (u, v) in enumerate(g.edges()):
        g[u][v]['weight'] = edge_values[i]

    labels = set_labels(data.x)
    node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

    # Set up color normalization and maps
    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)
    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    node_colors = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors = [edge_cmap(edge_norm(weight)) for weight in edge_values]

    pos = nx.spring_layout(g)

    # Draw and save the original full graph without colors
    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_size=150, ax=ax, width=2)
    plt.savefig(word1, dpi=300)
    plt.close()

    # Define thresholds for importance
    NODE_THRESHOLD = 0.5
    EDGE_THRESHOLD = 0.65

    # Get important nodes and edges
    important_nodes = [node for node, value in node_mask_values.items() if value >= NODE_THRESHOLD]
    important_edges = [(u, v) for (u, v) in g.edges() if g[u][v]['weight'] >= EDGE_THRESHOLD]

    # Add nodes that are connected by important edges
    for u, v in important_edges:
        important_nodes.extend([u, v])
    important_nodes = list(set(important_nodes))

    # Create subgraph with important nodes and edges
    subgraph = g.subgraph(important_nodes).copy()

    # If subgraph is empty, return without drawing
    if len(subgraph) == 0:
        print(f"No important regions found above threshold for graph {word}")
        return

    def get_canonical_pattern(graph, labels):
        """
        Generate a simplified canonical representation of the molecular structure
        """
        if len(graph) <= 1:
            return labels[list(graph.nodes())[0]] if len(graph) == 1 else ""
        
        # Convert graph to linear notation
        visited = set()
        pattern = []
        
        def dfs(node):
            visited.add(node)
            neighbors = sorted(graph.neighbors(node))
            pattern.append(labels[node])
            
            # Add bond connections
            for neighbor in neighbors:
                if neighbor not in visited:
                    pattern.append('-')
                    dfs(neighbor)
            
        # Start DFS from the node with smallest label value
        start_node = min(graph.nodes(), key=lambda x: labels[x])
        dfs(start_node)
        
        return ''.join(pattern)

    # Process each connected component in the subgraph
    for component in nx.connected_components(subgraph):
        component_subgraph = subgraph.subgraph(component).copy()
        pattern = get_canonical_pattern(component_subgraph, labels)
        
        # Update pattern count
        if pattern:
            subgraph_patterns[pattern] = subgraph_patterns.get(pattern, 0) + 1
            
            # Save counts to file after each update
            with open('subgraph_pattern_counts.txt', 'w') as f:
                for p, count in sorted(subgraph_patterns.items(), key=lambda x: (-x[1], x[0])):
                    f.write(f"{p}: {count}\n")

    # Draw and save the subgraph with important regions
    subgraph_pos = nx.spring_layout(subgraph)
    subgraph_node_colors = [node_colors[i] for i in important_nodes]
    subgraph_edge_colors = []
    for u, v in subgraph.edges():
        edge_idx = list(g.edges()).index((u, v))
        subgraph_edge_colors.append(edge_colors[edge_idx])

    fig, ax = plt.subplots()
    nx.draw(subgraph, 
            subgraph_pos,
            labels={node: labels[node] for node in subgraph.nodes()},
            with_labels=True,
            node_color=subgraph_node_colors,
            edge_color=subgraph_edge_colors,
            node_size=150,
            font_color='black',
            ax=ax,
            width=3)

    plt.savefig(word, dpi=300)
    plt.close()

    # Save individual components if there are multiple
    num_components = nx.number_connected_components(subgraph)
    if num_components > 1:
        for i, component in enumerate(nx.connected_components(subgraph)):
            component_subgraph = subgraph.subgraph(component).copy()
            component_pos = nx.spring_layout(component_subgraph)
            
            fig, ax = plt.subplots()
            nx.draw(component_subgraph,
                   component_pos,
                   labels={node: labels[node] for node in component_subgraph.nodes()},
                   with_labels=True,
                   node_color=[node_colors[node] for node in component_subgraph.nodes()],
                   edge_color=[edge_colors[list(g.edges()).index((u, v))] 
                             for u, v in component_subgraph.edges()],
                   node_size=150,
                   font_color='black',
                   ax=ax,
                   width=3)
            
            component_filename = f"{word.rsplit('.', 1)[0]}_component_{i}.png"
            plt.savefig(component_filename, dpi=300)
            plt.close()

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='object',
            edge_mask_type='object',
            model_config=model_config,
        )
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    i = 0
    for data in loader:
        i+=1
        data = data.to(device)
        output = model(data.x, data.edge_index, data)
        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            data = data
        )
        # print(explanation.node_mask)
        # print(explanation.edge_mask)
        word = "graph" + str(i) + ".png"
        word1 = "oldgraph" + str(i) + ".png"
        try:
            draw_graph(data, explanation, word, word1)
        except:
            continue

    return total_labels.numpy().flatten(),total_preds.detach().numpy().flatten()

datasets = ['kiba']
modelings = [GEN]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 1

model_config = ModelConfig(
    mode='regression',
    task_level='graph',
    return_type='raw',
)

result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run prepare_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'model_GEN_davis.pt'
            if os.path.isfile(model_file_name):            
                # model.state_dict(torch.load(model_file_name, map_location=device))
                model = torch.load(model_file_name, map_location=torch.device('cpu'))
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                ret = [dataset, model_st] + [round(e,3) for e in ret]
                result += [ ret ]
                print('dataset,model,rmse,mse,pearson,spearman')
                print(ret)
            else:
                print('model is not available!')
with open('result.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')

