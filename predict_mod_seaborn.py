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
import matplotlib.gridspec as gridspec
from collections import Counter

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
    """
    Enhanced function to draw and analyze molecular graphs with importance highlighting.
    
    Parameters:
    data: PyTorch Geometric Data object containing molecular graph
    explanation: GNNExplainer explanation object
    word: filename for saving the importance-highlighted graph
    word1: filename for saving the original graph structure
    
    Returns:
    dict: Analysis results for the important regions
    """
    
    def analyze_subgraph(g, subg, node_mask_values, edge_values, labels):
        """Analyze properties of important regions in the graph"""
        analysis = {
            'num_subgraphs': nx.number_connected_components(subg),
            'total_nodes': len(subg),
            'total_edges': len(subg.edges()),
            'avg_node_importance': np.mean([node_mask_values[n] for n in subg.nodes()]),
            'avg_edge_importance': np.mean([edge_values[list(g.edges()).index((u, v))] 
                                         for u, v in subg.edges()]) if len(subg.edges()) > 0 else 0,
            'components': []
        }
        
        # Analyze each connected component
        for i, component in enumerate(nx.connected_components(subg)):
            comp_subg = subg.subgraph(component).copy()
            
            # Calculate various centrality measures
            degree_cent = nx.degree_centrality(comp_subg)
            between_cent = nx.betweenness_centrality(comp_subg)
            close_cent = nx.closeness_centrality(comp_subg)
            
            # Find most central nodes by different measures
            central_nodes = {
                'degree': max(degree_cent.items(), key=lambda x: x[1])[0],
                'betweenness': max(between_cent.items(), key=lambda x: x[1])[0],
                'closeness': max(close_cent.items(), key=lambda x: x[1])[0]
            }
            
            comp_analysis = {
                'size': len(comp_subg),
                'edges': len(comp_subg.edges()),
                'avg_node_importance': np.mean([node_mask_values[n] for n in comp_subg.nodes()]),
                'density': nx.density(comp_subg),
                'diameter': nx.diameter(comp_subg) if nx.is_connected(comp_subg) else 0,
                'central_nodes': central_nodes,
                'node_types': Counter([labels[n] for n in comp_subg.nodes()]),
                'avg_clustering': nx.average_clustering(comp_subg)
            }
            analysis['components'].append(comp_analysis)
            
        return analysis

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

    # Get node labels
    labels = set_labels(data.x)
    node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

    # Set up color normalization and maps
    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)
    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    node_colors = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors = [edge_cmap(edge_norm(weight)) for weight in edge_values]

    # Calculate positions for the full graph
    pos = nx.spring_layout(g, k=1.5, iterations=50)

    # Draw and save the original full graph without colors
    plt.figure(figsize=(8, 6))
    nx.draw(g, pos, labels=labels, with_labels=True, node_size=150, width=2)
    plt.savefig(word1, dpi=300, bbox_inches='tight')
    plt.close()

    # Define thresholds for importance
    NODE_THRESHOLD = 0.5
    EDGE_THRESHOLD = 0.5

    # Create a new graph for important regions
    important_graph = nx.Graph()
    
    # Add edges where both nodes and the edge are important
    for i, (u, v) in enumerate(g.edges()):
        edge_importance = g[u][v]['weight']
        node_u_importance = node_mask_values[u]
        node_v_importance = node_mask_values[v]
        
        if (edge_importance >= EDGE_THRESHOLD and 
            node_u_importance >= NODE_THRESHOLD and 
            node_v_importance >= NODE_THRESHOLD):
            important_graph.add_edge(u, v)
            important_graph.nodes[u]['value'] = node_mask_values[u]
            important_graph.nodes[v]['value'] = node_mask_values[v]
            important_graph[u][v]['weight'] = edge_importance

    # If no important regions found, return
    if len(important_graph) == 0:
        print(f"No important regions found above threshold for graph {word}")
        return None

    # Perform analysis
    analysis_results = analyze_subgraph(g, important_graph, node_mask_values, edge_values, labels)

    # Create enhanced visualization
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1])

    # Main graph area
    ax_main = plt.subplot(gs[0, :])
    
    # Use positions from original graph for consistent layout
    important_pos = {node: pos[node] for node in important_graph.nodes()}

    # Add glow effect
    for node in important_graph.nodes():
        circle = plt.Circle(important_pos[node], 
                          radius=0.02, 
                          color='white', 
                          alpha=0.3,
                          zorder=1)
        ax_main.add_patch(circle)

    # Calculate edge widths based on importance
    edge_widths = [3 + 2 * important_graph[u][v]['weight'] for u, v in important_graph.edges()]

    # Draw the important subgraphs
    nx.draw(important_graph,
            important_pos,
            labels={node: labels[node] for node in important_graph.nodes()},
            with_labels=True,
            node_color=[node_colors[node] for node in important_graph.nodes()],
            edge_color=[edge_colors[list(g.edges()).index((u, v))] 
                       for u, v in important_graph.edges()],
            node_size=[100 + 100 * node_mask_values[node] for node in important_graph.nodes()],
            font_color='black',
            font_size=8,
            font_weight='bold',
            ax=ax_main,
            width=edge_widths,
            edgecolors='white',
            linewidths=1)

    # Add colorbars
    sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    plt.colorbar(sm_nodes, ax=ax_main, label='Node Importance', orientation='horizontal', pad=0.05)

    # Add analysis text
    ax_text = plt.subplot(gs[1, :])
    ax_text.axis('off')
    
    analysis_text = (
        f"Analysis of Important Regions:\n"
        f"Number of Subgraphs: {analysis_results['num_subgraphs']}\n"
        f"Total Important Nodes: {analysis_results['total_nodes']}\n"
        f"Average Node Importance: {analysis_results['avg_node_importance']:.3f}\n"
        f"Average Edge Importance: {analysis_results['avg_edge_importance']:.3f}\n\n"
    )
    
    for i, comp in enumerate(analysis_results['components']):
        analysis_text += (
            f"Component {i+1}:\n"
            f"  Size: {comp['size']} nodes, {comp['edges']} edges\n"
            f"  Density: {comp['density']:.3f}\n"
            f"  Avg Clustering: {comp['avg_clustering']:.3f}\n"
            f"  Most Central Atoms:\n"
            f"    By Degree: {labels[comp['central_nodes']['degree']]}\n"
            f"    By Betweenness: {labels[comp['central_nodes']['betweenness']]}\n"
            f"  Composition: {dict(comp['node_types'])}\n"
        )
    
    ax_text.text(0.05, 0.95, analysis_text, 
                transform=ax_text.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=8)

    plt.tight_layout()
    plt.savefig(word, dpi=300, bbox_inches='tight')
    plt.close()

    # Save analysis to a separate text file
    analysis_file = word.rsplit('.', 1)[0] + '_analysis.txt'
    with open(analysis_file, 'w') as f:
        f.write(analysis_text)
        
    return analysis_results

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

