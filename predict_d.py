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
    # Create the original graph
    edges = [[data.edge_index[0][i].item(), data.edge_index[1][i].item()] for i in range(len(data.edge_index[0]))]
    g = nx.Graph(edges)

    # Extract node_mask and calculate the average
    node_mask = explanation.node_mask.squeeze().tolist()
    print(f"Checking Node_mask: {node_mask}")  # Display node_mask
    avg = np.mean(node_mask)
    print(f"Average of Node_mask: {avg}")  # Display average
	

    # Set node attributes for original graph
    nx.set_node_attributes(g, {i: value for i, value in enumerate(node_mask)}, 'value')

    # Assign edge weights
    edge_mask = explanation.edge_mask.numpy()
    edge_avg = np.mean(edge_mask)

    for i, (u, v) in enumerate(g.edges()):
        g[u][v]['weight'] = edge_values[i]

    # Assign labels to nodes
    labels = set_labels(data.x)

    # 1. Draw the original graph
    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)

    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    node_colors_original = [node_cmap(node_norm(value)) for value in node_mask]
    edge_colors_original = [edge_cmap(edge_norm(edge_values[i])) for i, (u, v) in enumerate(g.edges())]

    pos = nx.spring_layout(g)  # Use the same layout for all graphs

    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_original, edge_color=edge_colors_original,
            node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"original_{word}", dpi=300)
    plt.close()

    # 2. Draw the explainable predicted graph
    node_colors_predicted = [node_cmap(node_norm(value)) for value in node_mask]
    edge_colors_predicted = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g.edges()]

    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_predicted, edge_color=edge_colors_predicted,
            node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"predicted_{word}", dpi=300)
    plt.close()

    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_predicted, edge_color=edge_colors_predicted,
            node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"post_predicted_{word}", dpi=300)
    plt.close()

	# 3. Draw the re-explainable predicted graph
    	
    significant_nodes = [i for i, value in enumerate(node_mask) if value >= avg ]
    significant_edges = [i for i, weight in enumerate(edge_mask) if weight >= edge_avg ]

    g = g.subgraph(significant_nodes).copy()  # Keep only significant nodes
	
	# Reassign edge weights for significant edges
    edge_values = {tuple(edges[i]): edge_mask[i] for i in significant_edges if edges[i][0] in g and edges[i][1] in g}
    nx.set_edge_attributes(g, edge_values, 'weight')
	
	# Set node attributes
    node_mask_values = {i: node_mask[i] for i in significant_nodes}
    nx.set_node_attributes(g, node_mask_values, 'value')
	
    labels = set_labels(data.x)
    node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)

    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    node_colors_pp = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors_pp = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g.edges()]

    pos = nx.spring_layout(g)

    fig, ax = plt.subplots()
    nx.draw(g, pos, labels={node: labels[node] for node in significant_nodes}, with_labels=True, 
            node_color=node_colors_pp, edge_color=edge_colors_pp, node_size=150, font_color='black', ax=ax, width=3)

    plt.savefig(word1, dpi=300)
    plt.close()



# def draw_graph(data, explanation, word, word1):
#     edges = []
#     for i in range(len(data.edge_index[0])):
#         edges.append([data.edge_index[0][i].item(), data.edge_index[1][i].item()])
#     g = nx.Graph(edges)

#     node_mask = explanation.node_mask.squeeze().tolist()
#     edge_mask = explanation.edge_mask.numpy()

        
#     # Filter nodes and edges with values above the average
#     node_avg = np.mean(node_mask)
#     edge_avg = np.mean(edge_mask)

#     significant_nodes = [i for i, value in enumerate(node_mask) if value > node_avg ]
#     significant_edges = [i for i, weight in enumerate(edge_mask) if weight > edge_avg ]

#     g = g.subgraph(significant_nodes).copy()  # Keep only significant nodes

#     # Reassign edge weights for significant edges
#     edge_values = {tuple(edges[i]): edge_mask[i] for i in significant_edges if edges[i][0] in g and edges[i][1] in g}
#     nx.set_edge_attributes(g, edge_values, 'weight')

#     # Set node attributes
#     node_mask_values = {i: node_mask[i] for i in significant_nodes}
#     nx.set_node_attributes(g, node_mask_values, 'value')

#     labels = set_labels(data.x)
#     node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

#     node_norm = plt.Normalize(vmin=0, vmax=1)
#     edge_norm = plt.Normalize(vmin=0, vmax=1)

#     node_cmap = plt.cm.Blues
#     edge_cmap = plt.cm.Reds

#     node_colors = [node_cmap(node_norm(value)) for value in node_values]
#     edge_colors = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g.edges()]

#     pos = nx.spring_layout(g)

#     fig, ax = plt.subplots()
#     nx.draw(g, pos, labels={node: labels[node] for node in significant_nodes}, with_labels=True, 
#             node_color=node_colors, edge_color=edge_colors, node_size=150, font_color='black', ax=ax, width=3)

#     plt.savefig(word, dpi=300)
#     plt.close()

# def draw_graph(data, explanation, word, word1):
#     # Create the original graph
#     edges = [[data.edge_index[0][i].item(), data.edge_index[1][i].item()] for i in range(len(data.edge_index[0]))]
#     g = nx.Graph(edges)

#     # Extract node_mask and calculate the average
#     node_mask = explanation.node_mask.squeeze().tolist()
#     print(f"Checking Node_mask: {node_mask}")  # Display node_mask
#     avg = np.median(node_mask)
#     print(f"Average of Node_mask: {avg}")  # Display average

#     # Set node attributes for original graph
#     nx.set_node_attributes(g, {i: value for i, value in enumerate(node_mask)}, 'value')

#     # Assign edge weights
#     edge_values = explanation.edge_mask.numpy()
#     for i, (u, v) in enumerate(g.edges()):
#         g[u][v]['weight'] = edge_values[i]

#     # Assign labels to nodes
#     labels = set_labels(data.x)

#     # 1. Draw the original graph
#     node_norm = plt.Normalize(vmin=0, vmax=1)
#     edge_norm = plt.Normalize(vmin=0, vmax=1)

#     node_cmap = plt.cm.Blues
#     edge_cmap = plt.cm.Reds

#     node_colors_original = [node_cmap(node_norm(value)) for value in node_mask]
#     edge_colors_original = [edge_cmap(edge_norm(edge_values[i])) for i, (u, v) in enumerate(g.edges())]

#     pos = nx.spring_layout(g)  # Use the same layout for all graphs

#     fig, ax = plt.subplots()
#     nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_original, edge_color=edge_colors_original,
#             node_size=150, font_color='black', ax=ax, width=3)
#     plt.savefig(f"original_{word}", dpi=300)
#     plt.close()

#     # 2. Draw the explainable predicted graph
#     node_colors_predicted = [node_cmap(node_norm(value)) for value in node_mask]
#     edge_colors_predicted = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g.edges()]

#     fig, ax = plt.subplots()
#     nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_predicted, edge_color=edge_colors_predicted,
#             node_size=150, font_color='black', ax=ax, width=3)
#     plt.savefig(f"predicted_{word}", dpi=300)
#     plt.close()

#     fig, ax = plt.subplots()
#     nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_predicted, edge_color=edge_colors_predicted,
#             node_size=150, font_color='black', ax=ax, width=3)
#     plt.savefig(f"post_predicted_{word}", dpi=300)
#     plt.close()


##############
    # edge_mask = explanation.edge_mask.numpy()
    # # Filter nodes and edges with values above the average
    # node_avg = np.mean(node_mask)
    # edge_avg = np.mean(edge_mask)

    # significant_nodes = [i for i, value in enumerate(node_mask) if value >= node_avg ]
    # significant_edges = [i for i, weight in enumerate(edge_mask) if weight >= edge_avg ]

    # g = g.subgraph(significant_nodes).copy()  # Keep only significant nodes

    # # Reassign edge weights for significant edges
    # edge_values = {tuple(edges[i]): edge_mask[i] for i in significant_edges if edges[i][0] in g and edges[i][1] in g}
    # nx.set_edge_attributes(g, edge_values, 'weight')

    # # Set node attributes
    # node_mask_values = {i: node_mask[i] for i in significant_nodes}
    # nx.set_node_attributes(g, node_mask_values, 'value')

    # labels = set_labels(data.x)
    # node_values = np.array([data['value'] for _, data in g.nodes(data=True)])

    # node_norm = plt.Normalize(vmin=0, vmax=1)
    # edge_norm = plt.Normalize(vmin=0, vmax=1)

    # node_cmap = plt.cm.Blues
    # edge_cmap = plt.cm.Reds

    # node_colors = [node_cmap(node_norm(value)) for value in node_values]
    # edge_colors = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g.edges()]

    # pos = nx.spring_layout(g)

    # fig, ax = plt.subplots()
    # nx.draw(g, pos, labels={node: labels[node] for node in significant_nodes}, with_labels=True, 
    #         node_color=node_colors, edge_color=edge_colors, node_size=150, font_color='black', ax=ax, width=3)

    # plt.savefig(word, dpi=300)
    # plt.close()

    # fig, ax = plt.subplots()
    # nx.draw(g, pos, labels={node: labels[node] for node in significant_nodes}, with_labels=True, 
    #         node_color=node_colors, edge_color=edge_colors, node_size=150, font_color='black', ax=ax, width=3)

    # plt.savefig(word1, dpi=300)
    # plt.close()

    """# 3. Draw the post-prediction graph (filtered by average)
    # Filter nodes and edges
    filtered_nodes = [node for node, value in enumerate(node_mask) if value >= avg]
    filtered_edges = [(u, v) for u, v in g.edges() if u in filtered_nodes and v in filtered_nodes]

    # Create filtered subgraph
    g_filtered = nx.Graph()
    g_filtered.add_nodes_from(filtered_nodes)
    g_filtered.add_edges_from(filtered_edges)

    # Update node values for the filtered subgraph
    filtered_node_values = [node_mask[node] for node in g_filtered.nodes()]
    filtered_node_colors = [node_cmap(node_norm(value)) for value in filtered_node_values]
    filtered_edge_colors = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in g_filtered.edges()]

    # Draw the filtered graph
    fig, ax = plt.subplots()
    nx.draw(g_filtered, pos, labels=labels, with_labels=True, node_color=filtered_node_colors,
            edge_color=filtered_edge_colors, node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"post_predicted_{word1}", dpi=300)
    plt.close()
"""
    
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
        if(i>=10):
            break
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
        word_p = "graphpp"+str(i) + ".png"
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

