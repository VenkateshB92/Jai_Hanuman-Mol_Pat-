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
import numpy

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
    edges = []
    for i in range(len(data.edge_index[0])):
        edges.append([data.edge_index[0][i].item(), data.edge_index[1][i].item()])
    g = nx.Graph(edges)

    # Extract node_mask and calculate the average
    node_mask = explanation.node_mask.squeeze().tolist()
    print(f"Checking Node_mask: {node_mask}")  # Display node_mask
    avg = numpy.average(node_mask)
    print(f"Average of Node_mask: {avg}")  # Display average

    # Filter node_mask values: Keep only those >= average, set others to None (skip drawing)
    filtered_node_mask_values = {i: (value if value >= avg else None) for i, value in enumerate(node_mask)}
    print("Filtered Node Mask Values:", filtered_node_mask_values)

    # Set node attributes for the graph (skip those below average)
    nx.set_node_attributes(g, {i: value for i, value in enumerate(node_mask)}, 'value')

    # Set edge weights
    edge_values = explanation.edge_mask.numpy()
    for i, (u, v) in enumerate(g.edges()):
        g[u][v]['weight'] = edge_values[i]

    print(f"Checking edge_mask: {edge_values}")  # Display edge_mask
    avged = numpy.average(edge_values)
    print(f"Average of edge_mask: {avged}")  # Display average
    # Assign labels to nodes
    labels = set_labels(data.x)

    # Filter out nodes with values below average for the predicted graph
    filtered_nodes = [node for node, value in filtered_node_mask_values.items() if value is not None]

    # Create a new subgraph with only the filtered nodes and edges
    g_filtered = g.subgraph(filtered_nodes).copy()

    # Update node values for the filtered subgraph
    node_values = [filtered_node_mask_values[node] for node in g_filtered.nodes()]

    # Normalize node and edge values for coloring
    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)

    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    # Node and edge colors for predicted graph
    node_colors_filtered = [node_cmap(node_norm(value)) for value in node_values]
    edge_colors_filtered = [edge_cmap(edge_norm(g_filtered[u][v]['weight'])) for u, v in g_filtered.edges()]

    # Node and edge colors for the original graph
    node_colors_original = [node_cmap(node_norm(value)) for value in node_mask]
    edge_colors_original = [edge_cmap(edge_norm(edge_values[i])) for i, (u, v) in enumerate(g.edges())]

    pos = nx.spring_layout(g)  # Use the same layout for all graphs

    # 1. Draw the original graph
    fig, ax = plt.subplots()
    nx.draw(g_filtered, pos, labels=labels, with_labels=True, node_color=node_colors_original, edge_color=edge_colors_original,
            node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"GT_{word}", dpi=300)
    plt.close()

    # 2. Draw the predicted graph (highlighted nodes and edges)
    fig, ax = plt.subplots()
    nx.draw(g_filtered, pos, labels=labels, with_labels=True, node_color=node_colors_filtered,
            edge_color=edge_colors_filtered, node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(f"pred_{word}", dpi=300)
    plt.close()

    # 3. Draw the post-prediction graph (comparison)
    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_size=150, ax=ax, width=2)
    plt.savefig(f"post_pred_{word1}", dpi=300)
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
        if i<=2:
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
                #print("Explanation node masks",explanation.node_mask)
                #print("Explanation edge masks",explanation.edge_mask)

            word = "graph_p" + str(i) + ".png"
            word1 = "oldgraph_p" + str(i) + ".png"

            try:
                draw_graph(data, explanation, word, word1)
            except:  #Exception as e
                #print(f"Error drawing graph: {e}")
                continue
            # Compute the average node mask value
            # node_mask = explanation.node_mask.squeeze().tolist()
            # avg_node_mask = np.mean(node_mask)
            # print(f"Graph {i}: Average Node Mask = {avg_node_mask}")

            # # Handle below-average graphs
            # if any(value < avg_node_mask for value in node_mask):
            #     word = f"graph_below_avg_{i}.png"
            #     word1 = f"oldgraph_below_avg_{i}.png"
            #     try:
            #         draw_graph(data, explanation, word, word1)
            #     except:  # Exception as e
            #         continue

            # # Handle above-average graphs
            # if any(value >= avg_node_mask for value in node_mask):
            #     word = f"graph_above_avg_{i}.png"
            #     word1 = f"oldgraph_above_avg_{i}.png"
            #     try:
            #         draw_graph(data, explanation, word, word1)
            #     except:  # Exception as e
            #         continue
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

