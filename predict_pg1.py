import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from gen import GEN
from models.hgc_gcn import HGC_GCN
from utils import *
from torch_geometric.explain import Explainer, PGExplainer, ModelConfig
import networkx as nx
import matplotlib.pyplot as plt
import sys
from torch_geometric.data import Data

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
        
def draw_graph(data, explanation, original_filename, explainable_filename): #, filtered_filename):
    # Construct the graph from edge_index
    edges = []
    for i in range(len(data.edge_index[0])):
        edges.append([data.edge_index[0][i].item(), data.edge_index[1][i].item()])
    g = nx.Graph(edges)

    # Extract node_mask and edge_mask values
    node_mask = explanation.node_mask.squeeze().tolist()
    edge_mask = explanation.edge_mask.numpy()

    print(f"Node Mask: {node_mask}")
    print(f"Edge Mask: {edge_mask}")

    # # Calculate average values
    # node_avg = np.median(node_mask)
    # edge_avg = np.median(edge_mask)
    # print(f"Node Mask Average: {node_avg}")
    # print(f"Edge Mask Average: {edge_avg}")

    # Assign node mask values as node attributes
    nx.set_node_attributes(g, {i: value for i, value in enumerate(node_mask)}, 'value')

    # Assign edge mask values as edge weights
    for i, (u, v) in enumerate(g.edges()):
        g[u][v]['weight'] = edge_mask[i]

    # Assign labels to nodes
    labels = set_labels(data.x)

    #g = g.subgraph(node_mask).copy()  # Keep only significant nodes

#     # Reassign edge weights for significant edges
    edge_values = {tuple(edges[i]): edge_mask[i] for i in edge_mask if edges[i][0] in g and edges[i][1] in g}
    nx.set_edge_attributes(g, edge_values, 'weight')

    # # Filter nodes and edges for the explainable subgraph
    # filtered_nodes = [node for node, value in enumerate(node_mask) if value >= node_avg]
    # filtered_edges = [(u, v) for u, v in g.edges() if g[u][v]['weight'] >= edge_avg]

    #     # Create filtered subgraph
    # g_filtered = nx.Graph()
    # g_filtered.add_nodes_from(filtered_nodes)
    # g_filtered.add_edges_from(filtered_edges)

    # Prepare color maps for visualization
    node_norm = plt.Normalize(vmin=0, vmax=1)
    edge_norm = plt.Normalize(vmin=0, vmax=1)

    node_cmap = plt.cm.Blues
    edge_cmap = plt.cm.Reds

    # # Node and edge colors for filtered subgraph
    # node_colors_filtered = [node_cmap(node_norm(node_mask[node])) for node in filtered_nodes]
    # edge_colors_filtered = [edge_cmap(edge_norm(g[u][v]['weight'])) for u, v in filtered_edges]

    node_colors_original = [node_cmap(node_norm(value)) for value in node_mask]
    edge_colors_original = [edge_cmap(edge_norm('weight')) for u, v in edge_values()]

    # Use the same layout for all graphs
    pos = nx.spring_layout(g)

    # 1. Draw the actual graph
    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_size=150, font_color='black', ax=ax)
    plt.savefig(original_filename, dpi=300)
    plt.close()

    # 2. Draw the explainable subgraph
    fig, ax = plt.subplots()
    nx.draw(g, pos, labels=labels, with_labels=True, node_color=node_colors_original,
            edge_color=edge_colors_original, node_size=150, font_color='black', ax=ax, width=3)
    plt.savefig(explainable_filename, dpi=300)
    plt.close()

    # # 3. Draw the filtered explainable subgraph
    # fig, ax = plt.subplots()
    # nx.draw(g_filtered, pos, labels=labels, with_labels=True, node_color=node_colors_filtered,
    #         edge_color=edge_colors_filtered, node_size=150, font_color='black', ax=ax, width=3)
    # plt.savefig(filtered_filename, dpi=300)
    # plt.close()

    if len(g.nodes) == 0 or len(g.edges) == 0:
        print(f"No nodes or edges above average for graph {explainable_filename}")
    return

    
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    #loader = DataLoader(dataset, batch_size=1, shuffle=True)
    #####################################################
    dataset = Planetoid(path, dataset, transform=transform)
    train_data, val_data, test_data = dataset[0]

    # Explain model output for a single edge:
    edge_label_index = val_data.edge_label_index[:, 0]

    explanation = explainer(
        x=data.x,
        edge_index=data.edge_index,
        edge_label_index=edge_label_index,
    )
    print(f'Generated model explanations in {explanation.available_explanations}')

    # Explain a selected target (phenomenon) for a single edge:
    edge_label_index = val_data.edge_label_index[:, 0]
    target = val_data.edge_label[0].unsqueeze(dim=0).long()

    explanation = explainer(
    x=train_data.x,
    edge_index=train_data.edge_index,
    target=target,
    edge_label_index=edge_label_index,
    )
    available_explanations = explanation.available_explanations
    print(f'Generated phenomenon explanations in {available_explanations}')
    ############################################################################
    # Initialize the explainer
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=30,lr=0.003), #**kwargs
        explanation_type='model',  #phenomenon
        #node_mask_type='attributes', #object
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw'
        ),
        #Include only the top 10 most important edges:
        threshold_config=dict(threshold_type='topk',value=10),
    )

    for epoch in range(30):
        for batch in loader:
            loss=explainer.algorithm.train(
                epoch, model, batch.x, batch.edge_index,target=batch.target,index=None)
        
    print(f"Making predictions for {len(loader.dataset)} samples...")
    for i, data in enumerate(loader):
        if i >= 20:  # Limit the range of displayed graphs (adjust as needed)
            break
        #i+=1
        data = data.to(device)
        output = model(data.x, data.edge_index, data,index=None)

        total_preds = torch.cat((total_preds, output.cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

        # Generate explanation for the current graph
        explanation = explainer(
            data[0].x,
            data[0].edge_index,data[0],index=0)
        #print(explanation.edge_mask)
        
        # Generate file names for graph images
        original_filename = f"orig_graph_{i + 1}.png"
        explainable_filename = f"expl_graph_{i + 1}.png"
        #filtered_filename = f"filt_graph_{i + 1}.png"

        try:
            # Draw and save the graphs
            draw_graph(data, explanation, original_filename, explainable_filename) #, filtered_filename)
        except Exception as e:
            print(f"Error drawing graph {i + 1}: {e}")
            continue

    return total_labels.numpy().flatten(), total_preds.detach().numpy().flatten()



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
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=True)  #shuffle=False -> modified
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

