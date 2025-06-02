import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import rdDepictor
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import io
import base64

class DrugSubgraphVisualizer:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        
    def extract_and_visualize_subgraphs(self, mol, explained_subgraph):
        """Extract and visualize subgraphs from explained regions"""
        # Convert importance scores to binary mask
        node_mask = (explained_subgraph.node_mask > self.threshold).numpy()
        edge_mask = (explained_subgraph.edge_mask > self.threshold).numpy()
        
        # Extract subgraphs
        subgraphs = self.extract_subgraphs(mol, node_mask, edge_mask)
        
        # Generate visualizations
        visualizations = self.visualize_subgraphs(mol, subgraphs)
        
        return subgraphs, visualizations
    
    def extract_subgraphs(self, mol, node_mask, edge_mask):
        """Extract significant subgraphs based on masks"""
        subgraphs = []
        visited = set()
        
        for atom_idx in range(mol.GetNumAtoms()):
            if node_mask[atom_idx] and atom_idx not in visited:
                # Find connected component
                subgraph = self.find_connected_subgraph(
                    mol, atom_idx, node_mask, edge_mask
                )
                
                if len(subgraph['atoms']) > 0:
                    subgraphs.append(subgraph)
                    visited.update(subgraph['atoms'])
        
        return subgraphs
    
    def find_connected_subgraph(self, mol, start_atom, node_mask, edge_mask):
        """Find connected subgraph starting from given atom"""
        subgraph = {'atoms': set(), 'bonds': set()}
        queue = [start_atom]
        
        while queue:
            current_atom = queue.pop(0)
            subgraph['atoms'].add(current_atom)
            
            # Check neighboring atoms through bonds
            for bond in mol.GetAtomWithIdx(current_atom).GetBonds():
                begin_atom = bond.GetBeginAtomIdx()
                end_atom = bond.GetEndAtomIdx()
                other_atom = end_atom if begin_atom == current_atom else begin_atom
                
                if (node_mask[other_atom] and 
                    edge_mask[bond.GetIdx()] and 
                    other_atom not in subgraph['atoms']):
                    queue.append(other_atom)
                    subgraph['bonds'].add(bond.GetIdx())
        
        return subgraph
    
    def create_subgraph_mol(self, mol, subgraph):
        """Create RDKit molecule from subgraph"""
        if not subgraph['atoms']:
            return None
            
        # Create empty editable mol object
        em = Chem.EditableMol(Chem.Mol())
        
        # Add atoms
        old_idx_to_new = {}
        for old_idx in subgraph['atoms']:
            atom = mol.GetAtomWithIdx(old_idx)
            new_idx = em.AddAtom(atom)
            old_idx_to_new[old_idx] = new_idx
        
        # Add bonds
        for bond_idx in subgraph['bonds']:
            bond = mol.GetBondWithIdx(bond_idx)
            begin_atom = old_idx_to_new[bond.GetBeginAtomIdx()]
            end_atom = old_idx_to_new[bond.GetEndAtomIdx()]
            em.AddBond(begin_atom, end_atom, bond.GetBondType())
        
        # Convert to mol object
        subgraph_mol = em.GetMol()
        
        # Clean up the molecule
        Chem.SanitizeMol(subgraph_mol)
        
        return subgraph_mol
    
    def visualize_subgraphs(self, mol, subgraphs):
        """Generate visualizations for subgraphs"""
        visualizations = []
        
        # Prepare molecule for visualization
        AllChem.Compute2DCoords(mol)
        
        # Original molecule visualization
        mol_img = Draw.MolToImage(mol)
        visualizations.append(('Original Molecule', mol_img))
        
        # Visualize each subgraph
        for i, subgraph in enumerate(subgraphs):
            subgraph_mol = self.create_subgraph_mol(mol, subgraph)
            if subgraph_mol is not None:
                # Generate 2D coordinates for subgraph
                AllChem.Compute2DCoords(subgraph_mol)
                
                # Draw subgraph
                img = Draw.MolToImage(subgraph_mol)
                visualizations.append((f'Subgraph {i+1}', img))
        
        return visualizations
    
    def analyze_subgraph_features(self, mol, subgraph):
        """Analyze chemical features of subgraph"""
        subgraph_mol = self.create_subgraph_mol(mol, subgraph)
        if subgraph_mol is None:
            return None
            
        features = {
            'num_atoms': len(subgraph['atoms']),
            'num_bonds': len(subgraph['bonds']),
            'molecular_weight': Chem.Descriptors.ExactMolWt(subgraph_mol),
            'rings': subgraph_mol.GetRingInfo().NumRings(),
            'smiles': Chem.MolToSmiles(subgraph_mol),
            'formula': Chem.rdMolDescriptors.CalcMolFormula(subgraph_mol)
        }
        
        return features

# Example usage
def main():
    # Create sample drug molecule
    smiles = "CC(=O)NC1=CC=C(C=C1)O"  # Acetaminophen
    mol = Chem.MolFromSmiles(smiles)
    
    # Initialize visualizer
    visualizer = DrugSubgraphVisualizer()
    
    # Create dummy explained subgraph (replace with actual GNN explainer output)
    explained_subgraph = type('DummyExplanation', (), {
        'node_mask': torch.rand(mol.GetNumAtoms()),
        'edge_mask': torch.rand(mol.GetNumBonds())
    })
    
    # Extract and visualize subgraphs
    subgraphs, visualizations = visualizer.extract_and_visualize_subgraphs(
        mol, explained_subgraph
    )
    
    # Analyze features for each subgraph
    analyses = []
    for subgraph in subgraphs:
        analysis = visualizer.analyze_subgraph_features(mol, subgraph)
        analyses.append(analysis)
    
    return subgraphs, visualizations, analyses

# Create molecule
smiles = "CC(=O)NC1=CC=C(C=C1)O"
mol = Chem.MolFromSmiles(smiles)

# Get explained subgraph from your GNN explainer
# Create an instance of GNNExplainer with your model
gnn_explainer = GNNExplainer(
    model=GNNExplainer,  # Your GNN model
    # Additional parameters as needed:
    epochs=100,
    lr=0.01,
    num_hops=3,
)
explained_subgraph = gnn_explainer.explain(mol)


# Initialize visualizer
visualizer = DrugSubgraphVisualizer()

# Extract and visualize subgraphs
subgraphs, visualizations = visualizer.extract_and_visualize_subgraphs(
    mol, explained_subgraph
)

# Analyze subgraph features
analyses = []
for subgraph in subgraphs:
    analysis = visualizer.analyze_subgraph_features(mol, subgraph)
    analyses.append(analysis)