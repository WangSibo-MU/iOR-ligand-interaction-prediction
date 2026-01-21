import numpy as np
from rdkit import Chem
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os
from torch_geometric.data import Data
import pickle

atom_dict = defaultdict(lambda: len(atom_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))

# ========== Compound graph processing ==========
def create_atoms(mol):
    atoms = []
    for a in mol.GetAtoms():
        atom_type = a.GetSymbol()  # Atomic Type
        degree = a.GetDegree()  # The degree of an atom
        formal_charge = a.GetFormalCharge()  # Atomic charge
        is_aromatic = a.GetIsAromatic()  # Aromatic atom
        hybridization = a.GetHybridization()  # Atomic hybridization state

        # Combine features into a tuple
        atom_features = (atom_type, degree, formal_charge, is_aromatic, hybridization)
        atoms.append(atom_features)

    # Convert atomic features into integer indices
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary of adjacent atoms and key types"""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = str(b.GetBondType())
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extracting subgraph fingerprints using Weisfeiler Lehman algorithm"""
    if len(atoms) == 1 or radius == 0:
        fingerprints = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jbond_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jbond_dict = _i_jedge_dict

    node_features = []
    for atom, fingerprint in zip(atoms, fingerprints):
        node_features.append([float(atom), float(fingerprint)])

    return np.array(node_features, dtype=np.float32)

def create_adjacency(mol, max_nodes=50):
    """Create an adjacency matrix for the molecular graph"""
    adjacency = Chem.GetAdjacencyMatrix(mol)
    if adjacency.shape[0] < max_nodes:
        adjacency = np.pad(adjacency, ((0, max_nodes - adjacency.shape[0]), (0, max_nodes - adjacency.shape[1])), mode='constant')
    else:
        adjacency = adjacency[:max_nodes, :max_nodes]
    return np.array(adjacency)

def smiles_to_graph(smiles, radius, max_nodes=50, max_edges=100):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None

    atoms = create_atoms(mol)
    i_jbond_dict = create_ijbonddict(mol)
    fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)

    if fingerprints.shape[0] < max_nodes:
        pad_width = ((0, max_nodes - fingerprints.shape[0]), (0, 0))
        fingerprints = np.pad(fingerprints, pad_width, mode='constant')
    else:
        fingerprints = fingerprints[:max_nodes]

    edge_index = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))

        bond_feature = [
            bond.GetBondTypeAsDouble(),
            bond.IsInRing(),
            bond.GetIsConjugated(),
            bond.GetStereo(),
        ]
        edge_features.append(bond_feature)
        edge_features.append(bond_feature)

    edge_index = np.array(edge_index, dtype=np.int64).T
    edge_features = np.array(edge_features, dtype=np.float32)

    if edge_index.shape[1] < max_edges:
        pad_width = ((0, 0), (0, max_edges - edge_index.shape[1]))
        edge_index = np.pad(edge_index, pad_width, mode='constant')
        edge_features = np.pad(edge_features, ((0, max_edges - edge_features.shape[0]), (0, 0)), mode='constant')
    else:
        edge_index = edge_index[:, :max_edges]
        edge_features = edge_features[:max_edges]

    x = torch.tensor(fingerprints, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    adjacency = create_adjacency(mol, max_nodes=max_nodes)
    adjacency = torch.tensor(adjacency, dtype=torch.float)

    x_with_adjacency = torch.cat([x, adjacency], dim=1)

    adjacency_edge = adjacency[edge_index[0], edge_index[1]].unsqueeze(1)
    edge_attr_with_adjacency = torch.cat([edge_attr, adjacency_edge], dim=1)

    data = Data(x=x_with_adjacency, edge_index=edge_index, edge_attr=edge_attr_with_adjacency)
    return data

# ========== Protein sequence processing ==========
def process_protein_sequence(protein, max_len=480):
    if len(protein) > max_len:
        return protein[:max_len]
    else:
        return protein.ljust(max_len)

# ========== Main process ==========
def process_compound_protein_data(file_path, radius=2, max_nodes=50, max_edges=100, 
                                 max_protein_len=480, test_size=0.2, random_state=42):

    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    compounds = []
    proteins = []
    labels = []
    original_smiles = []
    invalid_count = 0
    
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        smiles = parts[0]
        protein_seq = parts[1]
        label = int(parts[2])
        
        compound_graph = smiles_to_graph(smiles, radius, max_nodes, max_edges)
        
        if compound_graph is None:
            invalid_count += 1
            print(f"Skipping invalid SMILES at line {i+1}: {smiles}")
            continue
        
        processed_protein = process_protein_sequence(protein_seq, max_protein_len)
        
        compounds.append(compound_graph)
        proteins.append(processed_protein)
        labels.append(label)
        original_smiles.append(smiles)
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} samples...")
    
    print(f"\nData processing complete. Total samples: {len(compounds)}")
    print(f"Skipped {invalid_count} invalid SMILES")
    
    (compounds_train, compounds_test,
     proteins_train, proteins_test,
     smiles_train, smiles_test,
     labels_train, labels_test) = train_test_split(
        compounds, proteins, original_smiles, labels,
        test_size=test_size, random_state=random_state
    )
    
    print(f"Training set size: {len(compounds_train)}")
    print(f"Test set size: {len(compounds_test)}")
    
    os.makedirs('processed_data', exist_ok=True)
    
    all_protein_chars = set()
    for p in proteins:
        all_protein_chars.update(p)
    all_protein_chars.add(' ')
    
    with open('processed_data/atom_dict.pkl', 'wb') as f:
        pickle.dump(dict(atom_dict), f)
    with open('processed_data/fingerprint_dict.pkl', 'wb') as f:
        pickle.dump(dict(fingerprint_dict), f)
    with open('processed_data/edge_dict.pkl', 'wb') as f:
        pickle.dump(dict(edge_dict), f)
    with open('processed_data/protein_chars.pkl', 'wb') as f:
        pickle.dump(sorted(all_protein_chars), f)
    
    print("Saved feature dictionaries and protein character set")
    
    # Save processed data - Save training and testing sets separately
    torch.save(compounds_train, 'processed_data/train_compounds.pt')
    torch.save(compounds_test, 'processed_data/test_compounds.pt')
    
    np.savez('processed_data/train_data.npz',
             proteins=np.array(proteins_train),
             labels=np.array(labels_train))
    
    np.savez('processed_data/test_data.npz',
             proteins=np.array(proteins_test),
             labels=np.array(labels_test))
    
    with open('processed_data/train.txt', 'w') as f_train:
        for s, p, l in zip(smiles_train, proteins_train, labels_train):
            f_train.write(f"{s}\t{p}\t{l}\n")
    
    with open('processed_data/test.txt', 'w') as f_test:
        for s, p, l in zip(smiles_test, proteins_test, labels_test):
            f_test.write(f"{s}\t{p}\t{l}\n")
    
    metadata = {
        'num_train': len(compounds_train),
        'num_test': len(compounds_test),
        'max_nodes': max_nodes,
        'max_edges': max_edges,
        'max_protein_len': max_protein_len,
        'atom_dict_size': len(atom_dict),
        'fingerprint_dict_size': len(fingerprint_dict),
        'edge_dict_size': len(edge_dict),
        'protein_chars_size': len(all_protein_chars)
    }
    
    with open('processed_data/metadata.txt', 'w') as f_meta:
        for key, value in metadata.items():
            f_meta.write(f"{key}: {value}\n")
    
    print("\nOutput files saved in 'processed_data' directory:")
    print("- train_compounds.pt: Training compound graph structures")
    print("- test_compounds.pt: Test compound graph structures")
    print("- train_data.npz: Training protein sequences and labels")
    print("- test_data.npz: Test protein sequences and labels")
    print("- train.txt: Original format training data")
    print("- test.txt: Original format test data")
    print("- metadata.txt: Processing metadata")
    print("- atom_dict.pkl: Atom feature dictionary")
    print("- fingerprint_dict.pkl: Fingerprint dictionary")
    print("- edge_dict.pkl: Edge feature dictionary")
    print("- protein_chars.pkl: Protein character set")
    
    return {
        'compounds_train': compounds_train,
        'compounds_test': compounds_test,
        'proteins_train': proteins_train,
        'proteins_test': proteins_test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'metadata': metadata
    }

if __name__ == "__main__":
    input_file = 'dataset.txt'  # Replace with your data file path
    radius = 2                    # Weisfeiler-Lehman algorithm radius
    max_nodes = 50                # Maximum number of nodes
    max_edges = 100               # Maximum number of edges
    max_protein_len = 480         # Maximum length of protein sequence
    test_size = 0.2               # Test set ratio
    
    processed_data = process_compound_protein_data(
        input_file,
        radius=radius,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_protein_len=max_protein_len,
        test_size=test_size
    )
    
    print("\nData processing completed successfully!")