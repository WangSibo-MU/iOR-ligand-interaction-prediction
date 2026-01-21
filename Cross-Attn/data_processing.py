import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

def process_data(file_path, max_protein_len=480, max_smiles_len=70, test_size=0.2, random_state=42):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    smiles_list = []
    protein_list = []
    labels = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            smiles = parts[0]
            protein = parts[1]
            label = int(parts[2])
            
            smiles_list.append(smiles)
            protein_list.append(protein)
            labels.append(label)

    actual_max_smiles = max(len(s) for s in smiles_list)
    actual_max_protein = max(len(p) for p in protein_list)
    
    print(f"Actual maximum SMILES length: {actual_max_smiles}")
    print(f"Actual maximum protein length: {actual_max_protein}")
    print(f"Using SMILES length: {max_smiles_len}")
    print(f"Using protein length: {max_protein_len}")

    processed_smiles = []
    for smile in smiles_list:
        if len(smile) > max_smiles_len:
            processed = smile[:max_smiles_len]
        else:
            processed = smile.ljust(max_smiles_len)
        processed_smiles.append(processed)

    processed_proteins = []
    for protein in protein_list:
        if len(protein) > max_protein_len:
            processed = protein[:max_protein_len]
        else:
            processed = protein.ljust(max_protein_len)
        processed_proteins.append(processed)

    smiles_array = np.array(processed_smiles)
    protein_array = np.array(processed_proteins)
    labels_array = np.array(labels)

    (smiles_train, smiles_test, 
     protein_train, protein_test, 
     labels_train, labels_test) = train_test_split(
        smiles_array, protein_array, labels_array, 
        test_size=test_size, random_state=random_state
    )

    os.makedirs('processed_data', exist_ok=True)

    np.savez('processed_data/train.npz', 
             smiles=smiles_train, 
             proteins=protein_train, 
             labels=labels_train)
    
    np.savez('processed_data/test.npz', 
             smiles=smiles_test, 
             proteins=protein_test, 
             labels=labels_test)

    with open('processed_data/train.txt', 'w') as f_train:
        for s, p, l in zip(smiles_train, protein_train, labels_train):
            f_train.write(f"{s.strip()} {p.strip()} {l}\n")
    
    with open('processed_data/test.txt', 'w') as f_test:
        for s, p, l in zip(smiles_test, protein_test, labels_test):
            f_test.write(f"{s.strip()} {p.strip()} {l}\n")
    
    print(f"Data processing completed, saved to processed_data directory:")
    print(f"- Training set: {len(smiles_train)} samples")
    print(f"- Test set: {len(smiles_test)} samples")
    print(f"- Training set file: train.npz and train.txt")
    print(f"- Test set file: test.npz and test.txt")
    
    return {
        'train': {'smiles': smiles_train, 'proteins': protein_train, 'labels': labels_train},
        'test': {'smiles': smiles_test, 'proteins': protein_test, 'labels': labels_test},
        'actual_max_smiles': actual_max_smiles,
        'actual_max_protein': actual_max_protein
    }

if __name__ == "__main__":
    file_path = 'dataset.txt'
    processed_data = process_data(file_path)