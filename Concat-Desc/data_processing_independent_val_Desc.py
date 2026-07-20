import numpy as np
import pandas as pd
from rdkit import Chem
import os
import argparse
from rdkit.ML.Descriptors import MoleculeDescriptors
import json
import time

def load_selected_descriptor_names(metadata_path='descriptors/descriptor_metadata.json',
                                   names_path='descriptors/selected_descriptor_names_vif.txt'):
    """
    外部验证集必须复用训练阶段已经确定的去冗余描述符集合。
    不能在外部验证集上重新执行 variance/VIF 拟合，否则会造成特征空间与已训练模型不一致。
    """
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'selected_names' in metadata and len(metadata['selected_names']) > 0:
            selected_names = metadata['selected_names']
            print(f"  Loaded {len(selected_names)} selected descriptors from {metadata_path}")
            return selected_names

    if os.path.exists(names_path):
        with open(names_path, 'r') as f:
            selected_names = [name for name in f.read().strip().split(',') if name]

        if len(selected_names) > 0:
            print(f"  Loaded {len(selected_names)} selected descriptors from {names_path}")
            return selected_names

    raise FileNotFoundError(
        "Cannot find selected descriptor names. "
        "Please run the original training preprocessing first, so that "
        "'descriptors/descriptor_metadata.json' or "
        "'descriptors/selected_descriptor_names_vif.txt' exists."
    )

def compute_molecular_descriptors(smiles_list, descriptor_names):
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptors = []
    valid_indices = []
    failed_count = 0

    print(f"Starting calculation of selected descriptors for {len(smiles_list)} molecules...")
    for idx, smiles in enumerate(smiles_list):
        if idx % 500 == 0 and idx > 0:
            print(f"  processed {idx}/{len(smiles_list)} molecules...")

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                desc_values = calculator.CalcDescriptors(mol)
                descriptors.append(desc_values)
                valid_indices.append(idx)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"  [Error] Calculation failed: {smiles[:50]}... Error: {str(e)[:80]}")
                continue
        else:
            failed_count += 1
            if failed_count <= 5:
                print(f"  [Invalid] SMILES: {smiles[:50]}...")

    if len(descriptors) == 0:
        raise ValueError("No valid molecular descriptors were calculated. Please check the input SMILES.")

    descriptors = np.array(descriptors, dtype=np.float32)

    # 处理少量 RDKit 描述符可能产生的 nan / inf，避免 scaler.transform 或模型输入报错
    descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  ✓ Successed: {len(valid_indices)}, ✗ failed: {failed_count}")
    return descriptors, valid_indices

def process_protein_sequence(protein, max_len=480):
    if len(protein) > max_len:
        return protein[:max_len]
    else:
        return protein.ljust(max_len)

def process_external_validation_data(file_path,
                                     max_protein_len=480,
                                     random_state=42,
                                     balance_samples=False,
                                     metadata_path='descriptors/descriptor_metadata.json',
                                     names_path='descriptors/selected_descriptor_names_vif.txt',
                                     save_compat_test=True):
    print("\n" + "="*70)
    print("Independent external validation preprocessing pipeline started")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    start_time = time.time()

    print("\n[1/5] Loading raw external validation dataset...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    print(f"  Loaded {len(lines)} lines of data")

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

    if len(smiles_list) == 0:
        raise ValueError("No valid samples were found. Each line should contain: SMILES protein label")

    print(f"  Parsed valid rows: {len(smiles_list)} samples")

    actual_max_smiles = max(len(s) for s in smiles_list)
    actual_max_protein = max(len(p) for p in protein_list)

    print("\n[2/5] Loading training-stage de-redundant descriptor list...")
    selected_descriptor_names = load_selected_descriptor_names(
        metadata_path=metadata_path,
        names_path=names_path
    )

    print(f"\n  SMILES length - actual maximum: {actual_max_smiles}")
    print(f"  Protein length - actual maximum: {actual_max_protein}")
    print(f"  Protein length - setting used: {max_protein_len}")
    print(f"  Descriptor count used for external validation: {len(selected_descriptor_names)}")

    print("\n[3/5] Calculating selected molecular descriptors only...")
    selected_descriptors, valid_indices = compute_molecular_descriptors(
        smiles_list,
        selected_descriptor_names
    )

    smiles_arr = np.array([smiles_list[i] for i in valid_indices], dtype=object)
    protein_arr = np.array([protein_list[i] for i in valid_indices], dtype=object)
    labels_arr = np.array(labels, dtype=int)[valid_indices]
    descriptors_arr = selected_descriptors

    print(f"  {len(smiles_arr)} valid samples remain after invalid-SMILES filtering")

    print("\n[4/5] Building independent external validation set without train/test split...")
    if balance_samples:
        print("  → Applying dynamic downsampling balance on the whole external validation set (strict 1:1)")
        rng = np.random.default_rng(random_state)

        pos_idx = np.where(labels_arr == 1)[0]
        neg_idx = np.where(labels_arr == 0)[0]
        min_count = min(len(pos_idx), len(neg_idx))

        if min_count == 0:
            raise ValueError(
                "Cannot balance samples because at least one class has zero samples "
                f"(Pos: {len(pos_idx)}, Neg: {len(neg_idx)})."
            )

        pos_sampled = rng.choice(pos_idx, min_count, replace=False)
        neg_sampled = rng.choice(neg_idx, min_count, replace=False)
        balanced_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(balanced_idx)

        smiles_arr = smiles_arr[balanced_idx]
        protein_arr = protein_arr[balanced_idx]
        labels_arr = labels_arr[balanced_idx]
        descriptors_arr = descriptors_arr[balanced_idx]
    else:
        print("  → Balance disabled; using all valid external validation samples")

    processed_proteins = []
    for protein in protein_arr:
        processed_proteins.append(process_protein_sequence(str(protein), max_protein_len))

    print("\n[5/5] Saving independent external validation files...")
    os.makedirs('processed_data', exist_ok=True)
    os.makedirs('descriptors', exist_ok=True)

    # 保存外部验证集专用文件
    np.savez('processed_data/external_validation.npz',
             smiles=smiles_arr,
             proteins=np.array(processed_proteins),
             labels=labels_arr,
             descriptors=descriptors_arr)

    with open('processed_data/external_validation.txt', 'w') as f_ext:
        for s, p, l in zip(smiles_arr, processed_proteins, labels_arr):
            f_ext.write(f"{s} {p.strip()} {l}\n")

    # 保存描述符明细，便于核对与 SHAP 对齐
    ext_desc_df = pd.DataFrame(descriptors_arr, columns=selected_descriptor_names)
    ext_desc_df.insert(0, 'SMILES', smiles_arr)
    ext_desc_df.to_csv('descriptors/external_validation_descriptors_selected.csv', index=False)

    # 兼容原 validation_prediction.py / validation_SHAP.py：
    # 这两个脚本默认读取 processed_data/test.npz，因此可额外保存一份同内容 test.npz。
    if save_compat_test:
        np.savez('processed_data/test.npz',
                 smiles=smiles_arr,
                 proteins=np.array(processed_proteins),
                 labels=labels_arr,
                 descriptors=descriptors_arr)

        with open('processed_data/test.txt', 'w') as f_test:
            for s, p, l in zip(smiles_arr, processed_proteins, labels_arr):
                f_test.write(f"{s} {p.strip()} {l}\n")

    metadata = {
        'dataset_type': 'independent_external_validation',
        'num_external_validation': int(len(smiles_arr)),
        'positive_samples': int(np.sum(labels_arr == 1)),
        'negative_samples': int(np.sum(labels_arr == 0)),
        'max_protein_len': int(max_protein_len),
        'actual_max_smiles': int(actual_max_smiles),
        'actual_max_protein': int(actual_max_protein),
        'descriptor_count': int(len(selected_descriptor_names)),
        'descriptor_source_metadata_path': metadata_path,
        'descriptor_source_names_path': names_path,
        'balance_samples': bool(balance_samples),
        'saved_compat_test_files': bool(save_compat_test)
    }

    with open('processed_data/external_validation_metadata.json', 'w') as f_meta:
        json.dump(metadata, f_meta, indent=2)

    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("Independent external validation preprocessing completed!")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print("="*70)
    print(f"External validation set: {len(smiles_arr)} samples "
          f"(Pos: {np.sum(labels_arr == 1)}, Neg: {np.sum(labels_arr == 0)})")
    print(f"Descriptor dimension: {descriptors_arr.shape[1]}")
    print("\nSaved files:")
    print("  - processed_data/external_validation.npz")
    print("  - processed_data/external_validation.txt")
    print("  - processed_data/external_validation_metadata.json")
    print("  - descriptors/external_validation_descriptors_selected.csv")
    if save_compat_test:
        print("  - processed_data/test.npz   [compatibility copy for validation scripts]")
        print("  - processed_data/test.txt   [compatibility copy for validation scripts]")

    return {
        'external_validation': {
            'smiles': smiles_arr,
            'proteins': processed_proteins,
            'labels': labels_arr,
            'descriptors': descriptors_arr
        },
        'actual_max_smiles': actual_max_smiles,
        'actual_max_protein': actual_max_protein,
        'filtered_descriptor_names': selected_descriptor_names,
        'metadata': metadata
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process compound-protein interaction data for independent external validation."
    )
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to external validation dataset.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive/negative samples.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Max protein length.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state.')
    parser.add_argument('--metadata_path', type=str, default='descriptors/descriptor_metadata.json',
                        help='Path to training-stage descriptor metadata.')
    parser.add_argument('--names_path', type=str, default='descriptors/selected_descriptor_names_vif.txt',
                        help='Path to training-stage selected descriptor names.')
    parser.add_argument('--no_compat_test', action='store_true',
                        help='Do not save compatibility copies as processed_data/test.npz and test.txt.')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Configuration parameters")
    print("="*70)
    print(f"External validation file: {args.file_path}")
    print(f"Balance Samples: {args.balance_samples}")
    print(f"Descriptor metadata path: {args.metadata_path}")
    print(f"Descriptor names path: {args.names_path}")
    print(f"Save compatibility test files: {not args.no_compat_test}")

    process_external_validation_data(
        file_path=args.file_path,
        max_protein_len=args.max_protein_len,
        random_state=args.random_state,
        balance_samples=args.balance_samples,
        metadata_path=args.metadata_path,
        names_path=args.names_path,
        save_compat_test=not args.no_compat_test
    )