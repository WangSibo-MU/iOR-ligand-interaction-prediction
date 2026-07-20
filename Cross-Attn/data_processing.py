import numpy as np
from sklearn.model_selection import train_test_split
import os
import argparse
from collections import defaultdict


def print_label_distribution(labels, subset_name):
    """Print binary label distribution for diagnostics."""
    labels = np.asarray(labels)
    total = len(labels)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    pos_ratio = pos / total if total > 0 else 0.0
    neg_ratio = neg / total if total > 0 else 0.0
    print(
        f"  {subset_name}: {total} samples "
        f"(Pos: {pos}, Neg: {neg}, Pos ratio: {pos_ratio:.4f}, Neg ratio: {neg_ratio:.4f})"
    )


def _build_group_indices(groups):
    """Build a group -> sample indices mapping."""
    group_to_indices = defaultdict(list)
    for idx, group in enumerate(groups):
        group_to_indices[str(group).strip()].append(idx)
    return group_to_indices


def _score_group_split(test_n, test_pos, total_n, total_pos, test_size,
                       size_weight=1.0, label_weight=2.0, ratio_weight=1.0):
    """
    Score a candidate group split. Lower is better.

    The score jointly optimizes:
    1) test-set size close to the requested test_size;
    2) positive/negative counts close to stratified targets;
    3) positive ratio close to the global positive ratio.
    """
    total_neg = total_n - total_pos
    test_neg = test_n - test_pos

    target_test_n = total_n * test_size
    target_test_pos = total_pos * test_size
    target_test_neg = total_neg * test_size

    size_error = abs(test_n - target_test_n) / max(total_n, 1)
    pos_error = abs(test_pos - target_test_pos) / max(total_pos, 1)
    neg_error = abs(test_neg - target_test_neg) / max(total_neg, 1)
    label_error = pos_error + neg_error

    global_pos_ratio = total_pos / total_n if total_n > 0 else 0.0
    test_pos_ratio = test_pos / test_n if test_n > 0 else 0.0
    ratio_error = abs(test_pos_ratio - global_pos_ratio)

    # Strongly discourage splits in which train or test loses one class.
    penalty = 0.0
    train_pos = total_pos - test_pos
    train_neg = total_neg - test_neg
    if total_pos > 0 and (test_pos == 0 or train_pos == 0):
        penalty += 1000.0
    if total_neg > 0 and (test_neg == 0 or train_neg == 0):
        penalty += 1000.0

    return size_weight * size_error + label_weight * label_error + ratio_weight * ratio_error + penalty


def stratified_group_cold_start_split(groups, labels, test_size=0.2, random_state=42,
                                      n_trials=5000, candidates_per_step=64,
                                      size_weight=1.0, label_weight=2.0,
                                      ratio_weight=1.0):
    """
    Group-aware stratified train/test split for cold-start evaluation.

    Each group is assigned entirely to either train or test, preventing
    ligand/protein leakage. Among randomized greedy candidate splits, the
    function selects the split that best matches the requested test size and
    the global positive/negative label distribution.
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    rng = np.random.default_rng(random_state)
    groups = np.asarray([str(g).strip() for g in groups], dtype=object)
    labels = np.asarray(labels)

    unique_labels = set(np.unique(labels).tolist())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"stratified_group_cold_start_split expects binary labels 0/1, got {sorted(unique_labels)}"
        )

    total_n = len(labels)
    total_pos = int(np.sum(labels == 1))
    total_neg = total_n - total_pos

    if total_n == 0:
        raise ValueError("Cannot split an empty dataset.")
    if total_pos == 0 or total_neg == 0:
        raise ValueError("Cannot perform label-stratified splitting because only one class is present.")

    group_to_indices = _build_group_indices(groups)
    unique_groups = np.array(list(group_to_indices.keys()), dtype=object)
    if len(unique_groups) < 2:
        raise ValueError("Cannot perform cold-start split because fewer than two unique groups are available.")

    group_n = {}
    group_pos = {}
    for g, idxs in group_to_indices.items():
        idxs_arr = np.asarray(idxs, dtype=int)
        group_n[g] = len(idxs_arr)
        group_pos[g] = int(np.sum(labels[idxs_arr] == 1))

    target_test_n = total_n * test_size

    best_score = np.inf
    best_test_groups = None
    best_stats = None

    for trial in range(max(1, n_trials)):
        remaining = unique_groups.copy()
        rng.shuffle(remaining)
        remaining = list(remaining)

        test_groups = []
        test_n = 0
        test_pos = 0

        while remaining and test_n < target_test_n:
            if len(remaining) <= candidates_per_step:
                candidate_positions = np.arange(len(remaining))
            else:
                candidate_positions = rng.choice(len(remaining), size=candidates_per_step, replace=False)

            best_candidate_pos = None
            best_candidate_score = np.inf

            for pos in candidate_positions:
                g = remaining[int(pos)]
                cand_n = test_n + group_n[g]
                cand_pos = test_pos + group_pos[g]
                cand_score = _score_group_split(
                    cand_n, cand_pos, total_n, total_pos, test_size,
                    size_weight=size_weight,
                    label_weight=label_weight,
                    ratio_weight=ratio_weight,
                )
                if cand_score < best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_pos = int(pos)

            selected_group = remaining.pop(best_candidate_pos)
            test_groups.append(selected_group)
            test_n += group_n[selected_group]
            test_pos += group_pos[selected_group]

        score = _score_group_split(
            test_n, test_pos, total_n, total_pos, test_size,
            size_weight=size_weight,
            label_weight=label_weight,
            ratio_weight=ratio_weight,
        )

        if score < best_score:
            best_score = score
            best_test_groups = set(test_groups)
            best_stats = {
                "score": float(score),
                "trial": int(trial),
                "test_n": int(test_n),
                "test_pos": int(test_pos),
                "test_neg": int(test_n - test_pos),
                "test_group_count": int(len(test_groups)),
            }

    if best_test_groups is None:
        raise RuntimeError("Failed to construct a stratified group cold-start split.")

    train_indices = []
    test_indices = []
    for g, idxs in group_to_indices.items():
        if g in best_test_groups:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_groups = set(groups[train_indices].tolist())
    test_groups = set(groups[test_indices].tolist())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected: {len(overlap)} groups appear in both train and test.")

    split_info = {
        **best_stats,
        "train_n": int(len(train_indices)),
        "train_pos": int(np.sum(labels[train_indices] == 1)),
        "train_neg": int(np.sum(labels[train_indices] == 0)),
        "train_group_count": int(len(train_groups)),
        "total_group_count": int(len(unique_groups)),
        "target_test_size": float(test_size),
        "actual_test_size": float(len(test_indices) / total_n),
        "global_pos_ratio": float(total_pos / total_n),
        "train_pos_ratio": float(np.sum(labels[train_indices] == 1) / max(len(train_indices), 1)),
        "test_pos_ratio": float(np.sum(labels[test_indices] == 1) / max(len(test_indices), 1)),
    }

    return train_indices, test_indices, split_info


def balance_binary_subset(smiles, proteins, labels, random_state=42, subset_name="subset"):
    """Balance one subset to a 1:1 positive/negative ratio by random undersampling."""
    rng = np.random.default_rng(random_state)
    labels = np.asarray(labels)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError(
            f"Cannot balance {subset_name}: Pos={len(pos_idx)}, Neg={len(neg_idx)}. Both classes must be present."
        )

    min_count = min(len(pos_idx), len(neg_idx))
    pos_sampled = rng.choice(pos_idx, min_count, replace=False)
    neg_sampled = rng.choice(neg_idx, min_count, replace=False)
    balanced_idx = np.concatenate([pos_sampled, neg_sampled])
    rng.shuffle(balanced_idx)

    return smiles[balanced_idx], proteins[balanced_idx], labels[balanced_idx]


def process_data(file_path='dataset.txt', max_protein_len=480, max_smiles_len=70,
                 test_size=0.2, random_state=42, ligand_cold_start=False,
                 protein_cold_start=False, balance_samples=False,
                 stratified_group_trials=5000):
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

    if len(smiles_list) == 0:
        raise ValueError("No valid samples found. Each line should contain: SMILES protein label")

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

    print("\nGlobal label distribution:")
    print_label_distribution(labels_array, "All data")

    # 选择划分策略：配体冷启动、蛋白冷启动、或常规随机分层
    if ligand_cold_start:
        print("\n[Split Mode] Stratified Group Ligand Cold Start (配体分组分层冷启动)")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=smiles_array,
            labels=labels_array,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials,
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total ligand groups: {split_info['total_group_count']}")
        print(f"    Train ligand groups: {split_info['train_group_count']}")
        print(f"    Test ligand groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

        smiles_train, smiles_test = smiles_array[train_indices], smiles_array[test_indices]
        protein_train, protein_test = protein_array[train_indices], protein_array[test_indices]
        labels_train, labels_test = labels_array[train_indices], labels_array[test_indices]

    elif protein_cold_start:
        print("\n[Split Mode] Stratified Group Protein Cold Start (蛋白分组分层冷启动)")
        train_indices, test_indices, split_info = stratified_group_cold_start_split(
            groups=protein_array,
            labels=labels_array,
            test_size=test_size,
            random_state=random_state,
            n_trials=stratified_group_trials,
        )
        print("  Stratified group split diagnostics:")
        print(f"    Total protein groups: {split_info['total_group_count']}")
        print(f"    Train protein groups: {split_info['train_group_count']}")
        print(f"    Test protein groups: {split_info['test_group_count']}")
        print(f"    Target test size: {split_info['target_test_size']:.4f}")
        print(f"    Actual test size: {split_info['actual_test_size']:.4f}")
        print(f"    Global pos ratio: {split_info['global_pos_ratio']:.4f}")
        print(f"    Train pos ratio: {split_info['train_pos_ratio']:.4f}")
        print(f"    Test pos ratio: {split_info['test_pos_ratio']:.4f}")
        print(f"    Best split score: {split_info['score']:.6f} (trial {split_info['trial']})")

        smiles_train, smiles_test = smiles_array[train_indices], smiles_array[test_indices]
        protein_train, protein_test = protein_array[train_indices], protein_array[test_indices]
        labels_train, labels_test = labels_array[train_indices], labels_array[test_indices]

    else:
        print("\n[Split Mode] Random Stratified (随机分层)")
        smiles_train, smiles_test, protein_train, protein_test, labels_train, labels_test = train_test_split(
            smiles_array,
            protein_array,
            labels_array,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_array,
        )

    print("\nInitial split label distribution:")
    print_label_distribution(labels_train, "Train before balancing")
    print_label_distribution(labels_test, "Test before balancing")

    # 切分后的集合内动态强制平衡
    if balance_samples:
        print("\n[Balance] Applying post-split random undersampling to Train and Test sets...")
        smiles_train, protein_train, labels_train = balance_binary_subset(
            smiles_train, protein_train, labels_train,
            random_state=random_state,
            subset_name="training set",
        )
        smiles_test, protein_test, labels_test = balance_binary_subset(
            smiles_test, protein_test, labels_test,
            random_state=random_state + 1,
            subset_name="test set",
        )

        print("\nBalanced label distribution:")
        print_label_distribution(labels_train, "Train after balancing")
        print_label_distribution(labels_test, "Test after balancing")

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

    print(f"\nData processing completed, saved to processed_data directory:")
    if ligand_cold_start:
        mode_str = 'Stratified Group Ligand Cold Start (配体分组分层冷启动)'
    elif protein_cold_start:
        mode_str = 'Stratified Group Protein Cold Start (蛋白分组分层冷启动)'
    else:
        mode_str = 'Random Stratified (随机分层)'

    print(f"- Split Mode: {mode_str}")
    print(f"- Balanced Mode: {'Enabled (Strict 1:1 per set)' if balance_samples else 'Disabled'}")
    print(f"- Training set: {len(smiles_train)} samples (Pos: {np.sum(labels_train == 1)}, Neg: {np.sum(labels_train == 0)})")
    print(f"- Test set: {len(smiles_test)} samples (Pos: {np.sum(labels_test == 1)}, Neg: {np.sum(labels_test == 0)})")

    return {
        'train': {'smiles': smiles_train, 'proteins': protein_train, 'labels': labels_train},
        'test': {'smiles': smiles_test, 'proteins': protein_test, 'labels': labels_test},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process compound-protein interaction data.")
    parser.add_argument('--file_path', type=str, default='dataset.txt', help='Path to the dataset text file.')
    parser.add_argument('--ligand_cold_start', action='store_true', help='Enable ligand cold start with group-aware stratified split.')
    parser.add_argument('--protein_cold_start', action='store_true', help='Enable protein cold start with group-aware stratified split.')
    parser.add_argument('--balance_samples', action='store_true', help='Force equal number of positive and negative samples via post-split downsampling.')
    parser.add_argument('--max_protein_len', type=int, default=480, help='Maximum truncated/padded length for proteins.')
    parser.add_argument('--max_smiles_len', type=int, default=70, help='Maximum truncated/padded length for SMILES.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_state', type=int, default=42, help='Random state seed.')
    parser.add_argument('--stratified_group_trials', type=int, default=5000,
                        help='Number of randomized trials for stratified group cold-start splitting.')

    args = parser.parse_args()

    if args.ligand_cold_start and args.protein_cold_start:
        raise ValueError("Cannot enable both ligand cold start and protein cold start at the same time. Please choose one.")

    process_data(
        file_path=args.file_path,
        max_protein_len=args.max_protein_len,
        max_smiles_len=args.max_smiles_len,
        test_size=args.test_size,
        random_state=args.random_state,
        ligand_cold_start=args.ligand_cold_start,
        protein_cold_start=args.protein_cold_start,
        balance_samples=args.balance_samples,
        stratified_group_trials=args.stratified_group_trials,
    )
