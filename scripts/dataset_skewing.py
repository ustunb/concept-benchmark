import numpy as np


def create_sample(size, indices, dataset):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return dataset._full.filter(mask)


def create_skewed_splits(dataset, skew_specs, train_fraction=0.5, val_fraction=0.25, test_fraction=0.25, rng=None, drop_concepts=[], fractions_unique=True):
    """
    Skew training by ensuring minimum representation of specific concept patterns.

    Args:
        skew_specs: List of dicts, each with 'concepts' (dict of concept:value) and 'min_fraction' (float)
                   e.g., [{'concepts': {'body_shape': 0, 'foot_shape_3sided': 1}, 'min_fraction': 0.4},
                          {'concepts': {'body_shape': 0, 'foot_shape_4sided': 1}, 'min_fraction': 0.4}]
        train_fraction, val_fraction, test_fraction: Split proportions
        rng: Random number generator
        drop_concepts: List of concept names to drop from dataset after splitting
        fractions_unique: If True, fractions regard the unique set of concepts, not total samples
    """
    if rng is None:
        rng = np.random.default_rng()

    # print y labels all
    print("Overall class distribution in full dataset:")
    unique, counts = np.unique(dataset.y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    for cls, cnt in class_dist.items():
        print(f"  Class {cls}: {cnt} samples ({cnt / len(dataset.y):.1%})")

    total_size = len(dataset.C)
    total_unique_size = total_size if not fractions_unique else dataset.meta["num_unique_robots"]
    desired_train_size = int(total_unique_size * train_fraction)
    print("Desired training size:", desired_train_size)

    # Find indices matching each specification
    spec_indices = []
    for spec in skew_specs:
        mask = np.ones(total_size, dtype=bool)
        for concept_name, target_value in spec['concepts'].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= (dataset.C[:, concept_idx] == target_value)
        spec_indices.append(np.where(mask)[0])

    train_indices = []
    used = set()
    for spec, indices in zip(skew_specs, spec_indices):
        needed = int(desired_train_size * spec['min_fraction'])
        available = [i for i in indices if i not in used]
        rng.shuffle(available)
        take = available[:min(needed, len(available))]
        train_indices.extend(take)
        used.update(take)
        print(f"Added {len(take)} for spec {spec['concepts']} (wanted {needed})")

    # Fill remaining slots with any unused samples
    remaining_slots = desired_train_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in range(total_size) if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])
        print(f"Filled {min(remaining_slots, len(unused))} remaining slots")

    train_indices = np.array(train_indices)
    rng.shuffle(train_indices)

    # Validation and test from what's left
    remaining = np.array([i for i in range(total_size) if i not in train_indices])
    rng.shuffle(remaining)

    val_size = int(len(remaining) * val_fraction / (val_fraction + test_fraction))
    val_indices = remaining[:val_size]
    test_indices = remaining[val_size:]
    print("Resulting training size:", len(train_indices))

    dataset.drop_concepts(drop_concepts)

    dataset.training = create_sample(total_size, train_indices, dataset)
    dataset.validation = create_sample(total_size, val_indices, dataset)
    dataset.test = create_sample(total_size, test_indices, dataset)

    return dataset.training, dataset.validation, dataset.test


def filter_training_by_string(dataset, string, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2, rng=None):
    """
    Filter robots for training set based on model string, put rest in val/test.

    Args:
        dataset: ConceptDataset instance
        string: String condition to evaluate for training selection
        train_fraction, val_fraction, test_fraction: Split proportions
        rng: Random number generator

    Returns:
        train, validation, test splits
    """
    if rng is None:
        rng = np.random.default_rng()

    def create_row_dict(sample_idx):
        row = {}
        for i, concept_name in enumerate(dataset.concepts):
            concept_value = dataset.C[sample_idx, i]

            if concept_name == 'body_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'round'
            elif concept_name == 'head_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'round'
            elif concept_name in ['has_knees', 'has_elbows', 'has_antennae']:
                row[concept_name] = 'true' if concept_value == 1 else 'false'
            elif concept_name == 'ears_shape':
                row[concept_name] = 'square' if concept_value == 0 else 'triangle'
            elif concept_name == 'mouth_type':
                row[concept_name] = 'closed' if concept_value == 0 else 'open'
            elif concept_name == 'hand_shape':
                row[concept_name] = 'round_circle' if concept_value == 0 else 'edgy_triangle'
            elif concept_name == 'foot_shape':
                row[concept_name] = 'flat_4sided' if concept_value == 0 else 'pointy_3sided'
            else:
                row[concept_name] = concept_value

        return row

    train_candidates = []
    other_samples = []

    for idx in range(len(dataset.C)):
        row = create_row_dict(idx)
        try:
            print(row)
            if eval(string, {"row": row}):
                print("  -> Train candidate")
                train_candidates.append(idx)
            else:
                other_samples.append(idx)
        except Exception as e:
            print(f"Error evaluating condition for sample {idx}: {e}")
            other_samples.append(idx)

    train_candidates = np.array(train_candidates)
    other_samples = np.array(other_samples)

    print(f"Candidates for training (satisfy condition): {len(train_candidates)}")
    print(f"Other samples: {len(other_samples)}")

    total_size = len(dataset.C)
    desired_train_size = int(total_size * train_fraction)

    actual_train_size = min(len(train_candidates), desired_train_size)
    rng.shuffle(train_candidates)
    train_indices = train_candidates[:actual_train_size]

    unused_candidates = train_candidates[actual_train_size:]
    remaining_samples = np.concatenate([unused_candidates, other_samples])
    rng.shuffle(remaining_samples)

    remaining_size = len(remaining_samples)
    val_size = int(remaining_size * val_fraction / (val_fraction + test_fraction))

    val_indices = remaining_samples[:val_size]
    test_indices = remaining_samples[val_size:]

    dataset.training = create_sample(total_size, train_indices, dataset)
    dataset.validation = create_sample(total_size, val_indices, dataset)
    dataset.test = create_sample(total_size, test_indices, dataset)

    print(f"\nFinal splits:")
    print(f"Training: {len(train_indices)} samples ({len(train_indices) / total_size:.1%})")
    print(f"Validation: {len(val_indices)} samples ({len(val_indices) / total_size:.1%})")
    print(f"Test: {len(test_indices)} samples ({len(test_indices) / total_size:.1%})")

    return dataset.training, dataset.validation, dataset.test
