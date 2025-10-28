import numpy as np


def create_sample(size, indices, dataset):
    mask = np.zeros(size, dtype=bool)
    mask[indices] = True
    return dataset._full.filter(mask)


def create_skewed_splits(dataset, skew_specs, test_size=10000, train_skew_size=None,
                         val_fraction=0.2, rng=None, drop_concepts=[]):
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

    total_size = len(dataset.C)

    all_indices = np.arange(total_size)
    rng.shuffle(all_indices)
    test_indices = all_indices[:test_size]
    remaining_indices = all_indices[test_size:]

    if train_skew_size is None:
        train_skew_size = int(len(remaining_indices) * (1 - val_fraction))
        val_size = len(remaining_indices) - train_skew_size
    else:
        val_size = int((len(remaining_indices) - train_skew_size) * val_fraction)

    train_indices = create_skewed_training_set(
        dataset, skew_specs, remaining_indices, train_skew_size, rng
    )

    used_for_training = set(train_indices)
    val_candidates = [i for i in remaining_indices if i not in used_for_training]
    rng.shuffle(val_candidates)
    val_indices = np.array(val_candidates)[:val_size]

    print(f"Final splits - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # Create samples
    dataset.drop_concepts(drop_concepts)
    dataset.training = create_sample(total_size, train_indices, dataset)
    dataset.validation = create_sample(total_size, val_indices, dataset)
    dataset.test = create_sample(total_size, test_indices, dataset)

    return dataset.training, dataset.validation, dataset.test


def create_skewed_training_set(dataset, skew_specs, available_indices, target_size, rng):
    """Create training set that satisfies skewing requirements."""
    train_indices = []
    used = set()

    # Satisfy each skew specification
    for spec in skew_specs:
        mask = np.ones(len(dataset.C), dtype=bool)
        for concept_name, target_value in spec['concepts'].items():
            concept_idx = dataset.concepts.index(concept_name)
            mask &= (dataset.C[:, concept_idx] == target_value)

        # Only consider available indices
        spec_indices = [i for i in np.where(mask)[0] if i in available_indices and i not in used]
        needed = int(target_size * spec['min_fraction'])

        rng.shuffle(spec_indices)
        take = spec_indices[:min(needed, len(spec_indices))]
        train_indices.extend(take)
        used.update(take)

        print(f"Skew spec {spec['concepts']}: needed {needed}, got {len(take)} (max available {len(spec_indices)})")

    # Fill remaining slots
    remaining_slots = target_size - len(train_indices)
    if remaining_slots > 0:
        unused = [i for i in available_indices if i not in used]
        rng.shuffle(unused)
        train_indices.extend(unused[:remaining_slots])

    return np.array(train_indices)


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
