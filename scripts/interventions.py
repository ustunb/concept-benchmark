import numpy as np


def compute_intervention_score(pred_probs, frontend_model, budget_k, policy="top-k"):
    """
    Compute intervention score for a single sample.

    Args:
        pred_probs: (n_concepts,) - Concept prediction probabilities
        frontend_model: Trained frontend model
        budget_k: Number of concepts to intervene on
        policy: "top-1" or "top-k"

    Returns:
        score: float - Probability that intervention changes prediction
        best_concepts: list - Indices of concepts to intervene on
    """
    n_concepts = len(pred_probs)
    c_rounded = (pred_probs > 0.5).astype(np.float32)
    pred_original = np.argmax(frontend_model.predict_proba(c_rounded.reshape(1, -1))[0])

    if policy == "top-1":
        best_score = 0.0
        best_concept = None

        for j in range(n_concepts):
            total_prob_change = 0.0

            # Outcome 1: concept j = 1 (with probability pred_probs[j])
            c_if_one = c_rounded.copy()
            c_if_one[j] = 1
            pred_if_one = np.argmax(frontend_model.predict_proba(c_if_one.reshape(1, -1))[0])
            if pred_if_one != pred_original:
                total_prob_change += pred_probs[j]

            # Outcome 2: concept j = 0 (with probability 1 - pred_probs[j])
            c_if_zero = c_rounded.copy()
            c_if_zero[j] = 0
            pred_if_zero = np.argmax(frontend_model.predict_proba(c_if_zero.reshape(1, -1))[0])
            if pred_if_zero != pred_original:
                total_prob_change += (1 - pred_probs[j])

            if total_prob_change > best_score:
                best_score = total_prob_change
                best_concept = j

        return best_score, [best_concept] if best_concept is not None else []


    elif policy == "top-k":

        from itertools import combinations, product
        # Pre-generate all combinations
        all_combinations = np.array(list(product([0, 1], repeat=budget_k)))  # (2^K, K)
        n_combinations = len(all_combinations)

        # Get all subsets
        all_subsets = list(combinations(range(n_concepts), budget_k))
        n_subsets = len(all_subsets)

        # Process in batches to manage memory
        batch_size = 100
        best_score = 0.0
        best_subset = []

        for batch_start in range(0, n_subsets, batch_size):
            # print("Processing batch starting at subset index", batch_start, "of", n_subsets)
            batch_end = min(batch_start + batch_size, n_subsets)
            batch_subsets = all_subsets[batch_start:batch_end]
            n_batch = len(batch_subsets)

            # Create all concept vectors for this batch: (n_batch * 2^K, n_concepts)
            c_batch = np.tile(c_rounded, (n_batch * n_combinations, 1))
            prob_batch = np.ones(n_batch * n_combinations)

            # Apply interventions for all subsets in batch
            for batch_idx, subset in enumerate(batch_subsets):
                subset = np.array(subset)
                start_idx = batch_idx * n_combinations
                end_idx = start_idx + n_combinations

                # Set concepts for all combinations of this subset
                c_batch[start_idx:end_idx, subset] = all_combinations

                # Compute probabilities
                for idx, j in enumerate(subset):
                    prob_batch[start_idx:end_idx] *= np.where(
                        all_combinations[:, idx] == 1,
                        pred_probs[j],
                        1 - pred_probs[j]
                    )

            # Single prediction call for entire batch
            pred_probs_all = frontend_model.predict_proba(c_batch)
            pred_all = np.argmax(pred_probs_all, axis=1)
            prediction_changes = (pred_all != pred_original).astype(float)

            # Compute scores for each subset in batch
            for batch_idx in range(n_batch):
                start_idx = batch_idx * n_combinations
                end_idx = start_idx + n_combinations
                score = np.sum(prob_batch[start_idx:end_idx] * prediction_changes[start_idx:end_idx])
                if score > best_score:
                    best_score = score
                    best_subset = list(batch_subsets[batch_idx])

        return best_score, best_subset

    else:
        raise ValueError(f"Unknown policy: {policy}")


def select_samples_for_intervention(pred_probs, frontend_model, budget_k,
                                    intervention_threshold, policy="top-k"):
    """
    Score all samples and select which ones to intervene on.

    Args:
        pred_probs: (n_samples, n_concepts) - Concept prediction probabilities
        frontend_model: Trained frontend model
        budget_k: Number of concepts to intervene on per sample
        intervention_threshold: Minimum score required to intervene
        policy: "top-1" or "top-k"

    Returns:
        samples_to_intervene: list of (sample_idx, score, concepts_to_check)
    """
    n_samples = pred_probs.shape[0]
    samples_to_intervene = []

    for i in range(n_samples):
        score, best_concepts = compute_intervention_score(
            pred_probs[i], frontend_model, budget_k, policy
        )

        if score >= intervention_threshold:
            samples_to_intervene.append((i, score, best_concepts))

    return samples_to_intervene


def apply_interventions(pred_probs, ground_truth, frontend_model, budget_k,
                        intervention_threshold=0.0, human_accuracy=1.0,
                        policy="top-k", rng=None):
    """
    Apply human interventions to concept predictions.

    Args:
        pred_probs: (n_samples, n_concepts) - Concept prediction probabilities
        ground_truth: (n_samples, n_concepts) - True concept values (binary)
        frontend_model: Trained frontend model for final predictions
        budget_k: int - Number of concepts to intervene on per sample
        intervention_threshold: float - Minimum score to intervene on a sample (0 to 1)
        human_accuracy: float - Probability human gives correct intervention
        policy: str - "top-1" or "top-k" intervention selection
        rng: np.random.Generator - For reproducibility

    Returns:
        intervened_concepts: (n_samples, n_concepts) - Binary concepts after interventions
        intervention_stats: dict - Statistics about interventions applied
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stage 1: Score and select samples to intervene on
    samples_to_intervene = select_samples_for_intervention(
        pred_probs, frontend_model, budget_k, intervention_threshold, policy
    )

    # Initialize output - start with rounded binary concepts
    intervened_concepts = (pred_probs >= 0.5).astype(int)
    n_samples, n_concepts = pred_probs.shape
    edit_counts = np.zeros(n_samples, dtype=int)

    # Stage 2: Apply interventions to selected samples
    for sample_idx, score, concepts_to_check in samples_to_intervene:
        actual_edits = 0

        for j in concepts_to_check:
            original_value = intervened_concepts[sample_idx, j]

            # Human intervention
            if rng.random() < human_accuracy:
                # Human gives correct value
                new_value = ground_truth[sample_idx, j]
            else:
                # Human makes error
                new_value = 1 - ground_truth[sample_idx, j]

            # Update the binary concept value
            intervened_concepts[sample_idx, j] = new_value

            # Count as edit if value changed
            if original_value != new_value:
                actual_edits += 1

        edit_counts[sample_idx] = actual_edits

    # Statistics
    n_interventions = len(samples_to_intervene)
    stats = {
        "samples_intervened_on": int(n_interventions),
        "intervention_rate": float(n_interventions) / n_samples,
        "avg_edits_per_intervention": float(edit_counts[edit_counts > 0].mean()) if n_interventions > 0 else 0.0,
        "total_concept_checks": int(budget_k * n_interventions),
        "total_concept_edits_made": int(edit_counts.sum()),
        "avg_score": float(np.mean([s for _, s, _ in samples_to_intervene])) if n_interventions > 0 else 0.0,
        "max_score": float(max([s for _, s, _ in samples_to_intervene])) if n_interventions > 0 else 0.0,
        "min_score": float(min([s for _, s, _ in samples_to_intervene])) if n_interventions > 0 else 0.0,
    }

    return intervened_concepts, stats
