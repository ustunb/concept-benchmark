import copy
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np

from concept_benchmark.synthetic.robot import create_synthetic_dataset
from scripts.dataset_skewing import create_skewed_splits, filter_training_by_string

from scripts.robot_image_training import settings, main
from scripts.robot_utils import powerset, find_params_for_target_probabilities, _apply_label_noise, _apply_missing

# automated experimentation
# vary:
# - subconcepts: subconcepts in the dataset foot_shape, hand_shape or both
# - skew_concepts: subconcepts to include in the training dataset;
#   only setups where there are at least 2 subconcepts for value1 (e.g., pointy) and 2 subconcepts for value2 (e.g., flat)
# - skew_concepts min_fraction: percentage of the subconcepts in the training dataset; the distributio between
#   subconcepts for the same value of a concept need to sum to 0.5; but at least 1 value has to be 0.005, others have to
#   be >= 0.005 in increments of 0.005
# - drop_concepts: whether we train a subconcepts CBM and then drop the subconcepts that are not in the training set + foot_shape
#   or a CBM where we drop all subconcepts
# - logit_weights, scalar, intercept: chosen so that the feature combinations (tuples) for the features in the model
#   mapped to a desired P(glorp) e.g., {(0, 0): 0.05, (1, 0): 0.50, (0, 1): 0.95, (1, 1): 0.99}



def run_experiments_varying_footshape_subconcepts():
    base_subconcepts = [
        "foot_shape_flat_trapezoid",
        "foot_shape_flat_rounded",
        "foot_shape_flat_square",
        "foot_shape_flat_5sided",
        "foot_shape_flat_lshaped",
        "foot_shape_pointy_trapezoid",
        "foot_shape_pointy_rounded",
        "foot_shape_pointy_square",
        "foot_shape_pointy_3sided",
        "foot_shape_pointy_4sided",
    ]
    for r in range(2, len(base_subconcepts) + 1):
        for subset in combinations(base_subconcepts, r):
            if "pointy" not in "_".join(subset) or "flat" not in "_".join(subset):
                # skip subsets that do not have at least one pointy and one flat subtype
                continue
            S = copy.deepcopy(settings)
            skew_list = []
            drop_list = list(set(base_subconcepts) - set(subset))
            subset_pointy = [sc for sc in subset if "pointy" in sc]
            subset_flat = [sc for sc in subset if "flat" in sc]
            for sc in subset_pointy:
                skew_list.append({'concepts': {sc: 1}, 'min_fraction': round(0.5 / len(subset_pointy), 2)})
            for sc in subset_flat:
                skew_list.append({'concepts': {sc: 1}, 'min_fraction': round(0.5 / len(subset_flat), 2)})
            S["skew_concept"] = skew_list
            S["drop_concepts"] = drop_list + ["foot_shape"]
            S["run_name"] = "loop_footshape_" + "_".join([sc.split("_")[-2][0] + sc.split("_")[-1][0] for sc in subset])
            print(f"Running experiment with skewed subconcepts: {subset},mrun name: {S['run_name']}")
            main(S)


def get_subconcept_combinations(subconcept_list):
    """
    Generate all combinations of subconcepts from the given list that contain at least 2 subconcepts per each value of the concept.

    :param subconcept_list:
    :return:
    """
    subconcept_combinations = []
    n = len(subconcept_list)
    for r in range(2, n + 1):
        for subset in combinations(subconcept_list, r):
            subset_pointy = [sc for sc in subset if "pointy" in sc or "round" in sc]
            subset_flat = [sc for sc in subset if "flat" in sc or "edgy" in sc]
            if len(subset_pointy) >= 2 and len(subset_flat) >= 2:
                subconcept_combinations.append(subset)
    return subconcept_combinations


def get_N_floats_summing_to_0_5_with_min(num_floats, min_value=0.005, step=0.005):
    """
    Generate all combinations of `num_floats` floats that sum to 0.5, with each float >= min_value and in increments of `step`.

    :param num_floats:
    :param min_value:
    :param step:
    :return:
    """
    combinations = []
    def backtrack(remaining, current_combination):
        if len(current_combination) == num_floats - 1:
            last_value = round(0.5 - sum(current_combination), 3)
            if last_value >= min_value and round(last_value % step, 3) == 0:
                combinations.append(current_combination + [last_value])
            return
        for value in np.arange(min_value, 0.5 - sum(current_combination) - min_value * (num_floats - len(current_combination) - 1) + step, step):
            backtrack(remaining - 1, current_combination + [round(value, 3)])
    backtrack(num_floats, [])
    return combinations


def get_monotonic_probs(num_concepts, lowest=0.01, step=0.01):
    """
    Systematically generate probability assignments with subset monotonicity and dependencies
    """
    all_combos = list(product([0, 1], repeat=num_concepts))
    prob_values = [lowest + i * step for i in range(int((1.0 - lowest) / step) + 1)]
    all_combos.sort(key=lambda x: (sum(x), x))

    # Build constraint graph: combo_i -> [(type, combo_j), ...]
    constraints = {i: [] for i in range(len(all_combos))}

    for i, combo_i in enumerate(all_combos):
        for j, combo_j in enumerate(all_combos):
            if i == j:
                continue

            # Subset monotonicity: if combo_j ⊆ combo_i, then P(combo_j) ≤ P(combo_i)
            if all(combo_j[k] <= combo_i[k] for k in range(num_concepts)) and combo_j != combo_i:
                constraints[i].append(('>=', j))  # combo_i >= combo_j

    def propagate(domains):
        """
        Constraint propagation
        Shrinks the possible values each combination can take by applying all constraints repeatedly until no more
        shrinking is possible
        """
        changed = True
        while changed:
            changed = False
            for i in range(len(all_combos)):
                old_size = len(domains[i])

                for constraint_type, j in constraints[i]:
                    if constraint_type == '>=':
                        min_j = min(domains[j]) if domains[j] else float('inf')
                        domains[i] = [p for p in domains[i] if p >= min_j]

                if len(domains[i]) != old_size:
                    changed = True
                if not domains[i]:
                    return False
        return True

    def search(idx, assignment, domains):
        """Backtracking search"""
        # idx = which combination we're currently assigning
        # assignment = probability values assigned so far [p0, p1, p2, ...]
        # domains = remaining possible values for each combination
        if idx == len(all_combos):
            yield {all_combos[i]: assignment[i] for i in range(len(all_combos))}
            return

        # rry each possible value for current combination
        for prob in domains[idx]:
            new_domains = [d.copy() for d in domains]
            new_domains[idx] = [prob]

            # can we satisfy all constraints?
            if propagate(new_domains):
                yield from search(idx + 1, assignment + [prob], new_domains)

    # initialize and solve
    domains = [prob_values.copy() for _ in range(len(all_combos))]
    if propagate(domains):
        yield from search(0, [], domains)


def grid_search(sttngs, nonsalient_concepts, cbm_type="cbm"):
    # create all combinations of nonsalient concepts, including singletons
    nonsalient_concepts_combinations = powerset(nonsalient_concepts)[1:]  # exclude empty set
    num_concepts_in_model = len(sttngs['logit_weights'].keys())
    feature_names = list(sttngs['logit_weights'].keys())

    base_subconcepts = {
        'foot_shape_subtype': ['foot_shape_flat_trapezoid', 'foot_shape_flat_rounded',
                               'foot_shape_flat_square', 'foot_shape_flat_5sided',
                               'foot_shape_flat_lshaped', 'foot_shape_pointy_trapezoid',
                               'foot_shape_pointy_rounded', 'foot_shape_pointy_square',
                               'foot_shape_pointy_3sided', 'foot_shape_pointy_4sided'],
        'hand_shape_subtype': ["hand_shape_round_circle", "hand_shape_round_oval",
                               "hand_shape_round_oval2", "hand_shape_edgy_triangle",
                               "hand_shape_edgy_square", "hand_shape_edgy_trapezoid"]
    }
    all_subconcepts = [v for sublist in base_subconcepts.values() for v in sublist]

    for nonsalient_concept_exp_tuple in nonsalient_concepts_combinations:
        nonsalient_concept_exp = list(nonsalient_concept_exp_tuple)
        if len(nonsalient_concept_exp) == 2:
            print("GOT IT")

            subset_combinations_per_concept = {concept: [] for concept in nonsalient_concept_exp}
            for concept in nonsalient_concept_exp:
                subconcept_combinations = get_subconcept_combinations(base_subconcepts[concept])
                subset_combinations_per_concept[concept] = subconcept_combinations
            # create all combinations between subconcepts in the different concepts
            print("Sizes of subconcept combinations per concept:", {k: len(v) for k, v in subset_combinations_per_concept.items()})
            all_combs = list(product(*subset_combinations_per_concept.values()))
            print(f"Total combinations to evaluate: {len(all_combs)}")
            for subset in all_combs:
                # make one tuple from all tuples inside
                subset = tuple([item for subtuple in subset for item in subtuple])
                skew_list = []
                drop_list = list(set(all_subconcepts) - set(list(subset)))
                print(f"NEW SUBSET: {subset}")

                # get subconcept distributions separately for each concept in the experiment, then divide by len(nonsalient_concept_exp)
                # and merge into one skew_list
                subconcept_distributions1_per_concept, subconcept_distributions2_per_concept = {}, {}
                subset_value1, subset_value2 = [], []
                for concept in nonsalient_concept_exp:
                    print(f"Processing concept: {concept}")
                    nosubtype_concept = concept.split("_")[0]
                    sv1 = [sc for sc in subset if ("pointy" in sc or "round" in sc) and nosubtype_concept in sc]
                    sv2 = [sc for sc in subset if ("flat" in sc or "edgy" in sc) and nosubtype_concept in sc]
                    subset_value1 += sv1
                    subset_value2 += sv2
                    print(sv1, sv2)

                    print(f"  Subset value1 subconcepts: {sv1}")
                    subconcept_distributions1 = get_N_floats_summing_to_0_5_with_min(len(sv1), min_value=0.05, step=0.05)
                    print("Got distributions for value1:", len(subconcept_distributions1))
                    subconcept_distributions2 = get_N_floats_summing_to_0_5_with_min(len(sv2), min_value=0.05, step=0.05)
                    subconcept_distributions1_per_concept[concept] = subconcept_distributions1
                    subconcept_distributions2_per_concept[concept] = subconcept_distributions2

                # create all combinations between the distributions for the different concepts
                print("Generating all combinations of subconcept distributions across concepts...")
                subconcept_distributions1 = list(product(*subconcept_distributions1_per_concept.values()))
                subconcept_distributions1 = [el for dist_tuple in subconcept_distributions1 for el in dist_tuple]
                subconcept_distributions2 = list(product(*subconcept_distributions2_per_concept.values()))
                subconcept_distributions2 = [el for dist_tuple in subconcept_distributions2 for el in dist_tuple]
                # divide all over len(nonsalient_concept_exp) to ensure total sum is 1.0
                subconcept_distributions1 = [[val / len(nonsalient_concept_exp) for val in dist_tuple] for dist_tuple in subconcept_distributions1]
                subconcept_distributions2 = [[val / len(nonsalient_concept_exp) for val in dist_tuple] for dist_tuple in subconcept_distributions2]
                print(f"Total combinations of subconcept distributions to evaluate: {len(subconcept_distributions1) * len(subconcept_distributions2)}")

                for dist1 in subconcept_distributions1:
                    for dist2 in subconcept_distributions2:
                        print("NEW DISTRIBUTION COMBINATION: ", dist1, dist2)

                        for i, sc in enumerate(subset_value1):
                            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist1[i])})
                        for i, sc in enumerate(subset_value2):
                            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist2[i])})

                        # vary stochasticity
                        for target_probs in get_monotonic_probs(num_concepts=num_concepts_in_model, lowest=0.01, step=0.01):
                            vals = list(target_probs.values())
                            vals = [round(v, 2) for v in vals]
                            labeling_func_details = find_params_for_target_probabilities(feature_names=feature_names,
                                                                                         target_probs=vals)
                            achieved_probs = {combo: float(stats['achieved'])
                                              for combo, stats in labeling_func_details['verification'].items()}

                            S = copy.deepcopy(sttngs)
                            S["subconcepts"] = nonsalient_concept_exp
                            S['logit_weights'] = labeling_func_details['logit_weights']
                            S['logit_intercept'] = labeling_func_details['logit_intercept']
                            S['logit_scalar'] = labeling_func_details['logit_scalar']
                            S["skew_concept"] = skew_list
                            S["drop_concepts"] = drop_list + [concept.split("_")[0] for concept in nonsalient_concept_exp] if cbm_type == "subconcept_cbm" else [b for concept in nonsalient_concept_exp for b in base_subconcepts[concept]]
                            S["run_name"] = f"matrix_{nonsalient_concept_exp}_" + "_".join([sc.split("_")[-2][0] + sc.split("_")[-1][0] for sc in subset]) + \
                                            f"_monoprob_" + "_".join([f"{''.join(map(str, k))}-{int(v*100)}" for k, v in achieved_probs.items()]) + \
                                            f"_seed{int(S['seed'])}"

                            print(f"Running experiment with the following parameters:\n")
                            print(f"CBM type: {cbm_type}")
                            print("Used subconcepts:", nonsalient_concept_exp)
                            print("Skewed subconcepts with prevalence:", S["skew_concept"])
                            print("Dropped subconcepts:", S["drop_concepts"])
                            print("Desired probabilities:", target_probs)
                            print("Achieved probabilities:", achieved_probs)
                            print("Logit weights:", S['logit_weights'])
                            print("Logit scalar:", S['logit_scalar'])
                            print("Logit intercept:", S['logit_intercept'])

                            main(S)


def principled_adaptive_search(sttngs, nonsalient_concepts, max_trials=500):
    """
    Hierarchical adaptive sampling
    """
    results = []
    exploration_phase = max_trials // 3
    rng = np.random.default_rng(12345)

    concept_rewards = defaultdict(list)
    subset_rewards = defaultdict(list)
    dist_rewards = defaultdict(list)
    prob_rewards = defaultdict(list)

    for trial in range(max_trials):
        if trial < exploration_phase:
            config = sample_random_config(sttngs, nonsalient_concepts, rng)
        else:
            config = sample_promising_config(sttngs, nonsalient_concepts,
                                             concept_rewards, subset_rewards,
                                             dist_rewards, prob_rewards, rng)

        if config is None:
            continue

        # RUN TWO EXPERIMENTS TO COMPARE CBM vs SUBCONCEPT CBM
        config_cbm = copy.deepcopy(config)
        # drop all subconcepts for a cbm experiment
        config_cbm["drop_concepts"] = ['foot_shape_flat_trapezoid', 'foot_shape_flat_rounded',
                   'foot_shape_flat_square', 'foot_shape_flat_5sided',
                   'foot_shape_flat_lshaped', 'foot_shape_pointy_trapezoid',
                   'foot_shape_pointy_rounded', 'foot_shape_pointy_square',
                   'foot_shape_pointy_3sided', 'foot_shape_pointy_4sided',
                   'hand_shape_round_circle', 'hand_shape_round_oval',
                   'hand_shape_round_oval2', 'hand_shape_edgy_triangle',
                   'hand_shape_edgy_square', 'hand_shape_edgy_trapezoid']
        config_cbm["run_name"] = config["run_name"] + "_cbm"

        print("\nRunning CBM\n")
        metrics_cbm = main(config_cbm)
        cbm_acc = metrics_cbm.get('cbm_acc_detected', 0)

        config_subconcept = copy.deepcopy(config)
        config_subconcept["run_name"] = config["run_name"] + "_subconcept"

        print("\nRunning Subconcepts CBM\n")
        metrics_subconcept = main(config_subconcept)
        subconcept_acc = metrics_subconcept.get('cbm_acc_detected', 0)

        reward = abs(cbm_acc - subconcept_acc)
        print("\n\nREWARD (CBM acc - Subconcept CBM acc): ", reward)

        update_reward_tracking(config, reward, concept_rewards, subset_rewards,
                               dist_rewards, prob_rewards)

        results.append({
            'config': config,
            'reward': reward,
            'cbm_metrics': metrics_cbm,
            'subconcept_metrics': metrics_subconcept,
            'cbm_acc': cbm_acc,
            'subconcept_acc': subconcept_acc
        })

        if trial % 20 == 0:
            best_reward = max(r['reward'] for r in results) if results else 0
            print(f"Trial {trial}: Best reward so far = {best_reward:.4f}")

    return sorted(results, key=lambda x: x['reward'], reverse=True)


def sample_random_config(sttngs, nonsalient_concepts, rng):
    """Sample random configuration using grid search logic"""
    nonsalient_concepts_combinations = powerset(nonsalient_concepts)[1:]
    feature_names = list(sttngs['logit_weights'].keys())
    num_concepts_in_model = len(feature_names)

    base_subconcepts = {
        'foot_shape_subtype': ['foot_shape_flat_trapezoid', 'foot_shape_flat_rounded',
                               'foot_shape_flat_square', 'foot_shape_flat_5sided',
                               'foot_shape_flat_lshaped', 'foot_shape_pointy_trapezoid',
                               'foot_shape_pointy_rounded', 'foot_shape_pointy_square',
                               'foot_shape_pointy_3sided', 'foot_shape_pointy_4sided'],
        'hand_shape_subtype': ["hand_shape_round_circle", "hand_shape_round_oval",
                               "hand_shape_round_oval2", "hand_shape_edgy_triangle",
                               "hand_shape_edgy_square", "hand_shape_edgy_trapezoid"]
    }
    all_subconcepts = [v for sublist in base_subconcepts.values() for v in sublist]

    # pick index randomly
    ind = rng.integers(0, len(nonsalient_concepts_combinations))
    nonsalient_concept_exp_tuple = nonsalient_concepts_combinations[ind]
    nonsalient_concept_exp = list(nonsalient_concept_exp_tuple)
    print("Chosen nonsalient concepts for experiment:", nonsalient_concept_exp)

    # randomly sample subconcept combinations using your existing function
    subset_combinations_per_concept = {concept: [] for concept in nonsalient_concept_exp}
    for concept in nonsalient_concept_exp:
        subconcept_combinations = get_subconcept_combinations(base_subconcepts[concept])
        subset_combinations_per_concept[concept] = subconcept_combinations

    all_combs = list(product(*subset_combinations_per_concept.values()))
    if not all_combs:
        return None

    ind = rng.integers(0, len(all_combs))
    subset = all_combs[ind]
    subconcepts_subset = tuple([item for subtuple in subset for item in subtuple])

    print("Chosen subconcepts for experiment:", subconcepts_subset)

    subconcept_distributions1_per_concept = {}
    subconcept_distributions2_per_concept = {}

    for concept in nonsalient_concept_exp:
        concept_nosubtype = concept.split("_")[0]
        subset_value1 = [sc for sc in subconcepts_subset if ("pointy" in sc or "round" in sc) and concept_nosubtype in sc]
        subset_value2 = [sc for sc in subconcepts_subset if ("flat" in sc or "edgy" in sc) and concept_nosubtype in sc]

        if len(subset_value1) == 0 or len(subset_value2) == 0:
            return None

        subconcept_distributions1 = get_N_floats_summing_to_0_5_with_min(len(subset_value1))
        subconcept_distributions2 = get_N_floats_summing_to_0_5_with_min(len(subset_value2))
        subconcept_distributions1_per_concept[concept] = subconcept_distributions1
        subconcept_distributions2_per_concept[concept] = subconcept_distributions2

    # randomly sample from distribution combinations
    all_dist1_combos = list(product(*subconcept_distributions1_per_concept.values()))
    all_dist2_combos = list(product(*subconcept_distributions2_per_concept.values()))

    if not all_dist1_combos or not all_dist2_combos:
        return None

    ind1 = rng.integers(0, len(all_dist1_combos))
    ind2 = rng.integers(0, len(all_dist2_combos))
    dist1 = all_dist1_combos[ind1]
    dist2 = all_dist2_combos[ind2]

    # randomly sample probability configuration
    prob_configs = []
    prob_count = 0
    for target_probs in get_monotonic_probs(num_concepts=num_concepts_in_model, lowest=0.01, step=0.05):
        prob_configs.append(target_probs)
        prob_count += 1
    print(f"Total probability configurations generated: {prob_count}")

    if not prob_configs:
        return None

    ind = rng.integers(0, len(prob_configs))
    target_probs = prob_configs[ind]

    # build configuration using existing logic
    subset_value1 = [sc for sc in subconcepts_subset if "pointy" in sc or "round" in sc]
    subset_value2 = [sc for sc in subconcepts_subset if "flat" in sc or "edgy" in sc]

    skew_list = []
    # flatten and normalize distributions
    dist1_flat = [val / len(nonsalient_concept_exp) for dist_tuple in dist1 for val in
                  (dist_tuple if isinstance(dist_tuple, (list, tuple)) else [dist_tuple])]
    dist2_flat = [val / len(nonsalient_concept_exp) for dist_tuple in dist2 for val in
                  (dist_tuple if isinstance(dist_tuple, (list, tuple)) else [dist_tuple])]

    for i, sc in enumerate(subset_value1):
        if i < len(dist1_flat):
            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist1_flat[i])})
    for i, sc in enumerate(subset_value2):
        if i < len(dist2_flat):
            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist2_flat[i])})

    print("Chosen skew list for experiment:", skew_list)

    labeling_func_details = find_params_for_target_probabilities(
        feature_names=feature_names, target_probs=target_probs
    )

    print("Chosen target probabilities for experiment:", target_probs)

    S = copy.deepcopy(sttngs)
    S["subconcepts"] = nonsalient_concept_exp
    S['logit_weights'] = labeling_func_details['logit_weights']
    S['logit_intercept'] = labeling_func_details['logit_intercept']
    S['logit_scalar'] = labeling_func_details['logit_scalar']
    S["skew_concept"] = skew_list

    drop_list = list(set(all_subconcepts) - set(subconcepts_subset))
    S["drop_concepts"] = drop_list + [concept.replace("_subtype", "") for concept in nonsalient_concept_exp]

    achieved_probs = {combo: float(stats['achieved'])
                      for combo, stats in labeling_func_details['verification'].items()}
    S["run_name"] = f"adaptive_{len(subconcepts_subset)}sc_" + "_".join([
        f"{''.join(map(str, k))}-{int(v * 100)}" for k, v in achieved_probs.items()
    ])

    S["_meta"] = {
        'concepts': tuple(nonsalient_concept_exp),
        'subset_size': len(subconcepts_subset),
        'subset_balance': len(subset_value1) / len(subconcepts_subset),
        'dist1_mean': np.mean(dist1_flat) if dist1_flat else 0,
        'dist2_mean': np.mean(dist2_flat) if dist2_flat else 0,
        'prob_range': max(target_probs.values()) - min(target_probs.values()),
        'prob_mean': np.mean(list(target_probs.values()))
    }

    return S


def sample_promising_config(sttngs, nonsalient_concepts, concept_rewards, subset_rewards, dist_rewards, prob_rewards, rng):
    """Sample from regions that have shown high rewards"""

    nonsalient_concepts_combinations = powerset(nonsalient_concepts)[1:]
    feature_names = list(sttngs['logit_weights'].keys())
    num_concepts_in_model = len(feature_names)

    base_subconcepts = {
        'foot_shape_subtype': ['foot_shape_flat_trapezoid', 'foot_shape_flat_rounded',
                               'foot_shape_flat_square', 'foot_shape_flat_5sided',
                               'foot_shape_flat_lshaped', 'foot_shape_pointy_trapezoid',
                               'foot_shape_pointy_rounded', 'foot_shape_pointy_square',
                               'foot_shape_pointy_3sided', 'foot_shape_pointy_4sided'],
        'hand_shape_subtype': ["hand_shape_round_circle", "hand_shape_round_oval",
                               "hand_shape_round_oval2", "hand_shape_edgy_triangle",
                               "hand_shape_edgy_square", "hand_shape_edgy_trapezoid"]
    }
    all_subconcepts = [v for sublist in base_subconcepts.values() for v in sublist]

    def weighted_choice(candidates, rewards_dict, key_func):
        """Choose from candidates based on learned rewards"""
        if not rewards_dict or not candidates:
            ind = rng.integers(0, len(candidates))
            return candidates[ind]

        # calculate scores for each candidate
        scores = []
        for candidate in candidates:
            key = key_func(candidate)
            if key in rewards_dict:
                scores.append(np.mean(rewards_dict[key]) + 0.1)
            else:
                scores.append(0.1)  # small default score for unexplored

        scores = np.array(scores)
        if scores.sum() == 0:
            ind = rng.integers(0, len(candidates))
            return candidates[ind]

        probs = scores / scores.sum()
        chosen_idx = rng.choice(len(candidates), p=probs)
        return candidates[chosen_idx]

    # choose concept combination based on learned rewards
    concept_candidates = list(nonsalient_concepts_combinations)

    if not concept_candidates:
        return None

    nonsalient_concept_exp_tuple = weighted_choice(
        concept_candidates,
        concept_rewards,
        key_func=lambda x: tuple(x)
    )
    nonsalient_concept_exp = list(nonsalient_concept_exp_tuple)

    # choose subconcept combinations, biased toward good subset sizes
    subset_combinations_per_concept = {concept: [] for concept in nonsalient_concept_exp}
    for concept in nonsalient_concept_exp:
        subconcept_combinations = get_subconcept_combinations(base_subconcepts[concept])
        subset_combinations_per_concept[concept] = subconcept_combinations

    all_combs = list(product(*subset_combinations_per_concept.values()))
    if not all_combs:
        return None

    # filter combinations by promising subset sizes
    def get_subset_size(comb):
        subset_flat = tuple([item for subtuple in comb for item in subtuple])
        return len(subset_flat)

    subset = weighted_choice(
        all_combs,
        subset_rewards,
        key_func=get_subset_size
    )
    subset_flat = tuple([item for subtuple in subset for item in subtuple])

    # choose distributions biased toward promising distribution characteristics
    subconcept_distributions1_per_concept = {}
    subconcept_distributions2_per_concept = {}

    for concept in nonsalient_concept_exp:
        subset_value1 = [sc for sc in subset_flat if "pointy" in sc or "round" in sc]
        subset_value2 = [sc for sc in subset_flat if "flat" in sc or "edgy" in sc]

        if len(subset_value1) == 0 or len(subset_value2) == 0:
            return None

        subconcept_distributions1 = get_N_floats_summing_to_0_5_with_min(len(subset_value1))
        subconcept_distributions2 = get_N_floats_summing_to_0_5_with_min(len(subset_value2))
        subconcept_distributions1_per_concept[concept] = subconcept_distributions1
        subconcept_distributions2_per_concept[concept] = subconcept_distributions2

    all_dist1_combos = list(product(*subconcept_distributions1_per_concept.values()))
    all_dist2_combos = list(product(*subconcept_distributions2_per_concept.values()))

    if not all_dist1_combos or not all_dist2_combos:
        return None

    # choose distributions based on learned preferences for distribution means
    def get_dist_mean(dist_combo):
        flat_dist = [val for dist_tuple in dist_combo for val in
                     (dist_tuple if isinstance(dist_tuple, (list, tuple)) else [dist_tuple])]
        return round(np.mean(flat_dist), 2) if flat_dist else 0

    dist1 = weighted_choice(
        all_dist1_combos,
        dist_rewards,
        key_func=lambda d: f'dist1_mean_{get_dist_mean(d)}'
    )

    dist2 = weighted_choice(
        all_dist2_combos,
        dist_rewards,
        key_func=lambda d: f'dist2_mean_{get_dist_mean(d)}'
    )

    # choose probability configuration based on learned preferences
    prob_configs = []
    prob_count = 0
    for target_probs in get_monotonic_probs(num_concepts=num_concepts_in_model,
                                              lowest=0.01, step=0.05):
        prob_configs.append(target_probs)
        prob_count += 1
        if prob_count >= 50:
            break

    if not prob_configs:
        return None

    # choose probability config based on learned preferences for prob range
    def get_prob_range(probs):
        values = list(probs.values())
        return round(max(values) - min(values), 1)

    target_probs = weighted_choice(
        prob_configs,
        prob_rewards,
        key_func=lambda p: f'prob_range_{get_prob_range(p)}'
    )

    # build final configuration
    subset_value1 = [sc for sc in subset_flat if "pointy" in sc or "round" in sc]
    subset_value2 = [sc for sc in subset_flat if "flat" in sc or "edgy" in sc]

    skew_list = []
    dist1_flat = [val / len(nonsalient_concept_exp) for dist_tuple in dist1 for val in
                  (dist_tuple if isinstance(dist_tuple, (list, tuple)) else [dist_tuple])]
    dist2_flat = [val / len(nonsalient_concept_exp) for dist_tuple in dist2 for val in
                  (dist_tuple if isinstance(dist_tuple, (list, tuple)) else [dist_tuple])]

    for i, sc in enumerate(subset_value1):
        if i < len(dist1_flat):
            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist1_flat[i])})
    for i, sc in enumerate(subset_value2):
        if i < len(dist2_flat):
            skew_list.append({'concepts': {sc: 1}, 'min_fraction': float(dist2_flat[i])})

    labeling_func_details = find_params_for_target_probabilities(
        feature_names=feature_names, target_probs=target_probs
    )

    S = copy.deepcopy(sttngs)
    S["subconcepts"] = nonsalient_concept_exp
    S['logit_weights'] = labeling_func_details['logit_weights']
    S['logit_intercept'] = labeling_func_details['logit_intercept']
    S['logit_scalar'] = labeling_func_details['logit_scalar']
    S["skew_concept"] = skew_list

    drop_list = list(set(all_subconcepts) - set(subset_flat))
    S["drop_concepts"] = drop_list + [concept.replace("_subtype", "") for concept in nonsalient_concept_exp]

    achieved_probs = {combo: float(stats['achieved'])
                      for combo, stats in labeling_func_details['verification'].items()}
    S["run_name"] = f"promising_{len(subset_flat)}sc_" + "_".join([
        f"{''.join(map(str, k))}-{int(v * 100)}" for k, v in achieved_probs.items()
    ])

    S["_meta"] = {
        'concepts': tuple(nonsalient_concept_exp),
        'subset_size': len(subset_flat),
        'subset_balance': len(subset_value1) / len(subset_flat),
        'dist1_mean': np.mean(dist1_flat) if dist1_flat else 0,
        'dist2_mean': np.mean(dist2_flat) if dist2_flat else 0,
        'prob_range': max(target_probs.values()) - min(target_probs.values()),
        'prob_mean': np.mean(list(target_probs.values()))
    }

    return S


def update_reward_tracking(config, reward, concept_rewards, subset_rewards, dist_rewards, prob_rewards):
    """Update reward tracking based on config performance"""

    meta = config.get("_meta", {})

    concepts = meta.get('concepts')
    if concepts:
        concept_rewards[concepts].append(reward)

    subset_size = meta.get('subset_size')
    if subset_size:
        subset_rewards[subset_size].append(reward)

    subset_balance = meta.get('subset_balance')
    if subset_balance:
        balance_bin = round(subset_balance, 1)
        subset_rewards[f'balance_{balance_bin}'].append(reward)

    # Track rewards by distribution characteristics
    dist1_mean = meta.get('dist1_mean')
    if dist1_mean:
        # Discretize into bins
        dist_bin = round(dist1_mean, 2)
        dist_rewards[f'dist1_mean_{dist_bin}'].append(reward)

    dist2_mean = meta.get('dist2_mean')
    if dist2_mean:
        dist_bin = round(dist2_mean, 2)
        dist_rewards[f'dist2_mean_{dist_bin}'].append(reward)

    # Track rewards by probability characteristics
    prob_range = meta.get('prob_range')
    if prob_range:
        range_bin = round(prob_range, 1)
        prob_rewards[f'prob_range_{range_bin}'].append(reward)

    prob_mean = meta.get('prob_mean')
    if prob_mean:
        mean_bin = round(prob_mean, 1)
        prob_rewards[f'prob_mean_{mean_bin}'].append(reward)

    # Keep only recent rewards to adapt to changing patterns
    max_history = 50
    for rewards_dict in [concept_rewards, subset_rewards, dist_rewards, prob_rewards]:
        for key in list(rewards_dict.keys()):
            if len(rewards_dict[key]) > max_history:
                rewards_dict[key] = rewards_dict[key][-max_history:]

#principled_adaptive_search(settings, nonsalient_concepts=['foot_shape_subtype', 'hand_shape_subtype'])
