import numpy as np


def test_concept_detector_invariance_point(concept_detector, concept, concept_names, point, dataset, device):
    """
    Test if the concept_detector for concept is invariant to changes it other features.

    Check that the concept prediction for one point will not change over other counterfactual points that share the same
    concept value.

    :param concept_detector:
    :param concept:
    :param concept_names:
    :param point:
    :param dataset:
    :param device:
    :return:
    """
    concept_idx = concept_names.index(concept)
    concept_value = dataset.C[point, concept_idx]
    if "foot_shape" in concept:
        foot_type = 1 if concept.split("_")[2] == "pointy" else 0
        foot_subtype = concept.split("_")[3]
    # Get all points that share the same concept value
    dataset_indices = dataset.meta["df_indices"]
    subcatalog = dataset.meta["catalog_df"].iloc[dataset_indices]
    matching_points = np.where(dataset.C[:, concept_idx] == concept_value)[0] if "foot_shape" not in concept else \
        np.where((subcatalog["foot_shape"] == foot_type) & (subcatalog["foot_shape_subtype"] == foot_subtype))[0]
    # Get the dataset with just the original point
    dataset_point = dataset.filter(np.array([i == point for i in range(len(dataset))]))
    original_pred = concept_detector.predict(dataset_point, embed_params={"device": device})[0, concept_idx]
    print("Original prediction for point", point, "concept", concept, "value", concept_value, "is", original_pred)
    # count the number of matching points that have different predictions
    variant_points = []
    dataset_mp = dataset.filter(np.array([i in matching_points for i in range(len(dataset))]))
    cf_preds = concept_detector.predict(dataset_mp, embed_params={"device": device})
    for i, mp in enumerate(matching_points):
        if mp == point:
            continue
        cf_pred = cf_preds[i, concept_idx]
        if (original_pred >= 0.5) != (cf_pred >= 0.5):
            print("Error on matching point", mp, "concept", concept, "value", concept_value, "prediction", cf_pred)
            variant_points.append(mp)

    if len(variant_points) > 0:
        print(f"Concept detector for concept '{concept}' is NOT invariant for point {point} with concept value {concept_value}.")
        return False
    return True


def test_concept_detector_invariance(concept_detector, concept_to_test, concept_names, dataset, device, num_tests=10):
    """
    Test concept detector invariance for all concepts on random points in the dataset.

    :param concept_detector:
    :param concept_to_test:
    :param concept_names:
    :param dataset:
    :param num_tests:
    :return:
    """
    rng = np.random.default_rng(12345)
    n_samples = len(dataset)
    all_passed = True
    for _ in range(num_tests):
        point = rng.integers(0, n_samples)
        passed = test_concept_detector_invariance_point(concept_detector, concept_to_test, concept_names, point, dataset, device)
        if not passed:
            all_passed = False
    return all_passed
