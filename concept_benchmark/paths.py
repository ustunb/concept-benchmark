"""
This file defines paths for key directories and files. Contents include:
1. Directory Names: Path objects that specify the directories where we store code, data, results, etc.
2. File Name Generators: functions used to programatically name processed datasets, results, graphs etc.
"""

from pathlib import Path

# Directories

# path to the GitHub repository
repo_dir = Path(__file__).resolve().parent.parent

# path to the Python package
pkg_dir = repo_dir / "concept_benchmark/"

# directory where we store datasets
data_dir = repo_dir / "data/"

# path to the Python package
tests_dir = repo_dir / "tests/"

# directory where we store results
results_dir = repo_dir / "results/"


def get_dataset_dir(data_name: str, **kwargs) -> Path:
    """ """
    p = data_dir / data_name

    if "data_type" in kwargs:
        p = p / kwargs["data_type"]

    return p


def get_noisyconcept_data(
    concept_noise_probs: list,
    parity_inds: list,
    coefficients: list,
    intercept: float,
) -> str:
    """
    Generate a name for the noisy concept dataset based on its parameters.

    Args:
        concept_noise_probs (list): List of noise probabilities for each concept.
        parity_inds (list): List of parity indices for the concepts.
        coefs (list): List of coefficients for the logistic regression model.
        intercept (float): Intercept for the logistic regression model.

    Returns:
        str: A formatted string representing the dataset name.
    """
    concept_noise_prob_str = ",".join([str(cp) for cp in concept_noise_probs])
    parity_str = "-".join([",".join([str(v) for v in par]) for par in parity_inds])
    coefs_str = ",".join([str(c) for c in coefficients])

    f = f"p{concept_noise_prob_str}_p{parity_str}_c{coefs_str}_i{intercept}.data"

    return get_dataset_dir("noisyconcepts") / f
