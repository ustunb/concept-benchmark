"""
Generate datasets by as specified by the parameters concept_noise_probabilities, parity_inds, coefficients, intercept
"""
import argparse
import itertools
import os
import multiprocessing
import numpy as np
import shutil
import math

from numpy import ndarray
from tqdm import tqdm
from typing import List, Optional

from concept_benchmark.data import ConceptDataset
from concept_benchmark.paths import get_dataset_dir, get_noisyconcept_data
from concept_benchmark.ext import fileutils

NC_FOLDER = get_dataset_dir("noisyconcepts")
N_CONCEPTS = 3
PROB_PRECISION = 2

settings = {
    "concept_noise_probabilities": [
            np.repeat(round(v, PROB_PRECISION), N_CONCEPTS).tolist() 
            for v in np.arange(0., 1., 0.05)
        ],
    "x_probs": [.7] * 5,
    "n_samples": 100000,
    "master_seed": 42,
    "parity_inds": [[[0, 1, 3], [0, 1, 2], [0, 1, 4]]],
    "coefficients": [[1., 2., 3.]],
    "intercept": [-2.],
}


def bernoulli(p: float, rng: Optional[np.random.Generator] = None) -> bool:
    """Perform Bernoulli trial with probability p.

    Args:
        p (float): The probability of success.
        rng (Optional[np.random.Generator], optional): A random number generator. Defaults to None.

    Returns:
        bool: The outcome of the trial.
    """
    if rng:
        return rng.random() < p
    return np.random.random() < p


def parity(lst: List[int]) -> int:
    """True if the sum of the bits is odd.

    Args:
        lst (List[int]): A list of binary bits.

    Returns:
        int: The parity of the list.
    """
    return int(sum(lst) % 2)


def logistic(x: float) -> float:
    """Sigmoid/logistic function.

    Args:
        x (float): The input value.

    Returns:
        float: The value of the logistic function.
    """
    return 1 / (1 + math.exp(-x))


def compute_c_i_proba(x: ndarray, p_noise: float, p_inds: List[int]) -> float:
    """Compute the probability of a concept being true.

    Args:
        x (ndarray): The input features.
        p_noise (float): The probability of noise.
        p_inds (List[int]): The parity indices.

    Returns:
        float: The probability of the concept being true.
    """
    # With probability p_noise, the concept is random (0 or 1 with equal likelihood).
    cp = 0.5 * p_noise
    # With probability 1 - p_noise, the concept is the parity of the selected features.
    cp += parity(x[p_inds]) * (1 - p_noise)
    return cp

def compute_prob_y_given_c(c: List[int], coefs: List[float], intcpt: float) -> float:
    """Computes the probability of y given c.

    Args:
        c (List[int]): The concept vector.
        coefs (List[float]): The coefficients.
        intcpt (float): The intercept.

    Returns:
        float: The probability of y given c.
    """
    return logistic(sum([coef * c_i for coef, c_i in zip(coefs, c)]) + intcpt)


def compute_prob_ys_given_C(C: ndarray, coefs: List[float], intcpt: float) -> ndarray:
    """Computes the probabilities of y given C.

    Args:
        C (ndarray): The concept matrix.
        coefs (List[float]): The coefficients.
        intcpt (float): The intercept.

    Returns:
        ndarray: The probabilities of y given C.
    """
    n_samples = C.shape[0]
    return np.array(
        [compute_prob_y_given_c(c=C[i, :], coefs=coefs, intcpt=intcpt) for i in range(n_samples)])

def generate_all_datasets(
    n: int,
    xp: List[float],
    master_seed: int,
    overwrite=False,
    num_workers=1,
    **kwargs
) -> List[ConceptDataset]:
    """Generate datasets using all combinations of the parameters in kwargs.

    This function creates a directory for each dataset, generates the data, and saves it.
    It uses a master seed to generate child seeds for each dataset, ensuring reproducibility.

    Args:
        n (int): Number of samples to generate.
        xp (List[float]): Probabilities of x values.
        master_seed (int): The master seed for generating child seeds.
        overwrite (bool, optional): If True, overwrite existing datasets. Defaults to False.
        num_workers (int, optional): Number of worker processes to use. Defaults to 1.
        **kwargs: Keyword arguments for dataset parameters.

    Returns:
        List[ConceptDataset]: A list of generated datasets.
    """
    assert os.path.exists(NC_FOLDER), f"{NC_FOLDER} does not exist"

    if overwrite and os.path.exists(NC_FOLDER):
        print(f'Overwriting datasets in {NC_FOLDER}')
        shutil.rmtree(NC_FOLDER)
    os.makedirs(NC_FOLDER, exist_ok=True)

    # Create a random number generator for reproducible X
    rng = np.random.default_rng(master_seed)
    X = np.array([[float(bernoulli(p, rng)) for p in xp] for _ in range(n)])

    params_dict = dict(kwargs)
    params_combinations = list(itertools.product(*params_dict.values()))
    num_datasets = len(params_combinations)

    # Generate child seeds for each dataset
    child_seeds = rng.integers(low=0, high=2**32 - 1, size=num_datasets)

    # Save seeds to a file
    with open(os.path.join(NC_FOLDER, 'seeds.txt'), 'w') as f:
        for seed in child_seeds:
            f.write(f"{seed}\n")

    datasets = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        tasks = []
        for i, params in enumerate(params_combinations):
            task_kwargs = dict(zip(params_dict.keys(), params))
            task_kwargs['seed'] = child_seeds[i]
            task = pool.apply_async(
                create_dataset,
                args=[X],
                kwds={**task_kwargs, 'regenerate_dataset': overwrite}
            )
            tasks.append(task)

        for task in tqdm(tasks):
            datasets.append(task.get())

    print(f'{len(datasets)} datasets created, saved in {NC_FOLDER}')
    return datasets


def create_dataset(
    X: np.ndarray,
    concept_noise_probs: List[float],
    parity_inds: List[List[int]],
    coefs: List[float],
    intercept: float,
    seed: int,
    regenerate_dataset=False
) -> ConceptDataset:
    """Creates a single concept dataset.

    Args:
        X (np.ndarray): The input features.
        data_dir (str): The directory to save the dataset in.
        concept_noise_probs (List[float]): Probabilities of noise for each concept.
        parity_inds (List[List[int]]): Parity indices for each concept.
        coefs (List[float]): Coefficients for the logistic regression model.
        intercept (float): Intercept for the logistic regression model.
        seed (int): The random seed for data generation.
        regenerate_dataset (bool, optional): If True, regenerate the dataset even if it exists. Defaults to False.

    Returns:
        ConceptDataset: The generated dataset.
    """
    rng = np.random.default_rng(seed)

    n_concepts = len(concept_noise_probs)
    assert n_concepts == len(parity_inds) == len(coefs)

    dataset_path = get_noisyconcept_data(
        concept_noise_probs=concept_noise_probs,
        parity_inds=parity_inds,
        coefficients=coefs,
        intercept=intercept
    )

    if regenerate_dataset or not os.path.exists(dataset_path):
        C_probas = np.array([[compute_c_i_proba(x, cp_noise, p_inds) for x in X] for
                             cp_noise, p_inds in
                             zip(concept_noise_probs, parity_inds)]).T

        C_set = np.array([[bernoulli(cp, rng) for cp in c_proba] for c_proba in C_probas])
        y_proba_set = compute_prob_ys_given_C(C_set, coefs, intercept)
        y_set = np.array([bernoulli(yp, rng) for yp in y_proba_set])

        meta = {
            'classes': [0, 1],
            'concepts': [f'c{i+1}' for i in range(n_concepts)],
            'data_type': 'numeric',
            'y_proba': y_proba_set,
            'concept_noise_probs': concept_noise_probs,
            'parity_inds': parity_inds,
            'coefs': coefs,
            'intercept': intercept,
        }

        dataset_obj = ConceptDataset(
            X=X,
            C=C_set,
            y=y_set,
            meta=meta,
        )
        dataset_obj.generate_cvindices(strata=dataset_obj.y, seed=seed)

        fileutils.save(dataset_obj, dataset_path, msg=False, overwrite=True)

    dataset_obj = fileutils.load(dataset_path)

    return dataset_obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=settings['n_samples'], help='number of samples to generate')
    parser.add_argument('--x_probs', nargs='+', type=float, default=settings['x_probs'], help='probabilities of x values')
    parser.add_argument('--overwrite', action='store_true', help='force the creation of new datasets')
    parser.add_argument('--master_seed', type=int, default=settings['master_seed'], help='Master seed for reproducibility')
    parser.add_argument('--concept-noise-probabilities', nargs='+', type=float,
                        default=settings["concept_noise_probabilities"],
                        help='probabilities of noise for each of the concepts')
    parser.add_argument('--parity_inds', nargs='+', type=int, default=settings['parity_inds'],
                        help='parity indices')
    parser.add_argument('--coefficients', nargs='+', type=float, default=settings['coefficients'], help='coefficients')
    parser.add_argument('--intercept', nargs='+', type=float, default=settings['intercept'], help='intercept')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count() - 1, help='number of workers')

    args = parser.parse_args()

    # Convert concept noise probabilities to a list of lists if they are not already
    concept_noise_probabilities = args.concept_noise_probabilities
    if isinstance(concept_noise_probabilities[0], float):
        concept_noise_probabilities = [[cp] * len(args.coefficients[0]) for cp in concept_noise_probabilities]

    generate_all_datasets(
        n=args.n_samples,
        xp=args.x_probs,
        overwrite=args.overwrite,
        master_seed=args.master_seed,
        concept_noise_probs=concept_noise_probabilities,
        parity_inds=args.parity_inds,
        coefs=args.coefficients,
        intercept=args.intercept,
        num_workers=args.num_workers
    )

# Unused functions
# def update_c_i_probas(prob_cs: List[float], dependence: List[List[int]]):
#     prob_cs = prob_cs.copy()
#     dependence = np.array(dependence)
#     assert len(dependence) == len(prob_cs), "dependence must be a square matrix of size length x length"
#     for i in range(len(prob_cs)):
#         for j in range(len(prob_cs)):
#             if i != j and dependence[i, j]:
#                 # if dependence from concept c_j to c_i, replace c_i with p(parity(c_i, c_j))
#                 # note that p(parity(c_i, c_j)) = p(c_i)*(1-p(c_j)) + (1-p(c_i))*p(c_j)
#                 prob_cs[i] = prob_cs[i] * (1 - prob_cs[j]) + (1 - prob_cs[i]) * prob_cs[j]
#     return prob_cs


# def compute_joint_c_given_x(
#     c_vec: List[int],
#     x: ndarray,
#     p_noise: List[float],
#     p_inds: List[List[int]],
#     dependence: Optional[List[List[int]]] = None
# ) -> float:
#     """Compute the probability of a concept vector given x.

#     Args:
#         c_vec (List[int]): The concept vector.
#         x (ndarray): The input features.
#         p_noise (List[float]): The probabilities of noise for each concept.
#         p_inds (List[List[int]]): The parity indices for each concept.
#         dependence (Optional[List[List[int]]], optional): The dependence matrix. Defaults to None.

#     Returns:
#         float: The probability of the concept vector.
#     """
#     p_c_vec = 1.0
#     p_cs = []
#     for i, c in enumerate(c_vec):
#         p_c = compute_c_i_proba(x, p_noise[i], p_inds[i])
#         p_cs.append(p_c)
#     if dependence is not None:
#         p_cs = update_c_i_probas(p_cs, dependence)
#     for i, c in enumerate(c_vec):
#         if c == 1:
#             p_c_vec *= p_cs[i]
#         else:
#             p_c_vec *= 1 - p_cs[i]
#     return p_c_vec


# def compute_concept(
#     X: ndarray,
#     p_noise: float,
#     p_inds: List[int],
#     rng: Optional[np.random.Generator] = None
# ) -> ndarray:
#     """Features to concepts function.

#     Args:
#         X (ndarray): The input features.
#         p_noise (float): The probability of noise.
#         p_inds (List[int]): The parity indices.
#         rng (Optional[np.random.Generator], optional): A random number generator. Defaults to None.

#     Returns:
#         ndarray: The generated concepts.
#     """
#     c = np.array([bernoulli(compute_c_i_proba(x, p_noise, p_inds), rng) for x in X])
#     return c

# def introduce_dependence(
#     C_probs: List[ndarray],
#     p_dependence: float,
#     rng: Optional[np.random.Generator] = None
# ) -> List[ndarray]:
#     """Introduces dependence between concepts.

#     Args:
#         C_probs (List[ndarray]): A list of concept probabilities arrays.
#         p_dependence (float): The probability of dependence.
#         rng (Optional[np.random.Generator], optional): A random number generator. Defaults to None.

#     Returns:
#         List[ndarray]: A list of concept probabilities arrays with dependence.
#     """
#     C_dep = C_probs.copy()
#     if p_dependence > 0:
#         D = [bernoulli(0.7, rng) for _ in range(len(C_probs[0]))]
#         for c_i in range(len(C_dep)):
#             C_dep[c_i] = np.array([min(cp_i + (p_dependence * d_i), 1.0)
#                                    for cp_i, d_i in zip(C_probs[c_i], D)])
#     return C_dep


# def compute_prob_y_given_X(
#     X: ndarray,
#     coefs: List[float],
#     intcpt: float,
#     concept_noise_probs: List[float],
#     parity_inds: List[List[int]],
#     dependence: Optional[List[List[int]]] = None
# ) -> ndarray:
#     """Computes the probability of y given X.

#     Args:
#         X (ndarray): The input features.
#         coefs (List[float]): The coefficients.
#         intcpt (float): The intercept.
#         concept_noise_probs (List[float]): The probabilities of noise for each concept.
#         parity_inds (List[List[int]]): The parity indices for each concept.
#         dependence (Optional[List[List[int]]], optional): The dependence matrix. Defaults to None.

#     Returns:
#         ndarray: The probabilities of y given X.
#     """
#     n_samples = X.shape[0]
#     n_concepts = len(concept_noise_probs)
#     all_cs = generate_all_binary_vectors(n_concepts)
#     prob = np.zeros(n_samples)
#     for i in range(n_samples):
#         total_c_probs = []
#         for c in all_cs:
#             p_c = compute_joint_c_given_x(c, X[i], concept_noise_probs, parity_inds, dependence)
#             total_c_probs.append(p_c)
#             p_y_given_c = compute_prob_y_given_c(c, coefs, intcpt)
#             prob[i] += p_c * p_y_given_c
#         assert_almost_equal(sum(total_c_probs), 1.0)
#     return prob


# def compute_y_given_X(
#     X: ndarray,
#     coefs: List[float],
#     intcpt: float,
#     concept_noise_probs: List[float],
#     parity_inds: List[List[int]],
#     dependence: Optional[List[List[int]]] = None,
#     rng: Optional[np.random.Generator] = None
# ) -> ndarray:
#     """Computes y given X.

#     Args:
#         X (ndarray): The input features.
#         coefs (List[float]): The coefficients.
#         intcpt (float): The intercept.
#         concept_noise_probs (List[float]): The probabilities of noise for each concept.
#         parity_inds (List[List[int]]): The parity indices for each concept.
#         dependence (Optional[List[List[int]]], optional): The dependence matrix. Defaults to None.
#         rng (Optional[np.random.Generator], optional): A random number generator. Defaults to None.

#     Returns:
#         ndarray: The computed y values.
#     """
#     prob = compute_prob_y_given_X(X, coefs, intcpt, concept_noise_probs, parity_inds, dependence)
#     return np.array([bernoulli(p, rng) for p in prob])

# def generate_all_binary_vectors(n: int) -> List[List[int]]:
#     """Generates all binary vectors of length n.

#     Args:
#         n (int): The length of the binary vectors.

#     Returns:
#         List[List[int]]: A list of all binary vectors.
#     """
#     return [list(p) for p in product([0, 1], repeat=n)]