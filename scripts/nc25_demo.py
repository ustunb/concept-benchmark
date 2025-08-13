from concept_benchmark.paths import get_noisyconcept_data, results_dir
from concept_benchmark.ext import fileutils
from concept_benchmark.models import ConceptBasedModel
from concept_benchmark.metrics import calc_metric

import subprocess
import rich
import numpy as np
import matplotlib.pyplot as plt
from rich.panel import Panel

settings = {
    "concept_noise_probs": [0.25] * 3,
    "parity_inds": [[0, 1, 3], [0, 1, 2], [0, 1, 4]],
    "coefficients": [1., 2., 3.],
    "intercept": -2.,
}

if __name__ == "__main__":
    gen_dataset_cmd = "python scripts/generate_noisyconepts.py"

    rich.print(Panel(f"[bold]{gen_dataset_cmd}[/bold]"))
    subprocess.run(gen_dataset_cmd, shell=True, check=True)

    data = fileutils.load(get_noisyconcept_data(**settings))
    data.split('K05N01', fold_num_validation=4, fold_num_test=5)

    mdl = ConceptBasedModel(propagate=True)
    mdl.fit(data.training, data.validation, fit_params={"batch_size": 1024, "epochs": 20})

    fileutils.save(mdl, results_dir / 'noisyconcepts25_calibrated.model')

    no_prop_pr = mdl.predict_proba(data.test, propagate=False)
    prop_pr = mdl.predict_proba(data.test, propagate=True)

    no_prop_results = []
    for tau in np.linspace(0, 1, 50):
        no_prop_metrics = calc_metric(no_prop_pr[:, 1], data.test.y, tau=tau)
        no_prop_results.append(no_prop_metrics)
        if no_prop_metrics['coverage'] >= 1:
            break

    prop_results = []
    for tau in np.linspace(0, 1, 50):
        prop_metrics = calc_metric(prop_pr[:, 1], data.test.y, tau=tau)
        prop_results.append(prop_metrics)
        if prop_metrics['coverage'] >= 1:
            break

    no_prop_coverage = [r['coverage'] for r in no_prop_results]
    no_prop_selective_accuracy = [r['selective_accuracy'] for r in no_prop_results]

    prop_coverage = [r['coverage'] for r in prop_results]
    prop_selective_accuracy = [r['selective_accuracy'] for r in prop_results]

    fig = plt.figure(figsize=(8, 6))

    # Plotting the results
    plt.plot(no_prop_coverage, no_prop_selective_accuracy, label='No Propagation', color='blue', marker='o', markersize=3)
    plt.plot(prop_coverage, prop_selective_accuracy, label='Propagation', color='orange', marker='o', markersize=3)
    plt.xlabel('Coverage')
    plt.ylabel('Selective Accuracy')
    plt.title('NoisyConcepts25: Selective Accuracy vs Coverage')
    plt.xlim(-0.005, 1.005)
    plt.ylim(0.5, 1.005)
    plt.yticks(np.arange(0.5, 1.05, 0.1))
    plt.xticks(np.arange(0, 1.05, 0.2))
    plt.grid()
    plt.legend()

    fig.savefig(results_dir / 'noisyconcepts_coverage_vs_selective_accuracy.png', dpi=300, bbox_inches='tight')