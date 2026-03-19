concept-benchmark
=================

**Concept Benchmark** is a Python package for generating synthetic datasets to benchmark
`concept bottleneck models <https://arxiv.org/abs/2007.04612>`_ (CBMs).
It provides datasets with fully-specified ground-truth concept labels, letting you vary
concept granularity, annotation quality, and the labeling rule — then measure exactly how
each factor affects model performance and the value of interventions.

The package includes two benchmarks:

- **Robot Classification** — a decision-support task where a human corrects concept predictions to improve accuracy. Available as image and text modalities.
- **Sudoku Validation** — an automation task where the model handles routine cases and defers uncertain ones. Demonstrates selective classification and AND-fragility of concepts.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   benchmark_your_model

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   robot
   sudoku

.. toctree::
   :maxdepth: 2
   :caption: Evaluation

   interventions
   alignment

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api
