API Reference
=============

Dataset Generator
-----------------

.. autoclass:: concept_benchmark.DatasetGenerator
   :members:

Data Containers
---------------

.. autoclass:: concept_benchmark.data.ConceptDataset
   :members:

.. autoclass:: concept_benchmark.data.ConceptDatasetSample
   :members:

.. autoclass:: concept_benchmark.data.ConceptImageDatasetSample
   :members:

Benchmark Configurations
------------------------

.. autoclass:: concept_benchmark.config.RobotBenchmarkConfig
   :members:

.. autoclass:: concept_benchmark.config.SudokuBenchmarkConfig
   :members:


Models (experiments/)
---------------------

.. note::
   These classes require cloning the repository. They are not included in
   ``pip install concept-benchmark``.

.. autoclass:: experiments.models.ConceptDetector
   :members: fit, predict, save, load, to

.. autoclass:: experiments.models.FrontEndModel
   :members: fit, predict, predict_proba

.. autoclass:: experiments.models.ConceptBasedModel
   :members: predict, predict_proba

.. autoclass:: experiments.models.RobotConceptClassifier

.. autoclass:: experiments.models.GroupPoolingConceptSudokuCNN

Interventions (experiments/)
----------------------------

.. autoclass:: experiments.intervention.ConceptInterventionRunner
   :members: run, prepare

.. autoclass:: experiments.intervention.InterventionConfig
   :members:

.. autoclass:: experiments.intervention.InterventionStrategy
   :members: prepare, propose

.. autoclass:: experiments.intervention.InterventionBatch
   :members:

.. autoclass:: experiments.intervention.StrategyProposal
   :members:

.. autoclass:: experiments.intervention.InterventionResult
   :members:

.. autoclass:: experiments.kflip.KFlipInterventionStrategy
   :members: propose

.. autoclass:: experiments.intervention.ConceptualSafeguardsStrategy
   :members: propose

.. autoclass:: experiments.intervention.OrderedCBMStrategy
   :members: prepare, propose

.. autoclass:: experiments.intervention.RandomInterventionStrategy
   :members: propose

.. autoclass:: experiments.intervention.ScoreIntervention
   :members: propose

Utilities (experiments/)
------------------------

.. note::
   These functions require cloning the repository. They are not included in
   ``pip install concept-benchmark``.

.. autofunction:: experiments.evaluate.train_cbm

.. autofunction:: experiments.evaluate.train_dnn

.. autofunction:: experiments.evaluate.intervention_table

.. autofunction:: experiments.evaluate.automation_table

.. autofunction:: experiments.evaluate.coverage_at_target

.. autofunction:: experiments.evaluate.predict_labels

.. autofunction:: experiments.evaluate.predict_proba_positive

.. autofunction:: experiments.utils.run_alignment

.. autofunction:: concept_benchmark.utils.determine_device

.. autofunction:: concept_benchmark.utils.compute_accuracy

.. autofunction:: concept_benchmark.utils.get_loader_config

.. autofunction:: concept_benchmark.utils.set_deterministic_seed
