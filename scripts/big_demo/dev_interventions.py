from concept_benchmark.models import ConceptBasedModel
from concept_benchmark.ext.fileutils import load
from concept_benchmark.intervention import ConceptInterventionRunner, InterventionConfig, ConceptualSafeguardsStrategy, ScoreIntervention
from utils import get_dataset_file, get_model_file, determine_device

settings = {
    # 'data_name': 'robot',
    # 'data_type': 'image',
    # 'n': 1,
    'data_name': 'sudoku',
    'data_type': 'tabular',
    'n': 3,
    'max_corrupt': 21,
    'concept_noise': 0.15,
    'concept_missing': 0.00,
    'concept_missing_mech': 'none',
    'target_accuracy': 1.0, # doesn't matter but need for dataset loading
    'epochs': 50,
    'patience': 20,
}


device = determine_device()

fe = load(get_model_file(model_type="fe", **settings))
cd = load(get_model_file(model_type="cd", **settings))

data = load(get_dataset_file(**settings))
data.split(fold_id="K05N01", fold_num_validation=4, fold_num_test=5)

# CS intervention sample
cs = ConceptBasedModel(
    concept_detector=cd,
    front_end_model=fe,
    propagate=True,
    # mc_max_samples=
    mc_mode='mc',
)


runner = ConceptInterventionRunner(model=cs)
config = InterventionConfig(tau=0.05)
strategy = ConceptualSafeguardsStrategy()
result = runner.run(
    strategy=strategy,
    config=config,
    dataset=data.test,
)

result.strat_metrics

# ScoreIntervention example
cbm = ConceptBasedModel(
    concept_detector=cd,
    front_end_model=fe
)

runner = ConceptInterventionRunner(model=cbm)
config = InterventionConfig(score_threshold=0.3, max_concepts_per_instance=25)
strategy = ScoreIntervention()
score_result = runner.run(
    strategy=strategy,
    config=config,
    dataset=data.test
)

score_result.proposal.details