# Implementation Plan: Berk-Branch Refactor Adoption

## Context
Adopt 5 changes from `blm/slm/external/cbm/` into `concept-benchmark` without regressing paper numbers (robot seed=1014, sudoku seed=171). Full design rationale lives at `docs/berk_refactor_plan.md`; this file is the execution roadmap.

## Audit corrections to the design plan

1. **Venv path:** `.venv/bin/python` (not `./venv/bin/python` as MEMORY.md says). All commands must use the dot-prefix form.
2. **Test count:** 368 collected (not 83). MEMORY.md was stale.
3. **No sudoku regression test exists.** Only `test_robot_pipeline_regression.py` is automated; sudoku numbers require manual `scripts/sudoku_pipeline.py` runs gated on `tests/data/` reference artifacts that don't exist.
4. **Change 1c (`classes` tuple recoding) value is weaker than claimed** — grep found zero `2*y - 1` / `np.where(y==0,-1,1)` callsites. Downgrade to "store `classes` as metadata; do not recode `y`."
5. **Pickled-model backward-compat shim** at `test_robot_pipeline_regression.py:21-26` reroutes `concept_benchmark.models → experiments.models`. Change 5 risks breaking this if synthetic/ submodule renames affect pickled object module paths.

## Changes by file (with verified line numbers)

### Change 3 — `CBMTrainingMode` StrEnum
- **NEW `concept_benchmark/types.py`** — define `class CBMTrainingMode(StrEnum)` with `Independent`, `Sequential`.
- `concept_benchmark/config.py:270, 751` — rename `probcbm_train_class_mode: str` → `training_mode: CBMTrainingMode`.
- `experiments/baselines/probcbm.py:234, 272` — switch string equality to enum identity (`is CBMTrainingMode.Sequential`).
- No other callsites (verified by grep on `probcbm_train_class_mode` and `train_class_mode`).

### Change 6 — micro-perf (6b + 6c)
- `concept_benchmark/data.py:1791-1792` — `torch.from_numpy(np.array(c, dtype=np.float32))` → `torch.as_tensor(c, dtype=torch.float32)`; same for `y`. Audit `data.py` for the same pattern elsewhere.
- `concept_benchmark/synthetic/robot.py:144-149` — hoist `has_cm_col` and `default_cms` out of `for i, row in df.iterrows()` loop.

### Change 1 — `ConceptDataset` modernization
**1a. `input_type` field** — replace `meta["data_type"]` dispatch with typed field. Blast radius: ~17 production sites + ~13 test sites.
- `concept_benchmark/data.py:246-318` — add `input_type: InputType` required kwarg; route SampleClass on it; drop `meta.get("data_type")` lookups at `:289, :436`.
- `concept_benchmark/data.py:519, 800, 1680` — internal `ConceptDataset(...)` / `ConceptDatasetSample(...)` recreations must thread `input_type=self.input_type`.
- `concept_benchmark/generators.py:158` — pass `input_type=` (derive from config).
- `concept_benchmark/config.py:492` — emit `input_type` instead of `data_type`.
- `experiments/baselines/_common.py:437` — read `sample.input_type` not `sample.meta.get("data_type")`.
- All construction sites in tests (`conftest.py:157, 312`; `test_concept_dataset_sample.py:79,132,133,139,145,158,159,162`; `test_intervention.py:402,439,506,546`; `test_cem_integration.py:37`; `test_mini_pipelines.py:36`; `test_to_dataframe_explore.py:58`; `test_transforms.py:29,180`) — add `input_type=`.

**1b. `inputs` side-channel** — separate raw image paths from `X`. Production sites where `.X` means "path":
- `concept_benchmark/data.py:1777` — `ConceptImageDatasetSample.__getitem__` reads `self.inputs[idx]` (not `self.X[idx]`).
- `concept_benchmark/data.py:1854` — image-sample `filter()` reconstructs with `inputs=...`.
- `scripts/robot_pipeline.py:1649` — `[str(image_dir / p) for p in sample.inputs]`.
- `scripts/robot_pipeline.py:465` — likely `X=sample.X` → `inputs=sample.inputs` for image flow.
- `scripts/robot_text_pipeline.py:234` — verify is text or image.
- `experiments/baselines/probcbm.py:250` — `train_dataset.inputs[0]` if image regime.
- Tabular sites (`test_to_dataframe_explore.py`, `_common.py:462` for tabular) keep `.X` semantics — DO NOT migrate.

**1c. `classes` tuple (DOWNGRADED scope)** — store as metadata only; do NOT recode `y`.
- `concept_benchmark/data.py:246-318` — add `classes: tuple[int, ...]` kwarg, store as `self.classes`. No remapping (no callsites need it).

### Change 2 — cost-weighted interventions + rename
- `concept_benchmark/data.py:246-318` — add `concept_costs: dict[str, int]` kwarg; default `{name: 1 for name in concept_names}`. Add `__check_rep__` assertion.
- `experiments/intervention.py:129 (InterventionBatch)` — add `concept_costs: np.ndarray` field, auto-populated from `dataset.concept_costs`.
- `experiments/intervention.py:214-260` — hard rename `max_concepts_per_instance` → `per_instance_budget`. Update `per_instance_limit` helper.
- `experiments/intervention.py:679, 684` — update kflip-runner branches that read the old name.
- `experiments/intervention.py:_select_by_score` (find by grep) — replace `count + 1 ≤ budget` with `accumulated_cost + concept_costs[c] ≤ budget`. Use `continue` not `break`.
- **Rename blast radius: 47 sites across 7 files** — `experiments/kflip.py:25,47,78,81`; `experiments/intervention.py:163,185,216,247,249,679,684`; `tests/test_intervention.py` (16 sites); `tests/test_kflip.py` (14 sites); `tests/test_mini_pipelines.py:91`; `examples/{robot,sudoku}_pipeline_example.py`; `scripts/{robot_pipeline,robot_text_pipeline,sudoku_pipeline}.py`.

### Change 5 — `synthetic/` per-dataset subdirs
- `git mv synthetic/robot.py → synthetic/robot/__init__.py`
- `git mv synthetic/sudoku.py → synthetic/sudoku/__init__.py`
- `git mv synthetic/helper/robot_draw.py → synthetic/robot/draw.py`
- `git mv synthetic/helper/robot_catalog.py → synthetic/robot/catalog.py`
- `git mv synthetic/helper/textgen.py → synthetic/robot/textgen.py` (robot-only — verified)
- `git mv synthetic/helper/utils.py → synthetic/robot/colors.py` (only `robot_draw.py` imports it — verified)
- `git mv synthetic/helper/static → synthetic/robot/static` (text templates, robot-only)
- `git mv synthetic/helper/sudoku_utils.py → synthetic/sudoku/utils.py`
- `git mv synthetic/helper/sudoku_handwriting.py → synthetic/sudoku/handwriting.py`
- `git mv synthetic/robot_text → synthetic/robot/text`
- `git mv synthetic/sudoku_ocr → synthetic/sudoku/ocr`
- Delete now-empty `synthetic/helper/`.
- Fix imports: `synthetic/robot.py:12, :18`; `synthetic/sudoku.py:19, :20, :28`; `tests/test_smoke_pipelines.py:136`; `concept_benchmark/generators.py:119`; `scripts/preview_robot_render_space.py:18, :19`.
- **Caveat:** `concept_benchmark/helper/data_utils.py` is a DIFFERENT helper dir at the package root — it stays untouched.

## Implementation sequence

1. **Pre-flight**: `git checkout -b berk-refactor`; run `.venv/bin/python -m pytest tests/ -v` baseline; regenerate paper numbers fresh (`rm` cached models first, per `feedback_never_use_cached.md`); save `tests/data/*` reference fingerprints to `.ultraplan/baseline_hashes.txt`.
2. **Step 1 — Change 3** (smallest, isolated). Verify: `pytest -k probcbm`; smoke ProbCBM on seed=1014.
3. **Step 2 — Change 6** (trivial perf). Verify: `pytest tests/`; check tensor equality on a sample batch.
4. **Step 3 — Change 1** (constructor reshape; biggest blast radius). Verify: full `pytest tests/`; `pytest tests/test_robot_pipeline_regression.py` MUST pass (0.8746 / 0.8673 / 0.7812 ± tolerance).
5. **Step 4 — Change 2** (cost-weighted; depends on Change 1). Verify: `pytest tests/test_intervention.py tests/test_kflip.py` (unit costs preserve all current numbers); new tests for non-unit costs and global pool.
6. **Step 5 — Change 5** (import-path churn; last to avoid double-rewriting). Verify: `grep -rn "synthetic.helper" --include='*.py'` returns nothing; `pytest tests/`; `pytest tests/test_robot_pipeline_regression.py` (catches pickle-shim breaks).
7. **Post-flight**: diff `tests/data/*` results; manually re-run `scripts/sudoku_pipeline.py --seed 171` and confirm `0.995 / 99.5%` at ta=0.95.

## Edge cases & risks

- **Pickled-model unpickling (Step 5):** `test_robot_pipeline_regression.py:21-26` reroutes `concept_benchmark.models → experiments.models`. If Change 5 moves anything under `synthetic/`, check whether saved models reference `synthetic.helper.*` paths in their pickle. Mitigation: extend the shim or rebuild cached models before Step 5.
- **Sudoku numbers are not gated by CI** — manual verification required after Steps 3, 4, 5. Risk: silent regression on sudoku. Mitigation: write a `test_sudoku_pipeline_regression.py` skeleton during pre-flight using existing seed=171 baseline.
- **`drop_concepts` + internal `ConceptDataset` recreations** (`data.py:519, 800, 1680`) — these don't currently thread `input_type` because it comes from `meta`. After Change 1a, every internal recreation must explicitly thread the new fields. Easy to miss; grep for `ConceptDataset(` after the change.
- **Backward-compat ground rule** is HARD — no shims, no `@property concept_costs`. Tests must be updated in lockstep with each change, not deferred.
- **Test count is 368, not 83** — full `pytest tests/` runs longer than the plan suggested; budget ~5-10 min per verification gate.

## Verification

The single gate command (run after every step):
```bash
.venv/bin/python -m pytest tests/ -v && .venv/bin/python -m pytest tests/test_robot_pipeline_regression.py -v
```
After Step 5 only, additionally: manually run `.venv/bin/python scripts/sudoku_pipeline.py --seed 171` and confirm `CS sel acc ≈ 0.995, cov ≈ 99.5%` at `ta=0.95`.
