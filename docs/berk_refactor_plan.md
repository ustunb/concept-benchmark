# Berk-Branch Refactor Adoption Plan

Tracks which changes from `blm/slm/external/cbm/` (berk branch) we adopt upstream into `concept-benchmark`. Reference vendored copy:
`/Users/jskirzynski/Desktop/blm/slm/external/cbm/`.

**Ground rules:**
- No backward-compatibility shims. Hard renames, hard removals.
- Same external behavior — pipelines must produce identical results.
- One change → one PR-sized commit.

---

## Change 1 — `ConceptDataset` modernization (adopt 3 sub-ideas, drop BCD inheritance)

**Status:** Planned
**Files touched:** `concept_benchmark/data.py`, downstream callers of `meta["data_type"]` / `dataset.X` / raw `y` encoding, tests under `tests/`.

We do **not** inherit from his `BinaryClassificationDataset` (would require vendoring his `slm` package). We lift three orthogonal improvements.

### 1a. Hoist `input_type` from `meta["data_type"]` to a typed field
- Add `InputType = Literal["image", "tabular", "text"]` alias.
- New constructor kwarg `input_type: InputType` (required, no default).
- Remove all `meta.get("data_type")` lookups; route SampleClass selection on `self.input_type`.
- Update generators (`concept_benchmark/synthetic/{robot,sudoku}.py`, `generators.py`) and any factory functions to pass `input_type=` explicitly.
- `meta` becomes free-form payload only.
- Reference: `blm/slm/external/cbm/dataset.py:615, 678, 833`.

### 1b. Split raw inputs (`inputs`) from model-facing features
- Rename concept matrix slot to `C` (already named C — fine), keep separate `inputs: np.ndarray | None` field for raw image paths / sudoku grids.
- Constructor takes `C`, `y`, `inputs=None`. No more overloading `X` to mean "feature matrix OR list of paths."
- `ConceptImageDatasetSample.__getitem__` reads `self.inputs[i]` (path) and emits decoded tensor; tabular sample reads concept/feature matrix directly.
- Audit all sites that currently do `dataset.X[i]` for an image regime and switch to `dataset.inputs[i]`.
- Reference: `blm/slm/external/cbm/dataset.py:614, 668-670, 677`.

### 1c. Freeze label space with a `classes` tuple
- Add `classes: tuple[int, ...]` constructor kwarg (required).
- Recode `y` at `__init__` so `y` always matches `classes` thereafter. Store `self.classes`.
- Remove ad-hoc `{0,1}` ↔ `{-1,+1}` remapping scattered downstream (search for `2*y - 1`, `np.where(y==0, -1, 1)`, `expit` callsites).
- Reference: `blm/slm/external/cbm/dataset.py:616, 694`.

### Verification
- All existing tests pass with `./venv/bin/python -m pytest tests/ -v`.
- Robot demo (seed=1014) reproduces paper numbers in `MEMORY.md` (DNN 0.8746, Ideal CBM 0.8673, Subconcept CBM 0.7812).
- Sudoku seed 171 reproduces CS sel acc 0.995 / cov 99.5% at ta=0.95.

---

## Change 2 — Cost-weighted interventions (minimal upgrade of existing budget engine)

**Status:** Planned
**Files touched:** `concept_benchmark/data.py`, `experiments/intervention.py`, tests.

Background: `InterventionConfig` already has `concept_budget` / `instance_budget` / `max_concepts_per_instance` and a `_resolve_budget` helper (`experiments/intervention.py:214-216`). What's missing is a **per-concept cost vector** — today the budget engine counts units (1 per concept). This change upgrades the budget arithmetic to count costs.

With default unit costs, behavior is byte-identical to today. Paper results preserved.

**Reference:** `blm/slm/external/cbm/dataset.py:56`, `intervention.py:54,738-871,816-826`.

### 2a. Per-concept cost vector on the dataset
- Add `concept_costs: dict[str, int]` field on `ConceptDataset` (`concept_benchmark/data.py`).
- Default at construction: `{name: 1 for name in concept_names}` — uniform unit cost reproduces current behavior.
- `__check_rep__` asserts: every concept name has a non-negative int cost; keys equal `concept_names`.
- No `Costs` wrapper dataclass — plain dict. (Berk's wrapper is namespace-prep for axes we don't have. Skip.)

### 2b. `InterventionBatch.concept_costs` field
- Add `concept_costs: np.ndarray` to `InterventionBatch`.
- Auto-populated from `dataset.concept_costs`, ordered by `concept_names`.
- Validator: shape `(n_concepts,)`, non-negative.

### 2c. Upgrade selection loop from count-based to cost-weighted
- Replace per-concept `count + 1 ≤ budget` checks with `accumulated_cost + concept_costs[c] ≤ budget`.
- Applies to both the global pool path (`concept_budget`) and — see 2d — the per-instance path.
- Use `continue` (not `break`) when a candidate doesn't fit, so cheaper later candidates can still be picked.

### 2d. Rename `max_concepts_per_instance` → `per_instance_budget` (cost-aware)
- Hard rename — no back-compat alias.
- Semantics: per-instance cost cap. With unit costs and integer value `k`, equivalent to "at most k concepts per instance" (paper behavior).
- All callers updated; docstrings updated to reflect cost semantics.

### Verification
- Robot demo (seed=1014) — k=1,3,7 ideal CBM accuracies match `MEMORY.md` baselines (0.9734, 0.9767, 0.9767).
- Subconcept CBM k=1,3,12: 0.9212, 0.9439, 0.9439.
- New test: non-unit costs (e.g. `{c0: 1, c1: 2}`) with `per_instance_budget=3` selects different concept set than unit-cost case.
- New test: global pool (`concept_budget=B` only) selects highest-scoring `(instance, concept)` pairs across batch until budget exhausted; some instances may receive zero corrections.

---

## Change 3 — `CBMTrainingMode` StrEnum (Independent | Sequential)

**Status:** Planned
**Files touched:** new `concept_benchmark/types.py`, `concept_benchmark/config.py`, `experiments/baselines/probcbm.py`, any CLI that exposes training mode.
**Reference:** `blm/slm/external/cbm/types.py`

### 3a. New `concept_benchmark/types.py`
```python
from enum import StrEnum

class CBMTrainingMode(StrEnum):
    Independent = "independent"   # front-end fits on GT C; eval on predicted C
    Sequential  = "sequential"    # front-end fits on predicted C; train+eval both flow through backend
```

### 3b. Hard rename `probcbm_train_class_mode` → `training_mode`
- `config.py:270` and `:751`: replace `probcbm_train_class_mode: str = "independent"` with `training_mode: CBMTrainingMode = CBMTrainingMode.Independent`.
- Drop the `probcbm_` prefix — concept is CBM-wide, not ProbCBM-specific.
- No back-compat alias.

### 3c. Update ProbCBM call sites
- `experiments/baselines/probcbm.py:234`: `"train_class_mode": config.training_mode`
- `experiments/baselines/probcbm.py:272`: `if model.train_class_mode is CBMTrainingMode.Sequential:` (identity check, not string equality).

### 3d. CLI validation (if/when a `--training_mode` flag is added)
- Argparse `choices=[m.value for m in CBMTrainingMode]`.
- Reject `Sequential` + `--skip_backend` at parse time (Berk pattern, `train_cbm.py:500`). Only wire this if/when the flag exists.

### Verification
- Existing ProbCBM tests pass unchanged.
- Grep `probcbm_train_class_mode` and `train_class_mode` to confirm all call sites switched.
- Typing a bogus string (e.g. `"independnt"`) into a config raises at construction, not at runtime.

---

## Change 4 — Propagation improvements

**Status:** Skip
**Reference:** `blm/slm/external/cbm/predictor.py:33, 262, 286`

- Threshold bump (4096 → 32768): marginal, not worth touching.
- `|S|`-scoped marginalization: only pays off with sparse front-ends; our `label_predictor` is dense LR. Revisit if we add a sparse front-end.
- Closed-form noisy-OR / noisy-AND: only applies to Boolean disjunction/conjunction front-ends (Berk's SLM thesis). We don't have those.

---

## Change 5 — Per-dataset subdir layout inside `synthetic/`

**Status:** Planned
**Files touched:** `concept_benchmark/synthetic/` (whole subtree), all import sites.
**Reference:** Berk's `blm/data/{robots,sudoku,...}/` per-dataset convention.

**Motivation:** Today `concept_benchmark/synthetic/helper/` shares one directory between robot-specific and sudoku-specific code, separated only by filename prefix. Splitting into per-dataset subdirs makes ownership obvious, eases adding a third dataset later, and aligns conceptually with how Berk's repo treats datasets.

### 5a. Restructure to per-dataset subdirs
Target layout:
```
concept_benchmark/synthetic/
├── robot/
│   ├── __init__.py        # was robot.py
│   ├── draw.py            # was helper/robot_draw.py
│   ├── catalog.py         # was helper/robot_catalog.py
│   └── text/              # was robot_text/
└── sudoku/
    ├── __init__.py        # was sudoku.py
    ├── utils.py           # was helper/sudoku_utils.py
    ├── handwriting.py     # was helper/sudoku_handwriting.py
    └── ocr/               # was sudoku_ocr/
```
- `helper/` is dissolved.
- File renames drop the dataset prefix (no more `robot_draw.py` → just `draw.py` under `robot/`).

### 5b. Update imports
- Update every `from concept_benchmark.synthetic.helper.robot_draw import ...` → `from concept_benchmark.synthetic.robot.draw import ...`.
- Same for catalog, sudoku_utils, sudoku_handwriting.
- Update `robot.py` / `sudoku.py` module-level imports (which became `robot/__init__.py` / `sudoku/__init__.py`).
- Update `robot_text/` and `sudoku_ocr/` imports if they cross into helpers.
- Update `concept_benchmark/generators.py` and any public re-exports.

### 5c. Skip module-prefix rename in `__init__.py`
Public API stays the same. `from concept_benchmark.robots import DatasetGenerator` keeps working (it routes through whatever shim exists today). The change is internal layout only.

### Verification
- All tests pass with `./venv/bin/python -m pytest tests/ -v`.
- Robot demo (seed=1014) reproduces paper numbers.
- Sudoku demo (seed=171) reproduces CS sel acc 0.995.
- `grep -rn "synthetic.helper"` returns no hits.

---

## Change 6 — Micro-perf cleanups (6b + 6c only; skip 6a)

**Status:** Planned
**Files touched:** `concept_benchmark/data.py`, `concept_benchmark/synthetic/robot.py`.
**Reference:** blm commit `42fa419` ("simplify pass on foundational refactor").

### 6a. SKIP — `io.BytesIO` → `with Image.open(path)`
Skip. Current `io.BytesIO(Path(img_path).read_bytes())` pattern was likely written to dodge macOS DataLoader fd-exhaustion (see `MEMORY.md` MPS/DataLoader notes). Not worth the regression risk for a marginal perf win.

### 6b. Replace `torch.from_numpy(np.array(...))` with `torch.as_tensor(...)`
- `data.py:1791-1792`:
  ```python
  # before
  c = torch.from_numpy(np.array(c, dtype=np.float32))
  y = torch.from_numpy(np.array(y, dtype=np.int64))
  # after
  c = torch.as_tensor(c, dtype=torch.float32)
  y = torch.as_tensor(y, dtype=torch.int64)
  ```
- Avoids the redundant copy when dtype already matches.
- Audit `data.py` and other sample classes for the same `from_numpy(np.array(...))` pattern; convert all hits.

### 6c. Hoist loop-invariant color-mode branch in `synthetic/robot.py:144-149`
- Before:
  ```python
  for i, row in df.iterrows():
      cms = (
          row.get(color_mode_col, None)
          if (include_color and color_mode_col in df.columns)
          else (colorish(df) if include_color else "grayscale")
      )
  ```
- After:
  ```python
  has_cm_col = include_color and color_mode_col in df.columns
  default_cms = colorish(df) if include_color else "grayscale"
  for i, row in df.iterrows():
      cms = row.get(color_mode_col, None) if has_cm_col else default_cms
  ```
- Per-row work drops from "ternary + conditional `colorish(df)` call" to "dict lookup."

### Verification
- All tests pass with `./venv/bin/python -m pytest tests/ -v`.
- Robot demo (seed=1014) reproduces paper numbers in `MEMORY.md`.
- Numerical equivalence sanity check: `c` and `y` tensors are byte-identical to current output (`torch.equal(c_old, c_new)` on a sample batch).

---

## Decided: Skip

- `Predictor` ABC — don't collapse separate `FrontEndModel` / `ConceptBasedModel` / `ConceptDetector` models.
- Composition recipe registry — wrong abstraction for our axes of variation (regimes, not predictor × calibration × abstention triples).
- `VarianceInterventionStrategy` (Berk added it, we don't have it) — `KFlipInterventionStrategy` (`experiments/kflip.py`) is already the responsiveness-of-label-to-concept-changes scoring the paper defends. Berk's variance is a cheaper per-concept scalarization of the same idea; not strictly better and adding it post-submission risks muddying paper framing.
- Extract training out of `ConceptDetector` into standalone `training.py` (Berk's `fit_concept_model` / `train_concept_heads`) — keep sklearn-ish `.fit()` convention. Revisit only if we need to vary training procedure (distributed, mixed-precision, alternative losses).
- `formula.py` adoption — Berk's copy is byte-identical to ours except import path. No-op.
- `inception_lr_backend.py` — Berk-specific (medical-image CBM backbone).
- `slm_front_end.py` — Berk-specific (his SLM Boolean as a front-end).

---

# Implementation Plan

## Ordering rationale

Sequenced by **blast radius (small → large)** and **inter-change dependencies**:
- Change 3 is fully isolated — go first to bank a quick win.
- Change 6 is a local micro-perf with no API impact.
- Change 1 reshapes the dataset constructor; everything dataset-facing changes.
- Change 2 adds `concept_costs` as a new dataset field, so it must come after Change 1.
- Change 5 is pure import-path churn; do it last so prior PRs don't fight import shuffles.

## Pre-flight (do once, before any change)

1. Branch from `main`: `git checkout -b berk-refactor`.
2. **Regenerate baseline paper numbers from scratch** (per `feedback_never_use_cached.md`):
   - Robot demo seed=1014: confirm DNN 0.8746, Ideal CBM 0.8673, Subconcept CBM 0.7812; k-budget table matches `MEMORY.md`.
   - Sudoku seed=171: confirm CS sel acc 0.995, cov 99.5% @ ta=0.95.
   - `rm` any cached models/data first; never trust the cache during a refactor.
3. Save baseline outputs to `results/baseline_pre_refactor/` as the source of truth for every subsequent verification.
4. `./venv/bin/python -m pytest tests/ -v` → expect 83 pass + 4 skip (Feb 2026 baseline).

## Step 1 — Change 3 (`CBMTrainingMode` StrEnum)

**Why first:** zero coupling to other changes; smallest diff; immediate type-safety win.

Sub-steps:
1. Create `concept_benchmark/types.py` with the StrEnum.
2. Replace `probcbm_train_class_mode: str` in `config.py:270, :751` with `training_mode: CBMTrainingMode`.
3. Update `experiments/baselines/probcbm.py:234, :272` to use the enum identity check.
4. `grep -rn "probcbm_train_class_mode\|train_class_mode"` → fix all hits.

Verify: tests pass; ProbCBM smoke run on robot seed=1014 reproduces baseline; bogus string `"independnt"` raises at config construction.

Commit. Tag: `refactor: introduce CBMTrainingMode enum`.

## Step 2 — Change 6 (micro-perf 6b + 6c)

**Why second:** trivial; no API change; flushes the trivial wins before the bigger structural moves.

Sub-steps:
1. `data.py:1791-1792`: replace `torch.from_numpy(np.array(c, dtype=np.float32))` with `torch.as_tensor(c, dtype=torch.float32)`. Same for `y`.
2. Audit all other `from_numpy(np.array(...))` patterns in `data.py` and switch.
3. `synthetic/robot.py:144-149`: hoist `has_cm_col` and `default_cms` out of the row loop.

Verify: tests pass; robot demo seed=1014 numbers byte-identical to baseline. (No semantic change expected.)

Commit. Tag: `perf: avoid redundant tensor copies + hoist loop-invariant lookup`.

## Step 3 — Change 1 (`ConceptDataset` modernization)

**Why third:** biggest semantic refactor of the dataset constructor. Hard-rename, no back-compat (per ground rules).

Sub-steps (do as ONE PR but split mentally):
1. **1a — `input_type` field**
   - Add `InputType = Literal["image", "tabular", "text"]`.
   - Add required kwarg `input_type: InputType` to `ConceptDataset.__init__`.
   - Replace every `meta.get("data_type")` lookup with `self.input_type`.
   - Update every generator/factory to pass `input_type=` explicitly.
2. **1b — `inputs` channel split**
   - Add `inputs: np.ndarray | None` constructor kwarg and field.
   - `C` is now strictly the concept matrix; `inputs` holds raw image paths.
   - `ConceptImageDatasetSample.__getitem__` reads from `self.inputs[idx]`.
   - Audit every `dataset.X[i]` / `sample.X[i]` that meant "raw path" — switch to `inputs`.
3. **1c — `classes` tuple**
   - Add required kwarg `classes: tuple[int, ...]`.
   - Recode `y` at `__init__` to match `classes` encoding.
   - Store `self.classes`.
   - Remove all downstream ad-hoc `2*y - 1` / `np.where(y==0, -1, 1)`. Grep for them first.

After all three sub-steps:
- Verify tests pass.
- Robot demo seed=1014 reproduces baseline.
- Sudoku seed=171 reproduces baseline.

Commit (single commit OK; the three sub-changes interlock).

Tag: `refactor: modernize ConceptDataset constructor (input_type / inputs / classes)`.

## Step 4 — Change 2 (cost-weighted interventions)

**Why fourth:** requires Change 1's `ConceptDataset` constructor changes in place. Touches a different file (`intervention.py`) so it's a clean follow-up PR.

Sub-steps:
1. Add `concept_costs: dict[str, int]` field on `ConceptDataset`, default `{name: 1 for name in concept_names}`.
2. Add `__check_rep__` assertion for cost dict shape.
3. Add `concept_costs: np.ndarray` field on `InterventionBatch` auto-populated from dataset.
4. Hard rename `max_concepts_per_instance` → `per_instance_budget` everywhere (`InterventionConfig`, all strategies, tests, docstrings).
5. Upgrade selection loop: replace `count + 1 ≤ budget` with `accumulated_cost + concept_costs[c] ≤ budget`.
6. Use `continue` (not `break`) when a candidate doesn't fit.

Verify:
- All k-budget intervention tests pass unchanged (unit costs preserve behavior).
- Robot demo k=1,3,7 ideal: 0.9734, 0.9767, 0.9767.
- Subconcept k=1,3,12: 0.9212, 0.9439, 0.9439.
- New test: non-unit costs with `per_instance_budget=3` yields a different selected concept set than unit costs.
- New test: global pool (`concept_budget=B`) selects top-scoring `(instance, concept)` pairs across batch; some instances may receive zero corrections.

Commit. Tag: `feat: cost-weighted interventions (per-instance + global pool)`.

## Step 5 — Change 5 (per-dataset subdir in `synthetic/`)

**Why last:** big import-path churn. Done last so prior PRs don't fight repeated `from ... import ...` rewrites.

Sub-steps:
1. Use `git mv` for every file (preserves blame):
   - `synthetic/robot.py` → `synthetic/robot/__init__.py`
   - `synthetic/sudoku.py` → `synthetic/sudoku/__init__.py`
   - `synthetic/helper/robot_draw.py` → `synthetic/robot/draw.py`
   - `synthetic/helper/robot_catalog.py` → `synthetic/robot/catalog.py`
   - `synthetic/helper/sudoku_utils.py` → `synthetic/sudoku/utils.py`
   - `synthetic/helper/sudoku_handwriting.py` → `synthetic/sudoku/handwriting.py`
   - `synthetic/robot_text/` → `synthetic/robot/text/`
   - `synthetic/sudoku_ocr/` → `synthetic/sudoku/ocr/`
2. Delete now-empty `synthetic/helper/`.
3. `grep -rn "synthetic.helper\|synthetic.robot_text\|synthetic.sudoku_ocr"` → fix every import.
4. Update `concept_benchmark/__init__.py` and any `robots`/`sudoku` shim packages.

Verify: tests pass; robot + sudoku demos reproduce baselines; `grep -rn "synthetic.helper"` returns nothing.

Commit. Tag: `refactor: per-dataset subdir layout under synthetic/`.

## Post-flight

1. Diff `results/baseline_pre_refactor/` against current results — must match for all locked numbers.
2. `git log --oneline main..berk-refactor` should show 5 commits.
3. Open a PR per commit, or one stacked PR with the 5 commits, depending on review preference.
4. Update `MEMORY.md` to remove obsolete notes (e.g., references to `meta["data_type"]`, `max_concepts_per_instance`).

## Rollback strategy

Each step is a standalone commit; revert any one without affecting the others IF the dependency order is respected (i.e., revert later commits first). Specifically:
- Change 2 depends on Change 1's constructor → revert Change 2 first if both need to go.
- All other steps are independent of each other; revert in any order.

## Deferred / Skip

- `PlattCalibrated` wrapper — defer until calibration is on roadmap.
- `data/<name>/<name>_processing.py` convention — blm-specific layout.
- BCD inheritance — requires vendoring `slm.data` stack.
