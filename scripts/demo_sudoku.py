"""Demo: Sudoku benchmark — selective classification and concept interventions.

Determines whether a 9x9 Sudoku board is valid using 27 ground-truth concepts.
The system handles routine cases automatically and defers uncertain ones to a human.

Dataset: 1000 boards (50% valid, 50% invalid), rendered with handwritten digits.
Concepts (27 binary):
  row_valid_1..9   -- is row i free of duplicates?
  col_valid_1..9   -- is column j free of duplicates?
  block_valid_1..9 -- is 3x3 block k free of duplicates?

Label: valid (1) iff ALL 27 concepts are true (AND structure).

Pipeline: board images -> OCR (digit recognizer) -> inferred digits ->
          concept model -> 27 concept probs -> FrontEndModel -> valid/invalid

Run from the command line:
  cbm-benchmark sudoku --seed 171
"""
from concept_benchmark.benchmarks import sudoku
from concept_benchmark.config import SudokuBenchmarkConfig

cfg = SudokuBenchmarkConfig(
    seed=171,
    # --- Data generation ---
    n_samples=1000,                             # 1000 boards total
    max_corrupt=9,                              # up to 9 cells corrupted in invalid boards
    valid_ratio=0.5,                            # 50% valid / 50% invalid
    handwriting=True,                           # render with handwritten digit style
    # --- Selective classification ---
    target_accuracy=0.95,                       # minimum 95% accuracy on accepted samples
    # --- Interventions ---
    intervention_budgets=[1, 3, -1],            # k=1, k=3, k=max (resolves to 27)
    intervention_thresholds=[0.2, 0.4],         # concept confidence thresholds
)

# Stage 1: Generate sudoku boards + rendered images
# Creates 1000 boards with handwritten digits (450x450 PNGs).
# X: (1000, 81) flattened cell values | C: (1000, 27) validity concepts | y: (1000,)
sudoku.setup_dataset(cfg)

# Stage 2: Train OCR digit recognizer on 50x50 cell crops
sudoku.train_ocr(cfg)

# Stage 3: Train concept-supervised model
# GroupPoolingConceptSudokuCNN -> 27 concept probs -> FrontEndModel -> valid/invalid
cs_model = sudoku.train_cs(cfg)

# Stage 4: Train DNN baseline (digits -> valid/invalid, no concept layer)
dnn_weights = sudoku.train_dnn(cfg)

# Stage 5: Run concept interventions
# For abstained samples, verify uncertain concepts and re-predict.
interv_df = sudoku.run_interventions(cfg, cs_model)

# Stage 6: Selective classification
# Find confidence threshold where kept predictions achieve >= target_accuracy.
# Higher coverage = better model (handles more cases without human help).
sel_df = sudoku.compute_selective_results(cfg)

print("Selective results (mc9, target=95%):")
print(sel_df[sel_df["target_accuracy"] == 0.95][
    ["model", "selective_acc", "selective_cov"]
].to_string(index=False))

# Stage 7: Alignment
# Replace learned frontend with uniform positive weights for all 27 concepts.
# Each concept violation should push toward "invalid".
align_stats = sudoku.align(cfg)
print(f"\nAlignment: {align_stats}")

# ── Harder variant ───────────────────────────────────────────────────
# Uncomment to test with more corrupted cells (subtler errors, harder task).
#
# cfg_hard = SudokuBenchmarkConfig(
#     seed=171,
#     max_corrupt=21,                           # up to 21 cells corrupted
#     target_accuracy=0.90,                     # lower target for harder variant
#     intervention_budgets=[1, 3, -1],
# )
# sudoku.run(cfg_hard)

# ── Or run everything + collect into CSV in one call ─────────────────
# sudoku.run(cfg)
