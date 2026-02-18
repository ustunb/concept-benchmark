"""Demo: Sudoku benchmark — reproduce mc21 experiment (Table 3).

Determines whether a 9x9 sudoku board is valid using 27 ground-truth concepts.
This experiment uses the harder mc21 variant (up to 21 corrupted cells).

Dataset: 1000 boards (50% valid, 50% invalid), rendered with handwritten digits.
Concepts (27 binary):
  row_valid_1..9   -- is row i free of duplicates?
  col_valid_1..9   -- is column i free of duplicates?
  block_valid_1..9 -- is 3x3 block i free of duplicates?
Label: valid (1) iff ALL 27 concepts are true.

Pipeline: board images -> OCR (TinyResNet) -> inferred digits ->
          ConceptSudokuCNN -> concept probs -> FrontEndModel -> valid/invalid
"""
from concept_benchmark.benchmarks import sudoku
from concept_benchmark.config import SudokuBenchmarkConfig

cfg = SudokuBenchmarkConfig(
    seed=171,
    # --- Data generation ---
    n=3,                                    # 3x3 blocks -> standard 9x9 grid
    n_samples=1000,                         # 1000 boards total
    max_corrupt=21,                         # up to 21 cells corrupted (harder variant)
    valid_ratio=0.5,                        # 50% valid / 50% invalid
    # --- Image rendering (for OCR pipeline) ---
    cell_px=50,                             # 50px per cell
    handwriting=True,                       # render with handwritten digit style
    # --- DNN training (baseline: digits -> valid/invalid, no concepts) ---
    epochs=20,
    patience=5,
    batch_size=32,
    # --- Concept Supervision model training ---
    cs_epochs=100,                          # GroupPoolingConceptSudokuCNN + FrontEndModel
    cs_patience=20,                         # early stopping patience
    # --- Selective classification ---
    target_accuracy=0.9,                    # target: 90% accuracy on accepted samples
    decision_threshold=0.5,                 # label decision boundary
    intervention_thresholds=[0.2, 0.4, 0.6, 0.8],  # abstention confidence thresholds
)

# Stage 1: Generate sudoku boards + rendered images
# Creates:
#   Tabular: X (1000, 81) -- flattened 9x9 cell values (1-9)
#            C (1000, 27) -- binary validity concepts
#            y (1000,)    -- 0=invalid, 1=valid
#   Images:  450x450 PNG boards with handwritten digits
sudoku.setup_dataset(cfg)

# Stage 2: Train OCR digit recognizer (TinyResNet on 50x50 cell crops)
# Learns to read handwritten digits from rendered board images.
sudoku.train_ocr(cfg)

# Stage 3: Train concept supervision model
# Architecture: OCR digits -> GroupPoolingConceptSudokuCNN ->
#   27 concept probabilities -> FrontEndModel -> valid/invalid
# The concept layer is supervised with ground-truth row/col/block validity.
cs_model = sudoku.train_cs(cfg)

# Stage 4: Train DNN baseline (OCR digits -> valid/invalid, no concept layer)
dnn_weights = sudoku.train_dnn(cfg)

# Stage 5: Conceptual safeguards interventions
# Unlike robot (which uses k-flip), sudoku uses conceptual safeguards:
# 1. Abstain on samples where max(P(valid), P(invalid)) < threshold
# 2. For abstained samples, intervene on uncertain concepts
# 3. Re-predict with corrected concepts
interv_df = sudoku.run_interventions(cfg)

# Stage 6: Selective accuracy
# For target_accuracy=0.9: what fraction of test samples can we accept
# while maintaining >=90% accuracy? Higher coverage = better model.
# Returns DataFrame with selective_acc, selective_cov per model.
sel_df = sudoku.compute_selective_results(cfg)

print("Selective results (mc21, target=90%):")
print(sel_df.to_string(index=False))

# Stage 7: Alignment test
# Replace learned frontend with human-aligned weights:
#   w = [+1, +1, ..., +1] for all 27 concepts (each violation -> invalid)
# Check: does the aligned model maintain accuracy?
align_stats = sudoku.align(cfg)
print(f"\nAlignment: {align_stats}")

# --- Or run everything + collect into CSV in one call ---
# sudoku.run(cfg)
