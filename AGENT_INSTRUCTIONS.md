# Instructions for Claude Running in Google Colab

You are running the scaling experiment for an IDL course project at CMU.
This project replicates the "Who is Harry Potter?" paper's approach to machine unlearning,
targeting Stephen King knowledge in Qwen 2.5 3B, evaluated on the RWKU benchmark.

## Context

The team has already run the base pipeline (128 passages) and found negligible unlearning.
The goal now is to produce a **scaling curve** showing that the approach doesn't work
even at much larger corpus sizes — demonstrating that the method fundamentally requires
access to the original training data (which the original paper had via GPT-4 regeneration
of millions of instances).

## What To Do

### Step 0: Setup
```bash
# Already done if using the Colab notebook, but just in case:
pip install -q -r requirements.txt
```

Set the OUTPUT_DIR environment variable to your Google Drive path:
```python
import os
os.environ["OUTPUT_DIR"] = "/content/drive/MyDrive/who-is-harry-potter-output"
```

### Step 1: Run the base pipeline first (if not already done)
```bash
python 1_build_forget_corpus.py
python 3_train_reinforced.py
python 4_generate_labels.py
python 5_train_unlearn.py
python 6_evaluate_rwku.py
```

### Step 2: Run the scaling experiment
```bash
python 7_scaling_experiment.py
```

This script will:
1. Generate a pool of diverse passages about Stephen King using the base model (~5000 unique)
2. For each target size [128, 1000, 5000, 10000, 50000, 90000, 128000]:
   - Build corpus at that size (unique passages + oversampling for larger sizes)
   - Create anchor-based sanitized counterparts
   - Train reinforced model (10 epochs)
   - Generate alternative labels
   - Train unlearned model (5 epochs)
   - Evaluate on RWKU benchmark
3. Save results to `data/scaling_results.json`
4. Generate plots: `data/scaling_curve.png` and `data/scaling_curve_mia_utility.png`

**The script supports resuming** — if it crashes, just re-run it and it will skip completed sizes.

### Step 3: Also try RWKU negatives
The script automatically tries to download RWKU negative passages.
Check `data/rwku_negatives.json` after running to see what's available.

## Expected Runtime on A100
- Passage pool generation: ~1-2 hours (5000 unique passages)
- Per experiment size:
  - Size 128-1000: ~30-45 min
  - Size 5000-10000: ~45-60 min
  - Size 50000+: ~60-90 min (mostly training time)
- Total: ~6-10 hours

## Key Files
- `config.py` — All hyperparameters and paths
- `7_scaling_experiment.py` — The scaling experiment script
- `2_anchors.py` — Entity sanitization for anchor replacement (imported via `importlib` since name starts with digit)
- `6_evaluate_rwku.py` — Evaluation (imported by scaling script)
- `data/scaling_results.json` — Output results (saved incrementally)
- `data/scaling_curve.png` — Output plot

## Important Notes
- The script saves after EACH experiment size, so partial results are preserved
- Large corpus sizes (50K+) use oversampling from the unique pool — this is documented and expected
- The baseline (original model) evaluation is repeated for each size as a consistency check
- The import of `2_anchors.py` uses `_2_anchors` because Python module names can't start with digits

## Expected Result
We expect the scaling curve to show **flat or negligible improvement** across all corpus sizes,
demonstrating that without access to the actual training data, the "Who is Harry Potter"
approach cannot effectively unlearn deeply embedded knowledge like Stephen King.

## Existing Results (for reference)
Previous runs with 128 passages showed:
- Forget ROUGE-L barely changed (original ~0.030-0.045 vs unlearned ~0.031-0.046)
- MIA scores virtually identical
- The method has essentially no effect at this scale

These results are saved in `data/eval_results_run1.json` and `data/eval_results_run2.json`.
