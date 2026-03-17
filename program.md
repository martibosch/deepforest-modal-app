# autoforest

This is an experiment to have the LLM autonomously iterate on training a DeepForest tree crown detection model.

## Context

We are fine-tuning a [DeepForest](https://github.com/weecology/DeepForest) pre-trained tree crown detection model on the [TreeAI Global Initiative dataset](https://zenodo.org/records/15351054). The task is bounding-box tree crown detection from aerial RGB tiles (640x640 px). The model is a RetinaNet trained via PyTorch Lightning.

Key files:
- `train.py` — training script. Args dataclass, data loading, training loop, evaluation. **You can modify this.**
- `deepforest_modal_app/eval_utils.py` — evaluation utilities (IoU matching, precision/recall/F1 computation). **Read-only.** Do not modify.
- `deepforest_modal_app/settings.py` — default config values. Read for reference. Do not modify.
- `docs/treeai_utils.py` — dataset loading/preprocessing for TreeAI. **Read-only.** Do not modify.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar17`). The branch `autoforest/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoforest/<tag>` from current main.
3. **Read the in-scope files**: Read these files for full context:
   - `train.py` — training script with Args dataclass. This is where you tune.
   - `deepforest_modal_app/eval_utils.py` — evaluation logic. Understand how metrics are computed.
   - `docs/treeai_utils.py` — data loading. Understand the annotation format.
4. **Verify data exists**: Check that the TreeAI dataset directory exists at `/mnt/new-pvc/datasets/treeai/12_RGB_ObjDet_640_fL` (the PVC mount). If running locally, override with `--base_dir docs/treeai-data/12_RGB_ObjDet_640_fL`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. Training uses wandb for logging.

**What you CAN do:**
- Modify `train.py` — hyperparameters, optimizer settings, loss formulation, augmentations, batch size, learning rate schedule, evaluation logic, callbacks, etc.

**What you CANNOT do:**
- Modify `deepforest_modal_app/eval_utils.py`. It is read-only evaluation code.
- Modify `docs/treeai_utils.py`. It is read-only data loading code.
- Modify anything in `deepforest_modal_app/__init__.py` or `deepforest_modal_app/settings.py`.
- Install new packages or add dependencies beyond what's available.

**The goal is simple: get the best validation metrics.** We track three primary metrics:
- **box_precision** — proportion of predicted boxes that match a ground truth box (at IoU >= threshold)
- **box_recall** — proportion of ground truth boxes that are detected (at IoU >= threshold)
- **box_f1** — harmonic mean of precision and recall. This is the primary metric to optimize.

Higher is better for all metrics. F1 is the single number that matters most — it balances precision and recall.

**The baseline**: From the existing notebooks, the pre-trained model achieves ~0.63 precision, ~0.14 recall, ~0.22 F1. Fine-tuning with default settings and augmentations gets ~0.63 precision, ~0.18 recall, ~0.28 F1. We want to beat this.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome.

**The first run**: Your very first run should always be to establish the baseline, so you will run `train.py` as-is.

## Output format

The training script prints a summary at the end with pre-training and post-training metrics. You can also extract metrics from wandb logs or from the console output.

Key output to look for:
```
Post-training metrics
Precision: 0.XXXX
Recall:    0.XXXX
F1:        0.XXXX
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 6 columns:

```
commit	box_f1	box_precision	box_recall	status	description
```

1. git commit hash (short, 7 chars)
2. box_f1 (e.g. 0.2840) — use 0.0 for crashes
3. box_precision (e.g. 0.6321) — use 0.0 for crashes
4. box_recall (e.g. 0.1831) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	box_f1	box_precision	box_recall	status	description
a1b2c3d	0.2245	0.6284	0.1367	keep	baseline (pre-trained model defaults)
b2c3d4e	0.2840	0.6321	0.1831	keep	augmentations + 50 epochs
c3d4e5f	0.1900	0.5500	0.1150	discard	aggressive lr=1e-3
d4e5f6g	0.0	0.0	0.0	crash	batch_size=32 OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoforest/mar17`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Formulate a hypothesis and modify `train.py`
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "Post-training metrics" -A 3 run.log` and `grep "Summary" -A 6 run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If metrics improved (higher box_f1), you "advance" the branch, keeping the git commit
9. If metrics are equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Ideas to explore** (non-exhaustive):
- Learning rate: the current default is 1e-4 with ReduceLROnPlateau. Try cosine annealing, warmup, OneCycleLR, different base LRs
- Augmentations: the current set is from the notebook. Try different combinations, probabilities, or more aggressive augmentations (MixUp, Mosaic if supported)
- Batch size / accumulation: currently batch_size=4 with accumulate_grad_batches=4 (effective 16). Try larger/smaller
- Training duration: more epochs, different patience for early stopping
- IoU threshold for NMS during prediction: the default prediction uses 0.15, try tuning this
- Patch size / overlap for predict_tile: currently predicting single images, try tiled prediction with overlap
- Score threshold: filter low-confidence predictions before evaluation
- Pre-training model: try different model revisions or the bird model as a starting point
- Optimizer: AdamW vs SGD, weight decay, gradient clipping
- Precision: bf16-mixed vs 32-bit
- Unfreezing strategy: freeze backbone for N epochs then unfreeze

**Crashes**: If a run crashes (OOM, bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix and re-run. If the idea itself is fundamentally broken, log "crash", and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read the eval_utils for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.
