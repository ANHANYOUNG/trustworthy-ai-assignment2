# Trustworthy AI Assignment 2

This project implements `DeepXplore-style differential testing` for two `CIFAR-10 ResNet50` models.

Important note: I did **not** directly run the original `deepxplore` package for the final submission. Instead, I used the original DeepXplore idea and codebase as a reference, then rebuilt the actual testing pipeline in `PyTorch` for `CIFAR-10` and `ResNet50`.

The final submission entry point is `test.py`, and generated figures / reports are written to `results/`.

## Setup

- Tested with `Python 3.10`

Create and activate a conda environment first:

```bash
conda create -n trustworthy-ai-a2 python=3.10 -y
conda activate trustworthy-ai-a2
```

If your machine needs a specific CUDA build of PyTorch, install `torch` and `torchvision` first using the official PyTorch command for your system, then run:

```bash
pip install -r requirements.txt
```

Optional notebook kernel setup for `debug/debug.ipynb`:

```bash
python -m ipykernel install --user --name trustworthy-ai-a2 --display-name "Python (trustworthy-ai-a2)"
```

## Run

Basic run:

```bash
python test.py
```

This default run executes all three transformations:

- `light`
- `occl`
- `black`

Run only one transformation:

```bash
python test.py --transformation black
python test.py --transformation occl
python test.py --transformation light
```

If you want to choose a specific GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```

Useful arguments:

```bash
python test.py \
  --transformation all \
  --threshold 0.2 \
  --seeds 200 \
  --grad-iterations 20 \
  --generated-input-limit 5
```

## What `test.py` Does

- loads the `CIFAR-10` test set
- loads two trained `ResNet50` checkpoints:
  - `ckpts/resnet50_wo_aug.pt`
  - `ckpts/resnet50_w_aug.pt`
- evaluates both models on the clean test set
- measures clean prediction disagreement between the two models
- computes baseline neuron coverage for both models
- generates transformed inputs that try to:
  - increase model disagreement
  - increase neuron coverage
- saves suspicious input visualizations, coverage summaries, and markdown reports under `results/`

## DeepXplore Modifications

The original DeepXplore repository is included under `deepxplore/`, but it is kept only as a reference. The final submission does not execute that package directly. Instead, `test.py` and `utils.py` provide a newly written `PyTorch` implementation that follows the same differential-testing idea for this assignment setting.

Main modifications:

- replaced the legacy `Keras/TensorFlow` execution flow with a `PyTorch` implementation
- adapted the method from the original DeepXplore examples to `CIFAR-10` classification
- used `ResNet50` models instead of the original example networks
- modified `ResNet50` for `32x32` CIFAR-10 inputs:
  - `conv1` changed to `3x3, stride 1, padding 1`
  - initial `maxpool` removed with `nn.Identity()`
  - final `fc` changed to `10` output classes
- kept preprocessing consistent with the trained models using:
  - mean `=(0.4914, 0.4822, 0.4465)`
  - std `=(0.2470, 0.2435, 0.2616)`
- replaced the original coverage tracking with `PyTorch forward hooks`
- implemented three constraint types for gradient-based input generation:
  - `light`
  - `occl`
  - `black`
- added automatic saving of:
  - clean disagreement examples
  - generated suspicious inputs
  - coverage summary figures
  - per-transformation markdown reports

## Models

Two independently trained `CIFAR-10 ResNet50` checkpoints are used:

- `resnet50_wo_aug`: trained without data augmentation
- `resnet50_w_aug`: trained with data augmentation

The two models share the same architecture, but they were trained under different conditions. This difference is important for differential testing, because the goal is not just to compare two copies of the same network, but to compare models that may learn slightly different decision boundaries and feature preferences. In this project, clean disagreement between the two models is already non-trivial, which makes them suitable targets for DeepXplore-style testing.

## Outputs

When `--transformation all` is used, outputs are organized like this:

- `results/clean_disagreements/`
- `results/light/coverage_summary.png`
- `results/light/report.md`
- `results/light/generated_inputs/`
- `results/occl/coverage_summary.png`
- `results/occl/report.md`
- `results/occl/generated_inputs/`
- `results/black/coverage_summary.png`
- `results/black/report.md`
- `results/black/generated_inputs/`

Each `report.md` contains:

- model performance on clean test data
- clean disagreement count / rate
- baseline and updated neuron coverage
- generated disagreement-inducing inputs

Each generated figure shows:

- original seed image
- generated suspicious image
- absolute difference visualization
- predictions from both models before and after generation

If a single transformation is selected, files are written directly under the chosen `results/` directory instead of transformation-specific subdirectories.

## Included Result Snapshot

The repository already includes one full run under `results/`.

- clean disagreement count: `1883 / 10000`
- clean disagreement rate: `0.1883`
- `black`: `5` generated disagreement-inducing inputs, final average neuron coverage `0.4964`
- `occl`: `5` generated disagreement-inducing inputs, final average neuron coverage `0.4919`
- `light`: `0` generated disagreement-inducing inputs, final average neuron coverage `0.4913`

## Main Files

- `test.py`
- `utils.py`
- `requirements.txt`
- `ckpts/resnet50_wo_aug.pt`
- `ckpts/resnet50_w_aug.pt`
- `results/`
- `deepxplore/`
- `debug/debug.ipynb`

## Notes

- `test.py` automatically downloads `CIFAR-10` into `data/` if needed
- the `deepxplore/` folder is for reference only and is not the submission entry point
- `requirements.txt` lists packages needed for the recreated `PyTorch` pipeline, not the old legacy dependencies from the original DeepXplore examples
- `debug/debug.ipynb` was used for intermediate verification during implementation
