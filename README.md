# Drifting Models — PML Milestone 1

A runnable PyTorch implementation of "Generative Modeling via Drifting".

## Install (Windows PowerShell, Linux, macOS)

Use Python 3.10 or later

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` in bash. For a CPU installation:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m pytest -q
```

For CUDA, install a PyTorch build appropriate to your GPU before the editable
install; choose `--device cuda`. Included experiments were run on CPU. Exact
cross-device bitwise agreement is not promised. `environment-tested.txt` records
the packages used for the delivered runs; project dependencies use lower bounds
so other supported platforms can resolve appropriate wheels.

## Train, evaluate, generate

```sh
# A small initial sanity run (not enough for scientific conclusions)
python -m drifting.train --steps 50 --eval-every 25 --out runs/smoke

# All five datasets, three training seeds, and six controlled ablations per seed
python -m drifting.suite --out runs/full --steps 2000 --seeds 0 1 2 --ablations

# Recreate the delivered experiment budget
python -m drifting.suite --out runs/delivered --steps 1000 --seeds 0 1 2 --ablations

# Sample in ORIGINAL dataset units with the validation-selected checkpoint
python -m drifting.sample results/bimodal_seed0/best.pt --n 5000 --out generated.npy

# Rebuild aggregate CSV, Markdown table and overview figure
python -m drifting.summarize results
python -m drifting.analyze results
```

Each individual training output directory must be new (prevents accidental
replacement). The suite skips completed runs only when their configurations
match, and rejects incomplete directories; move aside an interrupted run before
rerunning it. There is no mid-run optimizer resume. Longer runs are a new
experiment and should use a fresh output directory.

## What is included

| Requirement | Implementation / evidence |
|---|---|
| Several 2D datasets | `data.py`: bimodal, 8-mode ring, moons, spiral, checkerboard |
| Three lazy Python generators with optional transforms | `make_loaders`; finite epochs, individual `(xi, yi)` pairs |
| Reproducible train/val/test split | 70/15/15; saved indices; training-only isotropic scaling |
| Extensible plots and evolution | `plots.py`; shared-axis snapshots, overlays, density mode |
| Drifting field and stop-gradient loss | `field.py`; paper Algorithm 2 and Eq. 10 variants |
| `Model(torch.nn.Module)` | `model.py`; configurable SiLU MLP, Gaussian noise input |
| Training all datasets, logging norm and saving samples | `train.py`; CSV, NPZ, checkpoints and figures |
| Paper-inspired toy experiments | Spiral/checkerboard; middle/far/concentrated bimodal starts |
| Evaluation and limitations | SWD, unbiased MMD², real–real floor, Gaussian baseline, mode metrics |
| Controlled comparisons | Small/large temperature, no repulsion, row-only field |
| Scientific correctness tests | Explicit triple sum, anti-symmetry, stop-gradient, geometry, data splits |
| Report preparation | `docs/METHOD.md`, `docs/REPORT_GUIDE.md`, measured `results/RESULTS.md` |
| AI disclosure | `docs/AI_DISCLOSURE.md` — adapt to your actual workflow |

Paths in the table are relative to `drifting/` unless explicitly stated.

## Read results correctly

`results/RESULTS.md` is generated from actual runs. Each run contains:

- `config.json`: hyperparameters, seeds, package versions and preprocessing.
- `splits.npz`: original row indices for auditing split disjointness.
- `history.csv`: loss and gradient norm each step; validation metrics periodically.
- `best.pt` / `last.pt`: validation-selected and final checkpoints.
- `samples.npz`: fixed-noise snapshots, selected-checkpoint samples with fresh noise,
  test reference, validation reference, and fixed latent inputs.
- `evolution.png` / `curves.png`: report figures.
- `metrics.json`: one final test evaluation plus reference baselines.

All metrics/plots use train-standardized units. The sampling CLI reverses that
normalization. Class/component labels are metadata only, never generator inputs.
There is no encoder, classifier-free guidance, diffusion sampler, discriminator,
or image generation in this milestone.

## Sources and scope

Deng et al., *Generative Modeling via Drifting*:
https://arxiv.org/abs/2602.04770 (implementation checked against v2,
https://arxiv.org/html/2602.04770v2, Algorithms 1–2 and Section 5.1).
Official demonstration:
https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb

This is an independently written educational implementation. It reproduces the
core field mathematically and studies the same qualitative toy questions;
it does **not** claim an exact numerical replication of the authors' experiments.
See `docs/METHOD.md` for all important deviations.

## Submission steps you must complete

1. Understand the implementation and write the coherent scientific report; omit
   code listings from it. Include your AI-use disclosure.
2. Create your GitHub repository, upload the source, create a "Milestone 1" GitHub
   milestone, and invite your actual supervisor. These account actions have not
   been performed by this package.
3. Submit the report through ISIS following your course's format/deadline.

