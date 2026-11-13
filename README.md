# Project Machine Learning: Generative Modeling via Drifting

A runnable PyTorch implementation of "Generative Modeling via Drifting".

## Setup Environment

Use Python 3.10 or later

```sh
pip install -r requirements.txt
```

For CUDA, install a PyTorch build appropriate to your GPU instead of the pytorch version in requirements.txt

## Run the experiments

Run commands from the project root.

To train on all toy datasets:

```sh
python run_all.py
```

This runs one experiment per dataset.

In `run_all.py`, you can change the output directory and device:

```python
results_folder = Path('runs')
seed = 42
device = 'cpu'  # Use 'cuda' for an available NVIDIA GPU.
```

Training settings are defined in `Config` in `drifting/train.py`. You can override them when constructing the configuration:

```python
config = Config(
    dataset=dataset,
    seed=seed,
    device=device,
    steps=2000,
    temperature=0.05,
)
```

To train on just one dataset:

```sh
python -m drifting.train --dataset spiral --out <output path>
```

For a short check that the pipeline works:

```sh
python -m drifting.train --dataset spiral --steps 50 --eval-every 25 --out runs/smoke
```

Existing experiment folders are not overwritten. Choose a new output directory when rerunning an experiment.

### Saved results

Each experiment produces:

| File or folder | Contents |
|---|---|
| `config.json` | Configuration, seeds, software versions, and preprocessing information |
| `best.pt` | Checkpoint with the lowest validation SWD |
| `last.pt` | Checkpoint after the final training update |
| `metrics.json` | Best and final checkpoint metrics, reference baselines, and runtime |
| `history.csv` | Training loss, mean squared drift, and validation metrics |
| `curves.png` | Loss and validation curves |
| `evolution/` | Separate sample plots at the saved training steps |
| `samples.npz` | Generated samples, reference points, and evaluation noise |
| `splits.npz` | Indices defining the fixed dataset splits |

## Using the individual components

### Create a toy dataset

Available datasets are `bimodal`, `ring`, `moons`, `spiral`, `checkerboard`, `heart`, `star`, and `hearts`.

```python
from drifting.data import make_dataset

points, labels = make_dataset('heart', n=12000, seed=123)

print(points.shape)  # (12000, 2): x and y coordinates
print(labels.shape)  # (12000,): component labels
```


### Data loading

`make_loaders()` returns three Python generators yielding individual `(point, label)` pairs.

```python
from drifting.data import make_dataset, make_loaders

points, labels = make_dataset('spiral', n=12000, seed=123)

train_loader, val_loader, test_loader = make_loaders(
    points,
    labels,
    ratio=(0.7, 0.15, 0.15),
    seed=321,
)

point, label = next(train_loader)
```

Each generator makes one finite pass through its split. Call `make_loaders()` again to create new generators.

An optional transformation is applied when each point is requested:

```python
import torch

train_loader, val_loader, test_loader = make_loaders(
    points,
    labels,
    transform=lambda point: torch.as_tensor(point, dtype=torch.float32),
)
```

To access the split arrays and optionally normalize:

```python
from drifting.data import split_dataset

splits = split_dataset(points, labels, normalize=False)

train_points = splits.train.x
validation_points = splits.val.x
test_points = splits.test.x
```

With `normalize=True`, all splits use the training split's mean and one shared scale for both coordinates, preserving the shapes

### Visualize points

```python
import matplotlib.pyplot as plt
from drifting.plots import plot_points

ax = plot_points(points, title='Heart dataset')
ax.legend()

ax.figure.savefig('heart.png', bbox_inches='tight')
plt.close(ax.figure)
```

Overlay generated samples by passing a second array:

```python
ax = plot_points(
    points,
    generated_points,
    title='Real and generated samples',
)

ax.legend()
ax.figure.savefig('comparison.png', bbox_inches='tight')
plt.close(ax.figure)
```

Here, `generated_points` is an `(N, 2)` array produced by your model.

Use `kind='density'` to display the real points as a density plot:

```python
ax = plot_points(points, kind='density', title='Sample density')
ax.figure.savefig('density.png', bbox_inches='tight')
plt.close(ax.figure)
```


### Compute the drifting field and loss

```python
import torch

from drifting.data import make_dataset
from drifting.field import compute_field, compute_loss
from drifting.model import Model

model = Model(noise_dim=32, hidden=256, depth=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

real_points, _ = make_dataset('spiral', n=256, seed=123)
real = torch.from_numpy(real_points)

noise = torch.randn(256, 32)
generated = model(noise)

# Inspect the field independently.
drift = compute_field(generated, real, temperature=0.05)
print(drift.shape)  # (256, 2)

# One optimization step. compute_loss() also returns the drift.
loss, drift = compute_loss(generated, real, temperature=0.05)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```


Paths in the table are relative to `drifting/` unless explicitly stated.

## Sources and scope

Deng et al., *Generative Modeling via Drifting*:
https://arxiv.org/abs/2602.04770
Official demonstration:
https://colab.research.google.com/github/lambertae/lambertae.github.io/blob/main/projects/drifting/notebooks/drifting_model_demo.ipynb

This is an independently written educational implementation.

