"""Train a 2D drifting generator

We follow the official toy demo's architecture sizes, batch size, optimizer,
learning rate, temperature, step count and numerical loss Validation selection

Run from the project root: python -m drifting.train --dataset spiral --out runs/spiral
"""

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from itertools import islice

import numpy as np
import torch

from .data import NAMES, make_dataset, split_dataset, make_loaders
from .field import compute_loss
from .metrics import evaluate, mode_metrics
from .model import Model
from .plots import evolution, curves

INITS = ('origin', 'far', 'mode')

@dataclass
class Config:
    # Dataset and seeds
    dataset: str = 'spiral'
    seed: int = 42
    data_seed: int = 123
    split_seed: int = 321
    n: int = 12000  # Size of the fixed reference dataset, NOT a training-data limit
    normalize: bool = False

    # Official toy-demo settings
    steps: int = 2000
    batch: int = 2048  # both real and generated samples per update
    hidden: int = 256
    depth: int = 3
    noise_dim: int = 32
    lr: float = 0.001
    temperature: float = 0.2
    # Weights of attraction V+ and repulsion V- (paper Table 1); only 1, 1 is anti-symmetric
    attraction: float = 1.0
    repulsion: float = 1.0
    # Start state (paper Fig. 3): 'origin' (untrained output, a small blob at 0),
    # 'far' (blob init_distance RMS units above the data centre), 'mode' (blob on the
    # centre of mixture component 0)
    init: str = 'origin'
    init_distance: float = 3.0

    # Validation, saved snapshots and hardware
    eval_every: int = 500
    eval_n: int = 1800
    device: str = 'cpu'
    threads: int = 1

def train(c: Config, out):
    """Train, select by validation SWD, then evaluate the selected model on test."""
    if c.dataset not in NAMES:
        raise ValueError(f'Choose a dataset from {NAMES}')
    if min(c.steps, c.eval_every, c.threads, c.hidden, c.depth, c.noise_dim) < 1:
        raise ValueError('Training counts and model dimensions must be positive')
    if c.batch < 20 or c.n < 20:
        raise ValueError('make_dataset requires n and batch to be at least 20')
    if c.eval_n < 2:
        raise ValueError('Evaluation needs at least two points')
    if not np.isfinite(c.temperature) or c.temperature <= 0:
        raise ValueError('Temperature must be finite and positive')
    if not np.isfinite(c.lr) or c.lr <= 0:
        raise ValueError('Learning rate must be finite and positive')
    if c.init not in INITS:
        raise ValueError(f'Choose init from {INITS}')

    device = torch.device(c.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested, but unavailable; use --device cpu')

    # Never overwrite an existing experiment directory
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=False)
    start_time = time.perf_counter()

    torch.set_num_threads(c.threads)
    torch.manual_seed(c.seed)

    reference_points, reference_labels = make_dataset(c.dataset, c.n, c.data_seed)
    splits = split_dataset(
        reference_points,
        reference_labels,
        seed=c.split_seed,
        normalize=c.normalize,
    )
    evaluation_count = min(
        c.eval_n, len(splits.train.x), len(splits.val.x), len(splits.test.x)
    )
    epoch = 0
    shuffle_seed = c.seed

    #Data loaders
    train_loader, val_loader, test_loader = make_loaders(
        reference_points,
        reference_labels,
        seed=c.split_seed,
        normalize=c.normalize
    )

    validation_points = np.stack([
        point for point, _ in islice(val_loader, evaluation_count)
    ])

    # Independent streams keep evaluation from changing the training trajectory
    noise_rng = torch.Generator(device=device).manual_seed(c.seed + 1000)
    validation_rng = torch.Generator(device=device).manual_seed(1234)
    fixed_noise = torch.randn(
        evaluation_count, c.noise_dim, generator=validation_rng, device=device
    )

    model = Model(
        noise_dim=c.noise_dim,
        hidden=c.hidden,
        depth=c.depth,
    ).to(device)

    # The untrained generator outputs a small blob near the origin. Shifting the
    # last layer's bias moves that blob to the requested start without changing it.
    if c.init != 'origin':
        train_points = splits.train.x
        if c.init == 'mode':
            start = train_points[splits.train.y == 0].mean(axis=0)
        else:
            centre = train_points.mean(axis=0)
            spread = np.sqrt(np.mean((train_points - centre) ** 2))
            start = centre + np.array([0.0, c.init_distance * spread])
        with torch.no_grad():
            model.net[-1].bias += torch.as_tensor(start, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)

    metadata = {
        'config': asdict(c),
        'python': platform.python_version(),
        'torch': str(torch.__version__),
        'numpy': np.__version__,
        'device': str(device),
        'normalization': {'mean': splits.mean.tolist(), 'scale': splits.scale},
        'selection': 'Best checkpoint selected by minimum validation SWD, best and last evaluated on identical test points and noise',
        'loss_convention': 'Coordinate MSE: loss = drift_norm_squared / 2 for 2D',
    }
    (output_dir / 'config.json').write_text(json.dumps(metadata, indent=2))
    np.savez_compressed(
        output_dir / 'splits.npz',
        train=splits.train.indices,
        val=splits.val.indices,
        test=splits.test.indices,
    )

    history = []
    snapshots = {}
    # Save the same training stages shown in the paper
    snapshot_steps = {0, 50, 100, 200, 500, 1000, 2000}
    snapshot_steps = {step for step in snapshot_steps if step <= c.steps}
    snapshot_steps.add(c.steps)  # Always include the final step
    best_validation_swd = float('inf')
    best_step = 0

    def save_checkpoint(filename, step):
        torch.save(
            {
                'model': model.state_dict(),
                'config': asdict(c),
                'mean': torch.tensor(splits.mean),
                'scale': splits.scale,
                'step': step,
            },
            output_dir / filename,
        )

    def validate(step):
        """Evaluate fixed validation noise; never inspect the test set here"""
        nonlocal best_validation_swd, best_step
        model.eval()
        with torch.no_grad():
            generated_points = model(fixed_noise).cpu().numpy()

        scores = evaluate(generated_points, validation_points)
        if not all(np.isfinite(value) for value in scores.values()):
            raise FloatingPointError(f'Nonfinite validation metric at step {step}')

        if step in snapshot_steps:
            snapshots[step] = generated_points.copy()

        if scores['swd'] < best_validation_swd:
            best_validation_swd = scores['swd']
            best_step = step
            save_checkpoint('best.pt', step)

        model.train()
        return {'step': step, 'val_swd': scores['swd'], 'val_mmd2': scores['mmd2']}

    #Validate at step 0
    history.append(validate(0))

    for step in range(1, c.steps + 1):
        batch = list(islice(train_loader, c.batch))

        # Restart after empty batch
        if len(batch) < c.batch:
            epoch += 1
            train_loader = splits.train.stream(
                shuffle=True,
                seed=shuffle_seed + epoch,
            )
            batch = list(islice(train_loader, c.batch))

        real = torch.as_tensor(
            np.stack([point for point, _ in batch]),
            dtype=torch.float32,
            device=device,
        )

        noise = torch.randn(len(real), c.noise_dim, generator=noise_rng, device=device)
        generated = model(noise)

        loss, drift = compute_loss(generated, real, c.temperature, c.attraction, c.repulsion)

        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite training loss at step {step}')

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        row = {
            'step': step,
            'loss': loss.detach().item(),
            'drift_norm_squared': drift.square().sum(dim=1).mean().item(),
        }
        if step % c.eval_every == 0 or step in snapshot_steps:
            row.update(validate(step))
            print(
                f'Step {step:5d}/{c.steps} | loss {row["loss"]:.6f} | '
                f'validation SWD {row["val_swd"]:.4f}',
                flush=True,
            )
        history.append(row)

    save_checkpoint('last.pt', c.steps)

    # Use identical test points and noise
    test_points = np.stack([
        point for point, _ in islice(test_loader, evaluation_count)
    ])
    test_rng = torch.Generator(device=device).manual_seed(71234)
    test_noise = torch.randn(
        evaluation_count,
        c.noise_dim,
        generator=test_rng,
        device=device,
    )

    result = {}
    checkpoint_samples = []

    for name in ('best', 'last'):
        checkpoint = torch.load(
            output_dir / f'{name}.pt',
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint['model'])
        model.eval()

        with torch.no_grad():
            generated_points = model(test_noise).cpu().numpy()

        checkpoint_samples.append(generated_points)

        # Retrieve the validation scores recorded at this checkpoint's step.
        validation_row = next(
            row for row in history
            if row['step'] == checkpoint['step'] and 'val_swd' in row
        )

        test_scores = evaluate(generated_points, test_points)
        scores = {
            'step': checkpoint['step'],
            'val_swd': validation_row['val_swd'],
            'val_mmd2': validation_row['val_mmd2'],
            'test_swd': test_scores['swd'],
            'test_mmd2': test_scores['mmd2'],
        }

        # Additional mode diagnostics for Gaussian mixtures.
        if c.dataset in ('bimodal', 'ring', 'grid'):
            if c.dataset == 'bimodal':
                centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
                sigma = 0.25
            elif c.dataset == 'ring':
                angles = np.arange(8) * np.pi / 4
                centers = 2 * np.column_stack([
                    np.cos(angles),
                    np.sin(angles),
                ])
                sigma = 0.12
            else:
                coords = np.arange(5) - 2.0
                centers = np.array([[a, b] for a in coords for b in coords])
                sigma = 0.05

            centers = (centers - splits.mean) / splits.scale
            scores.update(
                mode_metrics(
                    generated_points,
                    centers,
                    sigma / splits.scale,
                )
            )

        if not all(np.isfinite(value) for value in scores.values()):
            raise FloatingPointError(f'Nonfinite metrics for {name} checkpoint')

        result[name] = scores

    best_samples, last_samples = checkpoint_samples

    # Baselines are shared by both checkpoints
    baseline_rng = np.random.default_rng(71234)
    gaussian_points = baseline_rng.normal(size=(evaluation_count, 2))
    gaussian_points = (gaussian_points - splits.mean) / splits.scale

    baselines = {
        'real_real': splits.train.x[:evaluation_count],
        'gaussian': gaussian_points,
    }

    result['baselines'] = {}

    for name, points in baselines.items():
        scores = evaluate(points, test_points)

        if not all(np.isfinite(value) for value in scores.values()):
            raise FloatingPointError(f'Nonfinite {name} baseline metrics')

        result['baselines'][name] = scores

    columns = ['step', 'loss', 'drift_norm_squared', 'val_swd', 'val_mmd2']
    with (output_dir / 'history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)

    saved_samples = {f'step_{step}': points for step, points in snapshots.items()}
    np.savez_compressed(
        output_dir / 'samples.npz',
        **saved_samples,
        best=best_samples,
        last=last_samples,
        reference=test_points,
        validation_reference=validation_points,
        fixed_noise=fixed_noise.cpu().numpy(),
        test_noise=test_noise.cpu().numpy(),
    )

    # Save sample evolution and training curves
    evolution(
        validation_points,
        snapshots,
        output_dir / 'evolution.png',
    )
    curves(history, output_dir / 'curves.png')

    # Includes initialization, training, evaluation, saving and plotting.
    result['seconds'] = time.perf_counter() - start_time
    (output_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(f'Finished. Best checkpoint: step {best_step}. Results: {output_dir}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = asdict(Config())
    choices = {
        'dataset': NAMES,
        'init': INITS,
    }
    for name, default in defaults.items():
        argument = '--' + name.replace('_', '-')
        if isinstance(default, bool):
            parser.add_argument(argument, action=argparse.BooleanOptionalAction, default=default)
        else:
            options = {'type': type(default), 'default': default}
            if name in choices:
                options['choices'] = choices[name]
            parser.add_argument(argument, **options)

    parser.add_argument('--out', required=True, help='New directory for this experiment')
    arguments = vars(parser.parse_args())
    output_directory = arguments.pop('out')
    train(Config(**arguments), output_directory)


if __name__ == '__main__':
    main()
