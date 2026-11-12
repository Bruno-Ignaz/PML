"""Visualize toy datasets, training evolution, and metrics"""

import matplotlib
matplotlib.use('Agg')  # Save figures without opening a window

import json
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from .data import NAMES


# Shared appearance for all report figures
plt.rcParams.update({
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 140,
    'savefig.dpi': 300,
})

REAL_COLOR = '#247ba0'
GENERATED_COLOR = '#e87722'


def plot_points(
    points,
    generated=None,
    *,
    ax=None,
    title='',
    kind='scatter',
    limits=None,
):
    """Plot real points and optionally overlay generated points

    points, generated: lists or arrays with shape (N, 2)
    ax: an existing subplot, or None to create a new figure
    kind: 'scatter' or 'density' for the real points
    limits: optional ((xmin, xmax), (ymin, ymax))
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(4, 4),
            layout='constrained',
        )

    points = np.asarray(points)

    if kind == 'scatter':
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=4,
            alpha=0.35,
            color=REAL_COLOR,
            label='Real samples',
            rasterized=True,
        )

    elif kind == 'density':
        # Darker cells contain more real points
        ax.hexbin(
            points[:, 0],
            points[:, 1],
            gridsize=50,
            mincnt=1,
            cmap='Blues',
            rasterized=True,
        )

    else:
        raise ValueError("kind must be 'scatter' or 'density'")

    if generated is not None:
        generated = np.asarray(generated)

        ax.scatter(
            generated[:, 0],
            generated[:, 1],
            s=4,
            alpha=0.35,
            color=GENERATED_COLOR,
            label='Generated samples',
            rasterized=True,
        )

    ax.set_title(title)
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')

    # Equal coordinate scales
    ax.set_aspect('equal', adjustable='box')

    if limits is not None:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])

    return ax


def evolution(real, snapshots, path):
    """Save snapshots with a fixed view based on the real data."""
    output_folder = Path(path).with_suffix('')
    output_folder.mkdir(parents=True, exist_ok=True)

    real = np.asarray(real)

    # Calculate the view once, keeping real points stationary across steps.
    minimum = real.min(axis=0)
    maximum = real.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-6)

    padding = 0.15 * span
    lower = minimum - padding
    upper = maximum + padding

    limits = (
        (lower[0], upper[0]),
        (lower[1], upper[1]),
    )

    width = 6
    height = float(np.clip(width * span[1] / span[0] + 1, 3, 8))

    for step in sorted(snapshots):
        generated = np.asarray(snapshots[step])

        # Report generated points outside the fixed view.
        outside = np.any(
            (generated < lower) | (generated > upper),
            axis=1,
        )
        outside_count = int(outside.sum())

        figure, ax = plt.subplots(figsize=(width, height))

        plot_points(
            real,
            generated,
            ax=ax,
            title=f'Step {step}',
            limits=limits,
        )

        # Fixed layout and legend position for every snapshot.
        figure.subplots_adjust(
            left=0.15, right=0.97, bottom=0.22, top=0.88
        )

        handles, labels = ax.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.02),
            ncol=2,
            frameon=False,
            fontsize=9,
            markerscale=2,
        )

        """
        figure.suptitle(
            f'Generated points outside view: {outside_count}/{len(generated)}',
            fontsize=9,
            y=0.98,
        )"""

        # Fixed canvas size keeps all saved frames aligned.
        figure.savefig(output_folder / f'step_{step:04d}.png')
        plt.close(figure)


def curves(history, path):
    """Save the actual training loss and metrics"""
    panels = [
        ('loss', 'Drifting loss'),
        ('val_swd', 'Validation SWD'),
        ('val_mmd2', 'Validation MMD²'),
    ]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12, 3.5),
        layout='constrained',
    )

    for ax, (key, title) in zip(axes, panels):
        recorded_rows = [row for row in history if key in row]

        steps = [row['step'] for row in recorded_rows]
        values = [row[key] for row in recorded_rows]

        ax.plot(
            steps,
            values,
            color=REAL_COLOR,
            linewidth=1.2,
        )

        ax.set_title(title)
        ax.set_xlabel('Training step')
        ax.grid(alpha=0.2)

        if key == 'loss':
            if values and np.all(np.asarray(values) > 0):
                ax.set_yscale('log')

    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)


def overview(run_dir):
    """One panel per dataset: test points vs. samples of the best checkpoint.

    run_dir is a run folder created by run.sh (with data/<dataset>/ inside).
    Saves run_dir/overview.png and returns its path.
    """
    run_dir = Path(run_dir)
    datasets = [name for name in NAMES if (run_dir / 'data' / name / 'metrics.json').exists()]
    if not datasets:
        raise FileNotFoundError(f'No finished datasets in {run_dir / "data"}')

    columns = math.ceil(math.sqrt(len(datasets)))
    rows = math.ceil(len(datasets) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(2.7 * columns, 2.9 * rows), squeeze=False
    )

    for ax, name in zip(axes.flat, datasets):
        folder = run_dir / 'data' / name
        metrics = json.loads((folder / 'metrics.json').read_text())
        samples = np.load(folder / 'samples.npz')
        real, generated = samples['reference'], samples['best']

        ax.scatter(*real.T, s=1, alpha=0.3, color=REAL_COLOR, rasterized=True)
        ax.scatter(*generated.T, s=1, alpha=0.5, color=GENERATED_COLOR, rasterized=True)

        # Square view around the real data with 15 % padding, so all panels align
        lower, upper = real.min(axis=0), real.max(axis=0)
        center = (lower + upper) / 2
        half = 0.5 * 1.15 * max((upper - lower).max(), 1e-6)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_aspect('equal')
        ax.axis('off')

        best = metrics['best']
        subtitle = (f"SWD {best['test_swd']:.3f} · real {metrics['baselines']['real_real']['swd']:.3f}")
        if 'mode_coverage' in best:
            subtitle += f"\nmode coverage {best['mode_coverage']:.2f}"
        ax.set_title(f'{name}\n{subtitle}', fontsize=9)

    for ax in list(axes.flat)[len(datasets):]:
        ax.set_visible(False)

    config = json.loads((run_dir / 'data' / datasets[0] / 'config.json').read_text())['config']
    figure.suptitle(f"τ = {config['temperature']}, {config['steps']} steps")
    figure.legend(
        handles=[
            plt.Line2D([], [], ls='', marker='o', color=REAL_COLOR, label='Test data'),
            plt.Line2D([], [], ls='', marker='o', color=GENERATED_COLOR,
                       label='Generated (best checkpoint)'),
        ],
        loc='lower center',
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    path = run_dir / 'overview.png'
    figure.savefig(path, bbox_inches='tight')
    plt.close(figure)
    return path


if __name__ == '__main__':
    # python -m drifting.plots outputs/RUN_NAME [outputs/OTHER_RUN ...]
    import sys
    for run in sys.argv[1:]:
        print(overview(run))