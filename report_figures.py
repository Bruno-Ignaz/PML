"""Figures (PDF) and LaTeX tables for the Milestone 1 report, built from outputs/.

    python report_figures.py OUTPUTS_DIR TARGET_DIR

Seed 42 runs are the folders without suffix (Hydra), seeds 43/44 end in -seed43/-seed44.
SWD is always reported relative to the real-real reference (SWD between two real
samples of the same size), so 1 means "as close as a second real sample".
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from drifting.data import NAMES
from drifting.plots import GENERATED_COLOR, REAL_COLOR, plot_points

TAUS = ('0.02', '0.05', '0.1', '0.2', '0.5')
BATCHES = ('256', '512', '1024', '2048', '4096')
SEEDS = (42, 43, 44)
# Five clearly distinct line colours (validated for colour-vision deficiency); the
# legend lists them in order of tau
TAU_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
FIELDS = [  # (folder key, LaTeX label, FID in paper Table 1)
    ('a1-r1', r'$V^+ - V^-$', 8.46),
    ('a1.5-r1', r'$1.5\,V^+ - V^-$', 41.05),
    ('a1-r1.5', r'$V^+ - 1.5\,V^-$', 46.28),
    ('a2-r1', r'$2\,V^+ - V^-$', 86.16),
    ('a1-r2', r'$V^+ - 2\,V^-$', 112.84),
    ('a1-r0', r'$V^+$ only', 177.14),
]
TEXT_WIDTH = 4.95  # inches, text width of the report template

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'savefig.dpi': 600,  # for rasterized point clouds
    'axes.spines.top': False, 'axes.spines.right': False,
})

OUT = Path(sys.argv[1])
TARGET = Path(sys.argv[2])
TARGET.mkdir(parents=True, exist_ok=True)


# ---------- loading ----------

def folder(run, dataset, seed=42):
    name = f'{run}-seed{seed}'
    if seed == 42 and not (OUT / name).exists():
        name = run  # seed-42 runs from Hydra have no suffix
    return OUT / name / 'data' / dataset


def metrics(run, dataset, seed=42):
    path = folder(run, dataset, seed) / 'metrics.json'
    return json.loads(path.read_text()) if path.exists() else None


def ratio(m, checkpoint='best'):
    return m[checkpoint]['test_swd'] / m['baselines']['real_real']['swd']


def seed_ratios(run, dataset, checkpoint='best'):
    return [ratio(m, checkpoint) for s in SEEDS if (m := metrics(run, dataset, s))]


def samples(run, dataset, seed=42):
    return np.load(folder(run, dataset, seed) / 'samples.npz')


def history(run, dataset, seed=42):
    rows = list(csv.DictReader((folder(run, dataset, seed) / 'history.csv').open()))
    loss = np.array([[int(r['step']), float(r['loss'])] for r in rows if r['loss']])
    val = np.array([[int(r['step']), float(r['val_swd'])] for r in rows if r['val_swd']])
    return loss, val


def ablation_run(key):
    return 't0.1-norm-5000val_samples' if key == 'a1-r1' else f't0.1-norm-{key}-5000val_samples'


def init_run(init, tau):
    return f't{tau}-norm-5000val_samples' if init == 'origin' else f't{tau}-norm-init{init}-5000val_samples'


# ---------- point clouds ----------

def view(*arrays, pad=0.08):
    """Square view around all given point sets."""
    points = np.concatenate([a for a in arrays if a is not None])
    low, high = points.min(axis=0), points.max(axis=0)
    centre, half = (low + high) / 2, (1 + 2 * pad) * (high - low).max() / 2
    return centre, half


def cloud(ax, real, generated=None, box=None, count=700):
    rng = np.random.default_rng(0)
    pick = lambda a: a[rng.choice(len(a), min(count, len(a)), replace=False)]
    ax.scatter(*pick(real).T, s=0.8, lw=0, alpha=0.3, color=REAL_COLOR, rasterized=True)
    if generated is not None:
        ax.scatter(*pick(generated).T, s=0.8, lw=0, alpha=0.6, color=GENERATED_COLOR, rasterized=True)
    centre, half = box or view(real)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for side in ('top', 'right', 'bottom', 'left'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color('0.85')


def corner(ax, text):
    ax.text(0.97, 0.03, text, transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))


def save(figure, name):
    figure.savefig(TARGET / name, bbox_inches='tight', pad_inches=0.02)
    plt.close(figure)
    print(TARGET / name)


def legend_below(figure, generated_label='generated', gap=0.012):
    """Legend centred just below the lowest axes, including their tick labels."""
    figure.canvas.draw()  # resolves equal-aspect axes to their final size
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()
    boxes = [ax.get_tightbbox(renderer).transformed(to_figure) for ax in figure.axes]
    positions = [ax.get_position() for ax in figure.axes]
    centre = (min(p.x0 for p in positions) + max(p.x1 for p in positions)) / 2
    handles = [plt.Line2D([], [], ls='', marker='o', ms=3, color=REAL_COLOR, label='real (test/validation)'),
               plt.Line2D([], [], ls='', marker='o', ms=3, color=GENERATED_COLOR, label=generated_label)]
    figure.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                  bbox_to_anchor=(centre, min(b.y0 for b in boxes) - gap))


# ---------- figures ----------

def fig_datasets():
    """The nine toy distributions in original coordinates, drawn with plot_points."""
    figure, axes = plt.subplots(3, 3, figsize=(TEXT_WIDTH, TEXT_WIDTH + 0.2))
    for i, (ax, name) in enumerate(zip(axes.flat, NAMES)):
        points = samples('t0.1', name)['reference']  # raw test points
        plot_points(points, ax=ax, title=name)
        # 1000 points per panel: vector graphics stay sharp at any zoom
        ax.collections[0].set_sizes([2.5])
        ax.collections[0].set_alpha(0.5)
        ax.collections[0].set_linewidth(0)
        ax.collections[0].set_rasterized(False)
        ax.set_xlabel('$x_1$' if i >= 6 else '')
        ax.set_ylabel('$x_2$' if i % 3 == 0 else '')
    figure.tight_layout(h_pad=0.6, w_pad=0.4)
    save(figure, 'datasets.pdf')


def fig_evolution():
    """Paper Fig. 4: spiral and checkerboard with the official demo settings (raw data, tau 0.05)."""
    steps = ('0', '50', '200', '1000', '2000')
    figure = plt.figure(figsize=(TEXT_WIDTH, 2.25))
    grid = figure.add_gridspec(2, len(steps) + 2, width_ratios=[1] * len(steps) + [0.25, 1.9], wspace=0.08, hspace=0.25)
    for row, name in enumerate(('spiral', 'checkerboard')):
        s = samples('t0.05', name)
        real = s['validation_reference']
        for col, step in enumerate(steps):
            ax = figure.add_subplot(grid[row, col])
            cloud(ax, real, s[f'step_{step}'], view(real))
            if row == 0:
                ax.set_title(step, pad=2)
            if col == 0:
                ax.set_ylabel(name, labelpad=2)
        ax = figure.add_subplot(grid[row, -1])
        loss, _ = history('t0.05', name)
        smooth = np.convolve(loss[:, 1], np.ones(25) / 25, mode='valid')
        ax.plot(loss[:, 0], loss[:, 1], color='0.82', lw=0.5)
        ax.plot(loss[12:12 + len(smooth), 0], smooth, color=GENERATED_COLOR, lw=1.2)
        ax.set_yscale('log')
        ax.set_xlim(0, 2000)
        ax.tick_params(pad=1)
        # Ticks on the right keep the labels clear of the scatter panels
        ax.yaxis.tick_right()
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(False)
        if row == 0:
            ax.set_title(r'loss $\|V\|^2/2$', pad=2)
            ax.set_xticklabels([])
        else:
            ax.set_xlabel('training step', labelpad=1)
    legend_below(figure)
    save(figure, 'evolution.pdf')


def fig_init():
    """Paper Fig. 3: three start states on bimodal (normalized), tau 0.1 and 0.2, seed 42."""
    steps = ('0', '200', '2000')
    inits = (('origin', 'between'), ('far', 'far'), ('mode', 'collapsed'))
    taus = ('0.1', '0.2')
    real = samples(init_run('origin', '0.1'), 'bimodal')['validation_reference']
    starts = [samples(init_run(i, t), 'bimodal')['step_0'] for i, _ in inits for t in taus]
    box = view(real, *starts)
    figure = plt.figure(figsize=(TEXT_WIDTH, 2.55))
    grid = figure.add_gridspec(3, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1], wspace=0.06, hspace=0.06)
    for block, tau in enumerate(taus):
        for row, (init, label) in enumerate(inits):
            s = samples(init_run(init, tau), 'bimodal')
            for col, step in enumerate(steps):
                ax = figure.add_subplot(grid[row, block * 4 + col])
                cloud(ax, real, s[f'step_{step}'], box)
                if row == 0:
                    ax.set_title(step, pad=2)
                if col == 0 and block == 0:
                    ax.set_ylabel(label, labelpad=2)
    # Block headers centred over their three columns, above the step titles
    figure.canvas.draw()
    renderer, to_figure = figure.canvas.get_renderer(), figure.transFigure.inverted()
    top_row = sorted(figure.axes, key=lambda a: -a.get_position().y1)[:6]  # first row of both blocks
    top_row.sort(key=lambda a: a.get_position().x0)
    for block, tau in enumerate(taus):
        axes = top_row[block * 3:(block + 1) * 3]
        x = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
        y = max(a.get_tightbbox(renderer).transformed(to_figure).y1 for a in axes)
        figure.text(x, y + 0.01, rf'$\tau = {tau}$', ha='center', va='bottom', fontsize=8.5)
    legend_below(figure)
    save(figure, 'init_states.pdf')


def fig_tau_sweep():
    """All datasets x temperatures, normalized data, seed 42, best checkpoint vs test points."""
    figure, axes = plt.subplots(len(NAMES), len(TAUS), figsize=(TEXT_WIDTH, 8.0),
                                gridspec_kw=dict(wspace=0.05, hspace=0.05))
    for row, name in enumerate(NAMES):
        box = None
        for col, tau in enumerate(TAUS):
            run = f't{tau}-norm-5000val_samples'
            s = samples(run, name)
            box = box or view(s['reference'])
            cloud(axes[row, col], s['reference'], s['best'], box)
            corner(axes[row, col], f'{ratio(metrics(run, name)):.2f}' if ratio(metrics(run, name)) < 10 else f'{ratio(metrics(run, name)):.0f}')
            if row == 0:
                axes[row, col].set_title(rf'$\tau = {tau}$', pad=3)
            if col == 0:
                axes[row, col].set_ylabel(name, labelpad=2)
    legend_below(figure, 'generated (best checkpoint)')
    save(figure, 'tau_sweep.pdf')


def fig_failures():
    """Typical failure modes, last checkpoint."""
    cases = [  # details in the report caption
        ('t0.02-norm-5000val_samples', 'ring', '(a)'),
        ('t0.05', 'bimodal', '(b)'),
        ('t0.5-norm-5000val_samples', 'ring', '(c)'),
        ('t0.1-norm-5000val_samples', 'grid', '(d)'),
        ('t0.1-norm-b2048-steps8000-5000val_samples', 'grid', '(e)'),
    ]
    figure, axes = plt.subplots(1, len(cases), figsize=(TEXT_WIDTH, 1.45), gridspec_kw=dict(wspace=0.06))
    for ax, (run, name, title) in zip(axes, cases):
        s = samples(run, name)
        cloud(ax, s['reference'], s['last'], view(s['reference']))
        ax.set_title(title, pad=2, fontsize=7)
    legend_below(figure, 'generated (last checkpoint)')
    save(figure, 'failures.pdf')


def fig_ablation():
    """Paper Table 1 on ring: weights of attraction and repulsion, seed 42, best checkpoint."""
    figure, axes = plt.subplots(1, len(FIELDS), figsize=(TEXT_WIDTH, 1.2), gridspec_kw=dict(wspace=0.06))
    for ax, (key, label, _) in zip(axes, FIELDS):
        s = samples(ablation_run(key), 'ring')
        cloud(ax, s['reference'], s['best'])
        ax.set_title(label, pad=2, fontsize=7)
    legend_below(figure)
    save(figure, 'ablation_ring.pdf')


def fig_loss():
    """Loss and validation SWD per temperature on ring (normalized, seed 42)."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.9), gridspec_kw=dict(wspace=0.35))
    for tau, color in zip(TAUS, TAU_COLORS):
        loss, val = history(f't{tau}-norm-5000val_samples', 'ring')
        smooth = np.convolve(loss[:, 1], np.ones(15) / 15, mode='valid')
        left.plot(loss[7:7 + len(smooth), 0], smooth, color=color, lw=1.1, label=rf'$\tau={tau}$')
        right.plot(val[:, 0], val[:, 1], color=color, lw=1.1, marker='o', ms=2.5)
    left.set(xscale='log', yscale='log', xlabel='training step', ylabel=r'loss $\|V\|^2/2$ (smoothed)')
    right.set(yscale='log', xlabel='training step', ylabel='validation SWD')
    left.legend(frameon=False, ncol=1, loc='lower right', handlelength=1.2)
    for ax in (left, right):
        ax.grid(alpha=0.25, lw=0.4)
    save(figure, 'loss_vs_swd.pdf')


# ---------- tables ----------

def fmt(value):
    return f'{value:.1f}' if value >= 10 else f'{value:.2f}'


def mean_std(values):
    values = np.asarray(values)
    return values.mean(), (values.std(ddof=1) if len(values) > 1 else 0.0)


def table_tau():
    lines = [r'\begin{tabular}{l' + 'c' * len(TAUS) + '}', r'\hline',
             ' & ' + ' & '.join(rf'$\tau={t}$' for t in TAUS) + r' \\', r'\hline']
    for name in NAMES:
        stats = [mean_std(seed_ratios(f't{t}-norm-5000val_samples', name)) for t in TAUS]
        best = int(np.argmin([m for m, _ in stats]))
        cells = [(r'$\mathbf{%s \pm %s}$' if i == best else r'$%s \pm %s$') % (fmt(m), fmt(s)) for i, (m, s) in enumerate(stats)]
        lines.append(f'{name} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    write('table_tau.tex', lines)


def table_ablation():
    per_dataset = {k: {n: np.mean(seed_ratios(ablation_run(k), n)) for n in NAMES} for k, _, _ in FIELDS}
    wins = {k: 0 for k, _, _ in FIELDS}
    for n in NAMES:
        wins[min(per_dataset, key=lambda k: per_dataset[k][n])] += 1
    lines = [r'\begin{tabular}{lrcc}', r'\hline',
             r'drifting field & FID (paper) & SWD ratio (ours) & best on \\', r'\hline']
    for key, label, fid in FIELDS:
        medians = [np.median([ratio(metrics(ablation_run(key), n, s)) for n in NAMES]) for s in SEEDS]
        m, sd = mean_std(medians)
        lines.append(rf'{label} & {fid:.2f} & ${m:.2f} \pm {sd:.2f}$ & {wins[key]}/9 \\')
    lines += [r'\hline', r'\end{tabular}']
    write('table_ablation.tex', lines)


def table_init():
    labels = {'origin': 'between the modes', 'far': 'far from both', 'mode': 'collapsed on one'}
    taus = ('0.05', '0.1', '0.2')
    lines = [r'\begin{tabular}{ll' + 'c' * len(taus) + '}', r'\hline',
             r'data & start & ' + ' & '.join(rf'$\tau={t}$' for t in taus) + r' \\', r'\hline']
    for name in ('bimodal', 'ring'):
        for init in ('origin', 'far', 'mode'):
            cells = []
            for tau in taus:
                ms = [metrics(init_run(init, tau), name, s) for s in SEEDS]
                full = sum(m['last']['mode_coverage'] == 1 for m in ms)
                cells.append(f'{full}/3 ({fmt(np.mean([ratio(m, "last") for m in ms]))})')
            lines.append(f'{name if init == "origin" else ""} & {labels[init]} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    write('table_init.tex', lines)


def table_batch():
    lines = [r'\begin{tabular}{l' + 'c' * len(BATCHES) + '}', r'\hline',
             ' & ' + ' & '.join(f'$B={b}$' for b in BATCHES) + r' \\', r'\hline']
    for tau in TAUS:
        medians = [np.median([ratio(metrics(f't{tau}-norm-b{b}-steps8000-5000val_samples', n)) for n in NAMES]) for b in BATCHES]
        lines.append(rf'$\tau={tau}$ & ' + ' & '.join(fmt(m) for m in medians) + r' \\')
    lines.append(r'\hline')
    coverage = [metrics(f't0.1-norm-b{b}-steps8000-5000val_samples', 'grid')['best']['mode_coverage'] for b in BATCHES]
    lines.append(r'grid modes, $\tau=0.1$ & ' + ' & '.join(f'{round(c * 25)}/25' for c in coverage) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    write('table_batch.tex', lines)


def write(name, lines):
    (TARGET / name).write_text('\n'.join(lines) + '\n')
    print(TARGET / name)


if __name__ == '__main__':
    fig_datasets()
    fig_evolution()
    fig_init()
    fig_tau_sweep()
    fig_failures()
    fig_ablation()
    fig_loss()
    table_tau()
    table_ablation()
    table_init()
    table_batch()
