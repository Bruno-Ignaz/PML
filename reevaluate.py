"""Evaluate saved best/last checkpoints on the full test split.

    python reevaluate.py SOURCE_OUTPUTS TARGET_OUTPUTS
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from drifting.data import make_dataset, split_dataset
from drifting.metrics import mode_metrics, sliced_wasserstein
from drifting.model import Model

SOURCE, TARGET = Path(sys.argv[1]), Path(sys.argv[2])
NOISE_SEED = 71234
MIXTURES = {
    'bimodal': (np.array([[-2.0, 0.0], [2.0, 0.0]]), 0.25),
    'ring': (2 * np.column_stack([np.cos(np.arange(8) * np.pi / 4), np.sin(np.arange(8) * np.pi / 4)]), 0.12),
    'grid': (np.array([[a, b] for a in np.arange(5) - 2.0 for b in np.arange(5) - 2.0]), 0.05),
}
split_cache = {}


def splits_for(c):
    key = (c['dataset'], c['n'], c['data_seed'], c['split_seed'], c['normalize'])
    if key not in split_cache:
        x, y = make_dataset(c['dataset'], c['n'], c['data_seed'])
        split_cache[key] = split_dataset(x, y, seed=c['split_seed'], normalize=c['normalize'])
    return split_cache[key]


folders = sorted(p.parent for p in SOURCE.glob('*/data/*/best.pt'))
torch.set_grad_enabled(False)
for k, folder in enumerate(folders):
    out = TARGET / folder.relative_to(SOURCE)
    out.mkdir(parents=True, exist_ok=True)
    c = json.loads((folder / 'config.json').read_text())['config']
    sp = splits_for(c)
    test = sp.test.x
    count = len(test)

    original = np.load(folder / 'samples.npz')['reference']
    if not np.allclose(original, test[:len(original)], atol=1e-5):
        raise RuntimeError(f'{folder}: test split differs from the original evaluation')

    noise = torch.randn(count, c['noise_dim'], generator=torch.Generator().manual_seed(NOISE_SEED))
    model = Model(noise_dim=c['noise_dim'], hidden=c['hidden'], depth=c['depth'])
    result = {}
    for name in ('best', 'last'):
        checkpoint = torch.load(folder / f'{name}.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        generated = model(noise).numpy()
        scores = {'step': int(checkpoint['step']), 'test_swd': sliced_wasserstein(generated, test)}
        if c['dataset'] in MIXTURES:
            centers, sigma = MIXTURES[c['dataset']]
            scores.update(mode_metrics(generated, (centers - sp.mean) / sp.scale, sigma / sp.scale))
        result[name] = scores
    result['baselines'] = {'real_real': {'swd': sliced_wasserstein(sp.train.x[:count], test)}}
    result['evaluation'] = {'test_points': count, 'noise_seed': NOISE_SEED}
    (out / 'metrics.json').write_text(json.dumps(result, indent=1))
    for f in ('samples.npz', 'history.csv', 'config.json'):
        if not (out / f).exists():
            (out / f).symlink_to((folder / f).resolve())
    if k % 50 == 0:
        print(f'{k + 1}/{len(folders)} {folder.relative_to(SOURCE)}', flush=True)
print(f'done: {len(folders)} folders -> {TARGET}')
