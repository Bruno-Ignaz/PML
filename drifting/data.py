"""Toy datasets and data loading """
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import torch
import math

NAMES = (
    'bimodal', 'ring', 'moons', 'spiral', 'checkerboard',
    'heart', 'star', 'hearts', 'grid',
)

"""Creating toy datasets
Returns x and y according to dataset name, e.g. 'checkerboard' """
def make_dataset(name: str, n=12000, seed=123):
    if name not in NAMES or n < 20:
        raise ValueError(f'Choose {NAMES}; n must be >= 20')
    r = np.random.default_rng(seed)

    if name in ('bimodal', 'ring'):
        # choose center, then add Gaussian noise around it
        centers = (np.array([[-2., 0.], [2., 0.]]) if name == 'bimodal' else
                   2*np.c_[np.cos(np.arange(8)*np.pi/4), np.sin(np.arange(8)*np.pi/4)])
        y = r.integers(len(centers), size=n)
        x = centers[y] + r.normal(0, .25 if name == 'bimodal' else .12, (n, 2))

    elif name == 'moons':    
        # sample two semicircles, flip and shift one, then add noise  
        y = r.integers(2, size=n); t = r.uniform(0, np.pi, n)
        x = np.c_[np.cos(t), np.sin(t)]
        x[y == 1] = np.c_[1-np.cos(t[y == 1]), .5-np.sin(t[y == 1])]
        x += r.normal(0, .04, x.shape)

    elif name == 'spiral':
        # radius grows with angle, then scale and add noise.
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        u = torch.rand(n, generator=g)
        t = 0.5 * math.pi + 4.0 * math.pi * u
        pts = torch.stack([t * torch.cos(t), t * torch.sin(t)], dim=1)
        pts = pts / (pts.abs().max() + 1e-8)
        pts = pts + 0.03 * torch.randn(pts.shape, generator=g)

        x = pts.numpy()
        y = np.zeros(n, dtype=int)

    elif name == 'star':
        # alternate outer/inner corners and sample along equal-length edges
        angles = np.pi/2 + np.arange(10)*np.pi/5
        corners = np.tile([1., .45], 5)[:, None] * np.c_[
            np.cos(angles), np.sin(angles)]
        edge = r.integers(10, size=n)
        x = corners[edge] + r.random((n, 1)) * (
            corners[(edge + 1) % 10] - corners[edge])
        x += r.normal(0, .025, x.shape)
        y = np.zeros(n, dtype=int)
    
    elif name == 'heart':
        # construct the outline, then sample uniformly by distance along it
        angles = np.linspace(0, 2*np.pi, 4097)
        outline = np.c_[
            np.sin(angles)**3,
            (13*np.cos(angles) - 5*np.cos(2*angles)
                - 2*np.cos(3*angles) - np.cos(4*angles)) / 16
        ]
        distance = np.r_[0, np.cumsum(np.linalg.norm(
            np.diff(outline, axis=0), axis=1))]
        positions = r.uniform(0, distance[-1], n)
        x = np.c_[np.interp(positions, distance, outline[:, 0]),
                  np.interp(positions, distance, outline[:, 1])]
        x += r.normal(0, .025, x.shape)
        y = np.zeros(n, dtype=int)

    elif name == 'hearts':
        # generate heart points and randomly shift them to three locations
        x, _ = make_dataset('heart', n, seed=int(r.integers(2**32)))
        y = r.integers(3, size=n)
        x += np.array([[-2.2, -1.3], [2.2, -1.3], [0., 2.]])[y]

    elif name == 'grid':
        # 5x5 narrow Gaussians on a square lattice with spacing 1
        coords = np.arange(5) - 2.
        centers = np.array([[a, b] for a in coords for b in coords])
        y = r.integers(len(centers), size=n)
        x = centers[y] + r.normal(0, .05, (n, 2))

    #Checkerboard
    else:
        # alternating grid squares and add noise
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        b = torch.randint(0, 2, (n,), generator=g)
        i = torch.randint(0, 2, (n,), generator=g) * 2 + b
        j = torch.randint(0, 2, (n,), generator=g) * 2 + b
        u = torch.rand(n, generator=g)
        v = torch.rand(n, generator=g)
        pts = torch.stack([i + u, j + v], dim=1) - 2.0
        pts = pts / 2.0
        pts = pts + 0.05 * torch.randn(pts.shape, generator=g)

        x = pts.numpy()
        y = (i * 4 + j).numpy()  # Label identifies the occupied square.
    return x.astype(np.float32), y.astype(np.int64)

@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    def stream(self, transform: Callable | None = None, *, shuffle=False, seed=0):
        """One finite epoch, transformations run only at next()"""
        order = np.random.default_rng(seed).permutation(len(self.x)) if shuffle else range(len(self.x))
        for i in order:
            x = self.x[i].copy()
            yield (transform(x) if transform else x), int(self.y[i])

@dataclass
class DataSplits:
    train: Split
    val: Split
    test: Split
    mean: np.ndarray
    scale: float

def split_dataset(x, y, ratio=(.7, .15, .15), seed=321, normalize=True):
    """Split points into train/val/test sets and normalize using training data."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y)

    # Check that every point has two coordinates and one label
    if (x.ndim != 2 or x.shape[1] != 2
            or y.ndim != 1 or len(x) != len(y)
            or not np.isfinite(x).all()):
        raise ValueError('Expected finite (N,2) points and N labels')

    # Split ratio must be positive and add up to 100%
    if (len(ratio) != 3 or not np.isfinite(ratio).all()
            or min(ratio) <= 0 or not np.isclose(sum(ratio), 1)):
        raise ValueError('Three positive split ratio must sum to one')

    # Shuffle indices with seed
    indices = np.random.default_rng(seed).permutation(len(x))

    # Split using indices
    train_end = int(len(x) * ratio[0])
    val_end = int(len(x) * (ratio[0] + ratio[1]))
    train_ids = indices[:train_end]
    val_ids = indices[train_end:val_end]
    test_ids = indices[val_end:]

    # Check if a split has less than two data samples
    if min(len(train_ids), len(val_ids), len(test_ids)) < 2:
        raise ValueError('Each split needs at least two observations')

    # Optionally normalize using training statistics only
    mean = np.zeros(2, dtype=np.float32)
    scale = 1.0

    if normalize:
        train_points = x[train_ids]
        mean = train_points.mean(axis=0)
        scale = max(float(np.sqrt(np.mean((train_points - mean)**2))), 1e-8)

    normalized_x = (x - mean) / scale

    train = Split(normalized_x[train_ids], y[train_ids], train_ids)
    val = Split(normalized_x[val_ids], y[val_ids], val_ids)
    test = Split(normalized_x[test_ids], y[test_ids], test_ids)

    return DataSplits(train, val, test, mean, scale)

def make_loaders(x, y, transform=None, ratio=(.7,.15,.15), seed=321, normalize=True):
    """Return hree Python generators yielding individual (xi, yi) and optionally apply transformation"""
    s = split_dataset(x, y, ratio, seed, normalize=normalize)
    return tuple(part.stream(transform) for part in (s.train, s.val, s.test))
