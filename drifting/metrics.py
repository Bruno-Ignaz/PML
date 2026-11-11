"""Distributional metrics in train-standardized coordinates; lower is better."""
import numpy as np
from scipy.spatial.distance import cdist

def sliced_wasserstein(x, y, projections=128, seed=2026):
    """Mean one-dimensional W1 over fixed random unit projections."""
    if len(x) != len(y):
        raise ValueError('Use equal sample counts for empirical quantile comparison')
    d = np.random.default_rng(seed).normal(size=(x.shape[1],projections))
    d /= np.linalg.norm(d,axis=0,keepdims=True)
    return float(np.abs(np.sort(x@d,axis=0)-np.sort(y@d,axis=0)).mean())

def mmd2(x, y, bandwidths=(.1,.3,1.)):
    """Unbiased multi-RBF MMD squared. Finite-sample estimates may be negative."""
    xx, yy, xy = (cdist(a,b,'sqeuclidean') for a,b in ((x,x),(y,y),(x,y)))
    kernels = [sum(np.exp(-d/(2*s*s)) for s in bandwidths)/len(bandwidths) for d in (xx,yy,xy)]
    a,b,c = kernels; n,m = len(x),len(y)
    return float((a.sum()-np.trace(a))/(n*(n-1))+(b.sum()-np.trace(b))/(m*(m-1))-2*c.mean())

def evaluate(x,y):
    return {'swd':sliced_wasserstein(x,y),'mmd2':mmd2(x,y)}

def mode_metrics(x, centers, sigma):
    d = cdist(x,centers); assignment = d.argmin(1)
    valid = d.min(1) <= 3*sigma
    counts = np.bincount(assignment[valid],minlength=len(centers))
    # Include an outlier bin so off-support mass is not silently discarded.
    freq = np.r_[counts, (~valid).sum()]/len(x)
    target = np.r_[np.ones(len(centers))/len(centers),0.]
    return {'mode_coverage':float((counts >= .01*len(x)).mean()),
            'mode_tv_with_outliers':float(.5*np.abs(freq-target).sum()),
            'inlier_fraction':float(valid.mean())}
