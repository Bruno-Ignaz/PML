"""Drifting field and loss"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_field(generated, real, temperature, attraction_scale=1.0, repulsion_scale=1.0):
    """Compute one drift vector per generated point.

    attraction_scale, repulsion_scale: weights of V+ and V- (paper Table 1).
    Only the default 1, 1 is anti-symmetric.
    """
    count = len(generated)

    references = torch.cat([generated, real], dim=0)
    distances = torch.cdist(generated, references)

    distances[:, :count].fill_diagonal_(1e6)

    kernel = torch.exp(-distances / temperature)

    # Normalize using both row and column sums
    row_sums = kernel.sum(dim=1, keepdim=True)
    column_sums = kernel.sum(dim=0, keepdim=True)
    denominator = (row_sums * column_sums).clamp_min(1e-12).sqrt()
    weights = kernel / denominator

    negative_weights = weights[:, :count]
    positive_weights = weights[:, count:]
    positive_total = positive_weights.sum(dim=1, keepdim=True)
    negative_total = negative_weights.sum(dim=1, keepdim=True)

    # Balance attraction and repulsion using the opposite group's total weight.
    # Written with displacements (y - x): with equal scales the -x terms cancel,
    # with unequal scales (Table 1 ablation) they do not.
    offset = positive_total * negative_total * generated
    attraction = (positive_weights * negative_total) @ real - offset
    repulsion = (negative_weights * positive_total) @ generated - offset

    return attraction_scale * attraction - repulsion_scale * repulsion


def compute_loss(generated, real, temperature, attraction_scale=1.0, repulsion_scale=1.0):
    """Train generated points toward their detached, drifted positions."""
    drift = compute_field(generated, real, temperature, attraction_scale, repulsion_scale)

    # Treat the drifted positions as fixed targets during backpropagation
    target = (generated + drift).detach()

    loss = F.mse_loss(generated, target)

    return loss, drift