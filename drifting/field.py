"""Drifting field and loss"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_field(generated, real, temperature):
    """Compute one drift vector per generated point."""
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

    # Balance attraction and repulsion using the opposite group's total weight.
    attraction_weights = positive_weights * negative_weights.sum(
        dim=1, keepdim=True
    )
    repulsion_weights = negative_weights * positive_weights.sum(
        dim=1, keepdim=True
    )

    attraction = attraction_weights @ real
    repulsion = repulsion_weights @ generated

    return attraction - repulsion


def compute_loss(generated, real, temperature):
    """Train generated points toward their detached, drifted positions."""
    drift = compute_field(generated, real, temperature)

    # Treat the drifted positions as fixed targets during backpropagation
    target = (generated + drift).detach()

    loss = F.mse_loss(generated, target)

    return loss, drift