"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import math

import torch
from torch import Tensor

from src.uncertainty.bald_probs import entropy_from_probs, marginal_entropy_from_probs
from src.uncertainty.utils import check


def la_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor, labels_pool: Tensor) -> Tensor:
    """
    LA-EPIG(x,y)  = E_{p_*(x_*)}[ H[p(y_*|x_*)] - H[p(y_*|x_*,x,y)] ]
                 ~= 1/M Σ_{m=1}^M H[p(y_*^j|x_*^j)] - H[p(y_*^j|x_*^j,x,y)], x_*^j ~ p_*(x_*)

    p(y_*|x_*)  = E_{p(θ)}[p(y_*|x_*,θ)]
               ~= 1/K Σ_{i=1}^K p(y_*|x_*,θ_i), θ_i ~ p(θ)

    p(y_*|x_*,x,y)  = E_{p(θ|x,y)}[p(y_*|x_*,θ)]
                   ~= 1/K Σ_{i=1}^K p(y_*|x_*,θ_i), θ_i ~ p(θ|x,y)
                    = 1/K Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / p(y'=y|x), θ_i ~ p(θ)
                   ~= 1/K Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / ( 1/K Σ_{j=1}^K p(y'=y|x,θ_j) ), θ_i ~ p(θ), θ_j ~ p(θ)
                    = Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / Σ_{j=1}^K p(y'=y|x,θ_j), θ_i ~ p(θ), θ_j ~ p(θ)

    Quantities:
    - p(y_*|x_*) is the predictive prior over y_*.
    - p(y_*|x_*,x,y) is the predictive posterior over y_*.
    - p(y'=y|x,θ) is the likelihood of θ for data (x,y).

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        labels_pool: Tensor[int], [N_p]

    Returns:
        Tensor[float], [N_p]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    # Repeat the labels for each model sample to enable torch.gather() below.
    labels_pool = labels_pool[:, None, None].expand(-1, probs_pool.shape[1], -1)  # [N_p, K, 1]

    # Compute the likelihood, p(y'=y|x,θ_i), for each model sample, θ_i. This is the probability
    # assigned to the observed label, y, which we get by indexing into probs.
    liks_pool = torch.gather(probs_pool, dim=-1, index=labels_pool)  # [N_p, K, 1]
    liks_pool = liks_pool[:, None, :, :]  # [N_p, 1, K, 1]

    # Compute the likelihood-weighted parameter-conditional predictive distribution,
    # p(y_*|x_*,θ_i) p(y'=y|x,θ_i), for each model sample, θ_i.
    weighted_probs_targ = liks_pool * probs_targ[None, :, :, :]  # [N_p, N_t, K, Cl]

    # Estimate the predictive posterior over y_*, p(y_*|x_*,x,y).
    probs_targ_postr = torch.sum(weighted_probs_targ, dim=2)  # [N_p, N_t, Cl]
    probs_targ_postr /= torch.sum(liks_pool, dim=2)  # [N_p, N_t, Cl]

    # Compute the entropy of the predictive prior, H[p(y_*|x_*)].
    entropy_prior = marginal_entropy_from_probs(probs_targ)  # [N_t]

    # Compute the entropy of the predictive posterior, H[p(y_*|x_*,x,y)].
    entropy_postr = entropy_from_probs(probs_targ_postr)  # [N_p, N_t]

    scores = entropy_prior[None, :] - entropy_postr  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p]
    scores = check(
        scores,
        min_value=-math.log(probs_pool.shape[-1]),
        max_value=math.log(probs_pool.shape[-1]),
        score_type="LA-EPIG",
    )  # [N_p]

    return scores  # [N_p]
