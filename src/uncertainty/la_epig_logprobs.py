"""
Cl = number of classes
K = number of model samples
N_p = number of pool examples
N_t = number of target examples
"""

import math

import torch
from torch import Tensor

from src.uncertainty.bald_logprobs import entropy_from_logprobs, marginal_entropy_from_logprobs
from src.uncertainty.utils import check


def la_epig_from_logprobs(
    logprobs_pool: Tensor, logprobs_targ: Tensor, labels_pool: Tensor
) -> Tensor:
    """
    LA-EPIG(x,y)  = E_{p_*(x_*)}[ H[p(y_*|x_*)] - H[p(y_*|x_*,x,y)] ]
                 ~= 1/M Σ_{m=1}^M H[p(y_*^j|x_*^j)] - H[p(y_*^j|x_*^j,x,y)], x_*^j ~ p_*(x_*)

    log p(y_*|x_*)  = log E_{p(θ)}[p(y_*|x_*,θ)]
                   ~= log 1/K Σ_{i=1}^K exp log p(y_*|x_*,θ_i), θ_i ~ p(θ)

    log p(y_*|x_*,x,y)  = log E_{p(θ|x,y)}[p(y_*|x_*,θ)]
                       ~= log 1/K Σ_{i=1}^K p(y_*|x_*,θ_i), θ_i ~ p(θ|x,y)
                        = log ( 1/K Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / p(y'=y|x) ), θ_i ~ p(θ)
                       ~= log ( 1/K Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / ( 1/K Σ_{j=1}^K p(y'=y|x,θ_j) ) ), θ_i ~ p(θ), θ_j ~ p(θ)
                        = log ( Σ_{i=1}^K p(y_*|x_*,θ_i) p(y'=y|x,θ_i) / Σ_{j=1}^K p(y'=y|x,θ_j) ), θ_i ~ p(θ), θ_j ~ p(θ)
                        = log Σ_{i=1}^K exp log ( p(y_*|x_*,θ_i) p(y'=y|x,θ_i) ) - log Σ_{j=1}^K exp log p(y'=y|x,θ_j), θ_i ~ p(θ), θ_j ~ p(θ)

    Quantities:
    - p(y_*|x_*) is the predictive prior over y_*.
    - p(y_*|x_*,x,y) is the predictive posterior over y_*.
    - p(y'=y|x,θ) is the likelihood of θ for data (x,y).

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]
        labels_pool: Tensor[int], [N_p]

    Returns:
        Tensor[float], [N_p]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    # Repeat the labels for each model sample to enable torch.gather() below.
    labels_pool = labels_pool[:, None, None].expand(-1, logprobs_pool.shape[1], -1)  # [N_p, K, 1]

    # Compute the log likelihood, log p(y'=y|x,θ_i), for each model sample, θ_i. This is the log
    # probability assigned to the observed label, y, which we get by indexing into logprobs.
    logliks_pool = torch.gather(logprobs_pool, dim=-1, index=labels_pool)  # [N_p, K, 1]
    logliks_pool = logliks_pool[:, None, :, :]  # [N_p, 1, K, 1]

    # Compute the log of the likelihood-weighted parameter-conditional predictive distribution,
    # log ( p(y_*|x_*,θ_i) p(y'=y|x,θ_i) ), for each model sample, θ_i.
    weighted_logprobs_targ = logliks_pool + logprobs_targ[None, :, :, :]  # [N_p, N_t, K, Cl]

    # Estimate the log of the predictive posterior over y_*, log p(y_*|x_*,x,y).
    logprobs_targ_postr = torch.logsumexp(weighted_logprobs_targ, dim=2)  # [N_p, N_t, Cl]
    logprobs_targ_postr -= torch.logsumexp(logliks_pool, dim=2)  # [N_p, N_t, Cl]

    # Compute the entropy of the predictive prior, H[p(y_*|x_*)].
    entropy_prior = marginal_entropy_from_logprobs(logprobs_targ)  # [N_t]

    # Compute the entropy of the predictive posterior, H[p(y_*|x_*,x,y)].
    entropy_postr = entropy_from_logprobs(logprobs_targ_postr)  # [N_p, N_t]

    scores = entropy_prior[None, :] - entropy_postr  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p]
    scores = check(
        scores,
        min_value=-math.log(logprobs_pool.shape[-1]),
        max_value=math.log(logprobs_pool.shape[-1]),
        score_type="LA-EPIG",
    )  # [N_p]

    return scores  # [N_p]
