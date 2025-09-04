"""
Cl = number of classes
K = number of model samples
N = number of examples
"""

import torch
from torch import Tensor

from src.math import logmeanexp
from src.uncertainty.utils import check


def mic_from_logprobs(logprobs: Tensor, labels: Tensor, eta: float) -> Tensor:
    """
    MIC(x,y,η) = -log p(y'=y|x) + η log p(y'=y|x,y)
               = -log p(y'=y|x) + η log( p(y'=y,y''=y|x) / p(y'=y|x) )
               = -log p(y'=y|x) + η log p(y'=y,y''=y|x) - η log p(y'=y|x)
               = -(1+η) log p(y'=y|x) + η log p(y'=y,y''=y|x)

    log p(y'=y|x)  = log E_{p(θ)}[p(y'=y|x,θ)]
                  ~= log 1/K Σ_{i=1}^K exp log p(y'=y|x,θ_i), θ_i ~ p(θ)

    log p(y'=y,y''=y|x)  = log E_{p(θ)}[p(y'=y,y''=y|x,θ)]
                         = log E_{p(θ)}[p(y'=y|x,θ)p(y''=y|x,θ)]
                         = log E_{p(θ)}[p(y'=y|x,θ)^2]
                        ~= log 1/K Σ_{i=1}^K exp log p(y'=y|x,θ_i)^2, θ_i ~ p(θ)

    Quantities:
    - p(y'|x) is a predictive distribution over y' (a normalized vector of length Cl).
    - p(y',y''|x) is a predictive distribution over y' and y'' (a normalized grid of size Cl * Cl).
    - p(y'=y|x,θ) is the likelihood of θ for data (x,y).
    - p(y'=y,y''=y|x,θ) is the likelihood of θ for data ((x,y),(x,y')).
    - p(y'=y|x) is the marginal likelihood for data (x,y).
    - p(y'=y,y''=y|x) is the marginal likelihood for data ((x,y),(x,y')).

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N]
        eta: float

    Returns:
        Tensor[float], [N]
    """
    assert logprobs.ndim == 3

    # Repeat the labels for each model sample to enable torch.gather() below.
    labels = labels[:, None, None].expand(-1, logprobs.shape[1], -1)  # [N, K, 1]

    # Compute the log likelihood, log p(y'=y|x,θ_i), for each model sample, θ_i. This is the log
    # probability assigned to the observed label, y, which we get by indexing into logprobs.
    logliks = torch.gather(logprobs, dim=-1, index=labels)  # [N, K, 1]

    # Estimate the single-example log marginal likelihood, log p(y'=y|x).
    single_marg_logliks = logmeanexp(logliks, dim=(1, 2))  # [N]

    # Estimate the two-example log marginal likelihood, log p(y'=y,y''=y|x).
    joint_marg_logliks = logmeanexp(2 * logliks, dim=(1, 2))  # [N]

    scores = -(1 + eta) * single_marg_logliks + eta * joint_marg_logliks  # [N]
    scores = check(scores, score_type="MIC")  # [N]

    return scores  # [N]
