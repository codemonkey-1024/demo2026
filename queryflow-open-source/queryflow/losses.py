from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.nn import functional as F  # noqa:N812


class BaseLoss(ABC):
    """Base abstract class for all loss functions.

    This class serves as a template for implementing custom loss functions. All loss
    functions should inherit from this class and implement the required abstract methods.

    Methods:
        __init__(*args: Any, **kwargs: Any) -> None
            Initialize the loss function with given parameters.

        __call__(pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any) -> Any
            Calculate the loss between predicted and target values.

    Args:
        pred : torch.Tensor
            The predicted values from the model
        target : torch.Tensor
            The ground truth values
        *args : Any
            Variable length argument list
        **kwargs : Any
            Arbitrary keyword arguments

    Returns:
        Any: The computed loss value
    """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Any:
        pass


class WeightedPairwiseLoss(BaseLoss):
    """Rank every positive above known hard negatives, preserving target mass.

    Targets are document/step-balanced positive weights, zero for known
    negatives and negative for unknowns. Positive weights are not renormalized.
    """

    def __init__(
        self, hard_negatives: int = 64, margin: float = 0.5,
        temperature: float = 1.0, *args: Any, **kwargs: Any,
    ) -> None:
        if hard_negatives <= 0 or temperature <= 0 or margin < 0:
            raise ValueError("hard_negatives and temperature must be positive; margin must be nonnegative")
        self.hard_negatives = int(hard_negatives)
        self.margin = margin
        self.temperature = temperature

    def __call__(self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any):
        query_losses = []
        for scores, weights in zip(pred, target):
            positive = weights > 0
            negative_scores = scores[weights == 0]
            if not positive.any() or negative_scores.numel() == 0:
                continue
            negative_scores = negative_scores.topk(
                min(self.hard_negatives, negative_scores.numel())
            ).values
            pair_losses = F.softplus(
                (negative_scores[None, :] - scores[positive, None] + self.margin)
                / self.temperature
            )
            query_losses.append((weights[positive] * pair_losses.mean(dim=-1)).sum())
        if not query_losses:
            return pred.sum() * 0.0
        return torch.stack(query_losses).mean()


class LambdaRankLoss(BaseLoss):
    """Binary-relevance RankNet weighted by detached NDCG@k swap deltas.

    Only positive-negative pairs are materialized. ``positive_mean`` preserves
    swap magnitudes without diluting hard pairs with additional easy negatives.
    """

    def __init__(
        self, top_k: int = 30, sigma: float = 1.0,
        pair_reduction: str = "weight_mean", *args: Any, **kwargs: Any
    ) -> None:
        if top_k <= 0 or sigma <= 0:
            raise ValueError("top_k and sigma must be positive")
        if pair_reduction not in {"weight_mean", "positive_mean"}:
            raise ValueError("pair_reduction must be weight_mean or positive_mean")
        self.top_k = int(top_k)
        self.sigma = sigma
        self.pair_reduction = pair_reduction

    def __call__(
        self, pred: torch.Tensor, target: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        query_losses = []
        for scores, labels in zip(pred, target):
            positive = labels > 0.5
            negative = ~positive
            if not positive.any() or not negative.any():
                continue
            with torch.no_grad():
                order = scores.argsort(descending=True, stable=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device)
                discounts = torch.where(
                    ranks <= self.top_k,
                    1.0 / torch.log2(ranks.to(scores.dtype) + 1.0),
                    0.0,
                )
                ideal_ranks = torch.arange(
                    1, min(int(positive.sum()), self.top_k) + 1,
                    device=scores.device, dtype=scores.dtype,
                )
                ideal_dcg = (1.0 / torch.log2(ideal_ranks + 1.0)).sum()
                weights = (
                    discounts[positive, None] - discounts[negative][None, :]
                ).abs() / ideal_dcg
            pair_loss = F.softplus(
                self.sigma * (scores[negative][None, :] - scores[positive, None])
            )
            denominator = (
                positive.sum() if self.pair_reduction == "positive_mean" else weights.sum()
            )
            query_losses.append((weights * pair_loss).sum() / denominator)
        if not query_losses:
            return pred.sum() * 0.0
        return torch.stack(query_losses).mean()
