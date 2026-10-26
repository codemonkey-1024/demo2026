from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


class BaseDocRanker(ABC):
    """
    Abstract class for document ranker

    Args:
        ent2doc (torch.Tensor): Mapping from entity to document
    """

    def __init__(self, ent2doc: torch.Tensor) -> None:
        self.ent2doc = ent2doc

    @abstractmethod
    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        pass


class SoftplusIDFWeightedRanker(BaseDocRanker):
    """Aggregate every entity after positive transformation and IDF weighting."""

    def __init__(self, ent2doc: torch.Tensor) -> None:
        super().__init__(ent2doc)
        frequency = torch.sparse.sum(ent2doc, dim=-1).to_dense()
        self.idf_weight = torch.where(
            frequency > 0,
            1.0 / frequency,
            torch.zeros_like(frequency),
        )

    def __call__(self, ent_pred: torch.Tensor) -> torch.Tensor:
        transformed = F.softplus(ent_pred)
        weighted = transformed * self.idf_weight.to(ent_pred.device).unsqueeze(0)
        return torch.sparse.mm(weighted, self.ent2doc)


