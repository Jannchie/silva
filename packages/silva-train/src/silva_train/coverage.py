"""Embedding-coverage distance: how far is a query from the labelled training manifold?

One kernel, two consumers:

  - **Sparse-region expansion** (``scripts/coverage_probe.py``): rank probe images by
    distance to the training set — the sparsest regions are where new labels buy the most
    new-distribution robustness.
  - **Out-of-domain defence** (:class:`DomainReference`): photos / 3D / memes are "not
    applicable", not "low score". A query whose kNN distance exceeds a threshold
    calibrated on the training set itself gets flagged instead of trusted.

Distance is mean cosine distance (``1 - cos``) to the ``k`` nearest reference rows —
the query-vs-reference sibling of :func:`silva_train.neighbours.nearest_neighbours`
(which is self-excluded within one set). The threshold is calibrated leave-one-out, which
mirrors how real queries relate to the reference (they are never members of it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file, save_file
from torch.nn import functional as F

from silva_train.neighbours import nearest_neighbours

if TYPE_CHECKING:
    from pathlib import Path


def knn_distance_to_reference(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    batch_size: int = 2048,
) -> torch.Tensor:
    """Mean cosine distance of each ``query`` row to its ``k`` nearest ``reference`` rows.

    ``query`` ``[Q, D]`` and ``reference`` ``[R, D]`` are L2-normalised internally.
    Returns ``[Q]`` on the query's device. Computed in query blocks of ``batch_size`` so
    the ``[Q, R]`` similarity matrix is never materialised whole.
    """
    q = F.normalize(query.float(), dim=1)
    ref = F.normalize(reference.float(), dim=1)
    out = torch.empty(q.shape[0], device=q.device)
    for start in range(0, q.shape[0], batch_size):
        end = min(start + batch_size, q.shape[0])
        cos, _ = (q[start:end] @ ref.T).topk(k, dim=1)
        out[start:end] = 1.0 - cos.mean(dim=1)
    return out


def calibrate_threshold(
    reference: torch.Tensor,
    k: int,
    quantile: float = 0.99,
    batch_size: int = 2048,
) -> float:
    """In-domain distance ceiling: the ``quantile`` of leave-one-out kNN distance.

    Each reference row's distance to its own ``k`` nearest peers (self excluded) is what
    an in-domain query would experience; the quantile of that distribution is the
    threshold above which a query is sparser than (almost) anything in-domain.
    """
    _, cos = nearest_neighbours(reference, k, batch_size)
    dist = 1.0 - cos.mean(dim=1)
    return float(torch.quantile(dist, quantile))


@dataclass(frozen=True)
class DomainReference:
    """A shippable applicability gate: reference embeddings + calibrated distance ceiling."""

    embeddings: torch.Tensor
    k: int
    threshold: float

    @classmethod
    def fit(
        cls,
        embeddings: torch.Tensor,
        k: int = 8,
        quantile: float = 0.99,
        max_rows: int = 4096,
        seed: int = 42,
    ) -> DomainReference:
        """Subsample to ``max_rows`` (so the artifact stays small), then calibrate.

        Calibration happens AFTER subsampling: the threshold must describe the reference
        actually shipped, not the denser set it was drawn from.
        """
        if embeddings.shape[0] > max_rows:
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(embeddings.shape[0], generator=g)[:max_rows]
            embeddings = embeddings[idx]
        embeddings = F.normalize(embeddings.float(), dim=1)
        return cls(embeddings=embeddings, k=k, threshold=calibrate_threshold(embeddings, k, quantile))

    def distance(self, query: torch.Tensor) -> torch.Tensor:
        return knn_distance_to_reference(query, self.embeddings.to(query.device), self.k)

    def in_domain(self, query: torch.Tensor) -> torch.Tensor:
        """Boolean ``[Q]``: True where the query is within the calibrated distance ceiling."""
        return self.distance(query) <= self.threshold

    def save(self, path: str | Path) -> None:
        save_file(
            {
                "embeddings": self.embeddings.contiguous().cpu(),
                "k": torch.tensor([self.k]),
                "threshold": torch.tensor([self.threshold], dtype=torch.float64),
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> DomainReference:
        data = load_file(str(path))
        return cls(embeddings=data["embeddings"], k=int(data["k"][0]), threshold=float(data["threshold"][0]))
