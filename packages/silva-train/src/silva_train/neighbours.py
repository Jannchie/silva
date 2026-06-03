"""Visual-neighbour label statistics over precomputed embeddings.

The shared kernel behind the label-audit probes. A label that disagrees sharply with the
labels of its nearest neighbours in SigLIP space is a relabel candidate; the scripts read
that signal at different granularities (per-image, per-artist, single-nearest for blind-spot
diagnosis). Two seams:

  - :func:`nearest_neighbours` — the blocked, self-excluded cosine top-k kernel (the fiddly
    bit: the self-exclusion indexing and the block boundaries). Used by the k-NN mean below
    and by ``blindspot_verify`` (k=1, reading the index + cosine directly).
  - :func:`neighbour_score_mean` — each row's neighbours' mean label (``label_audit`` /
    ``artist_audit``).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def nearest_neighbours(embeddings: torch.Tensor, k: int, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    """Indices and cosine sims of each row's ``k`` nearest neighbours, self excluded.

    ``embeddings`` ``[N, D]`` is L2-normalised internally (cosine = dot). Returns
    ``(idx [N, k], cos [N, k])`` on the embeddings' device, ordered by descending cosine.
    Computed in row blocks of ``batch_size`` so the ``[N, N]`` similarity matrix is never
    materialised whole.
    """
    emb = F.normalize(embeddings.float(), dim=1)
    n = emb.shape[0]
    idx_out = torch.empty((n, k), dtype=torch.long, device=emb.device)
    cos_out = torch.empty((n, k), device=emb.device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims = emb[start:end] @ emb.T
        rows = torch.arange(end - start, device=emb.device)
        sims[rows, torch.arange(start, end, device=emb.device)] = -1e9  # drop self before top-k
        cos, idx = sims.topk(k, dim=1)
        idx_out[start:end] = idx
        cos_out[start:end] = cos
    return idx_out, cos_out


def neighbour_score_mean(embeddings: torch.Tensor, scores: torch.Tensor, k: int, batch_size: int = 2048) -> torch.Tensor:
    """Mean label of each row's ``k`` nearest neighbours in cosine space (self excluded).

    ``scores`` ``[N]`` are the per-row labels. Returns ``[N]`` on the embeddings' device.
    """
    idx, _ = nearest_neighbours(embeddings, k, batch_size)
    return scores.float()[idx].mean(dim=1)
