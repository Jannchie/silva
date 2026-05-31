"""Histogram specification: remap a batch of scores onto a target band distribution.

The trained head emits an aesthetic *latent* whose distribution is whatever the data
imposed (e.g. bimodal, with a 3<->4 gap). For library-wide display you may instead want
the scores to follow a *prescribed* shape — e.g. 5 bands A..E with a chosen fraction in
each (single-peaked, very few in the top band). This rank-maps the values so exactly
``target_fracs[k]`` of them land in band ``[k/L, (k+1)/L]``, spreading each band linearly
inside. It is strictly rank-preserving (ordering / selection is untouched) — only the
*distribution shape* changes. Needs the whole batch (a global rank), so it is a write-time
calibration for scoring a library, not a per-image model output.
"""

from __future__ import annotations

import numpy as np


def histogram_specify(values: np.ndarray, target_fracs: list[float] | np.ndarray, smooth: bool = False) -> np.ndarray:
    """Rank-map ``values`` to ``[0, 1]`` so each 1/L band holds ``target_fracs`` of them.

    ``target_fracs`` need not be normalised. Returns an array the same shape as ``values``;
    band ``k`` occupies ``[k/L, (k+1)/L]`` and is filled linearly by within-band rank.

    ``smooth=True`` replaces the piecewise-linear band map with a monotone cubic (PCHIP)
    interpolation of the cumulative target, removing the density jumps at band edges. The
    exact per-band fractions blur slightly, but the distribution *shape* is preserved and
    the output has no kinks. Still strictly rank-preserving.
    """
    values = np.asarray(values, dtype=float)
    t = np.asarray(target_fracs, dtype=float)
    t = t / t.sum()
    cum = np.concatenate([[0.0], np.cumsum(t)])
    cum[-1] = 1.0  # guard against float drift so searchsorted covers the top edge
    n, levels = len(values), len(t)

    order = np.argsort(values)
    cumfrac = np.empty(n)
    cumfrac[order] = (np.arange(n) + 0.5) / n  # each value's global rank-fraction in (0,1)

    if smooth:
        from scipy.interpolate import PchipInterpolator

        # invert the target CDF smoothly: score = G^{-1}(rank-fraction). Knots are the band
        # edges (score axis) against their cumulative target mass; PCHIP keeps it monotone.
        band_edges = np.linspace(0.0, 1.0, levels + 1)
        inv = PchipInterpolator(cum, band_edges)
        return np.clip(inv(cumfrac), 0.0, 1.0)

    seg = np.clip(np.searchsorted(cum, cumfrac, side="right") - 1, 0, levels - 1)
    local = (cumfrac - cum[seg]) / (cum[seg + 1] - cum[seg])  # position within the band
    return (seg + local) / levels


def build_calibration_lut(
    latents: np.ndarray, target_fracs: list[float] | np.ndarray, n_knots: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Bake the histogram-specification curve into a ``(lat_knots, score_knots)`` lookup table.

    The batch :func:`histogram_specify` needs the whole library (a global rank). To let a
    *per-image* model reproduce the same calibrated score, snapshot the curve as ``n_knots``
    monotone (latent -> score) points: take ``latents`` quantiles as the latent grid, and the
    smooth histogram-spec map at those (their rank-fractions are exactly the quantile levels)
    as the scores. At inference, ``interp(latent, lat_knots, score_knots)`` recovers the score
    with no global context. Returns float32 arrays, both sorted ascending.
    """
    p = (np.arange(n_knots) + 0.5) / n_knots
    lat_knots = np.quantile(np.asarray(latents, dtype=float), p)
    score_knots = histogram_specify(lat_knots, target_fracs, smooth=True)
    return lat_knots.astype(np.float32), score_knots.astype(np.float32)
