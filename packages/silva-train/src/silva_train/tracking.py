"""Map ``cfg.train.report_to`` onto accelerate's ``log_with`` argument.

The training loop logs everything through ``accelerator.log(...)``; this just decides
which tracker backs it. Accelerate resolves its *built-in* trackers (tensorboard, ...)
from a string, but a custom tracker must be passed as an instance — pandm ships one
(``pandm.integrations.accelerate.PandmTracker``), so for ``report_to: "pandm"`` we hand
that instance to ``Accelerator(log_with=[...])``. ``pandm`` is imported lazily so the
training package never requires the optional dependency unless tracking is enabled.
"""

from __future__ import annotations

from typing import Any


def build_log_with(report_to: str, project: str, run_name: str | None = None, **pandm_kwargs: Any) -> Any:
    """Translate ``cfg.train.report_to`` into accelerate's ``log_with`` argument.

    - ``"none"``   -> ``None`` (tracking disabled)
    - ``"pandm"``  -> ``[PandmTracker(...)]`` (pandm's own accelerate tracker, local-first)
    - otherwise (``"tensorboard"``, ...) -> the string, handled by accelerate built-ins
    """
    if report_to == "none":
        return None
    if report_to == "pandm":
        from pandm.integrations.accelerate import PandmTracker  # noqa: PLC0415 — optional dependency, imported only when enabled

        return [PandmTracker(project=project, name=run_name, **pandm_kwargs)]
    return report_to
