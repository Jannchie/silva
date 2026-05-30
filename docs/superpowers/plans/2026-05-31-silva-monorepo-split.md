# SILVA monorepo split + HF publishing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the single `silva/` package into a uv-workspace monorepo with a light public inference library (`packages/silva`) and a private training package (`packages/silva-train`), then wire the Hugging Face publishing path on top.

**Architecture:** Two independent packages under `packages/`, uv workspace, `src/` layout. `silva-train` depends on `silva`; `silva` never depends on `silva-train`. The model definition and score-reconstruction live once in `silva`; training imports them. Heavy deps (`transformers`, `accelerate`, `wandb`, `scipy`) stay out of the library core — `transformers`+`pillow` are an optional `[backbone]` extra for image→score and the CLI.

**Tech Stack:** Python 3.12, uv workspaces, hatchling, PyTorch, huggingface-hub (`PyTorchModelHubMixin`), transformers (extra only), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-05-31-silva-monorepo-publish-design.md`

---

## Conventions (read before starting)

- **Branch:** all work on `feat/monorepo-split` (Task 0 creates it).
- **Preserve history:** move files with `git mv`, never delete-and-recreate.
- **Import-shadow shield:** until the legacy root `silva/` directory is removed in Task 7, the repo root is on `sys.path` and would shadow the freshly-installed `silva` package. ALWAYS run tests as `uv run python -P -m pytest <paths>` (the `-P` flag drops CWD from `sys.path`). The root pytest config also sets `--import-mode=importlib`. After Task 7 the legacy dir is gone and plain `uv run pytest` is safe.
- **No `tests/__init__.py`:** with `importlib` import mode the test dirs must not be packages. Do not recreate them.
- **Editable installs:** `uv sync --all-packages` installs both packages editable into `.venv`, so `import silva` / `import silva_train` resolve to `packages/*/src/*`.

---

### Task 0: Branch and preserve the in-flight correctness fixes

The working tree has uncommitted patch14/pooling fixes that MUST survive the moves: modified `scripts/push_to_hub.py`, modified `silva/model_card.py`, and untracked `scripts/verify_embedding.py`.

**Files:**
- Modify (commit as-is): `scripts/push_to_hub.py`, `silva/model_card.py`
- Add (commit): `scripts/verify_embedding.py`

- [ ] **Step 1: Create the feature branch**

Run: `git checkout -b feat/monorepo-split`
Expected: `Switched to a new branch 'feat/monorepo-split'`

- [ ] **Step 2: Commit the in-flight fixes**

```bash
git add scripts/push_to_hub.py silva/model_card.py scripts/verify_embedding.py
git commit -m "fix: pin backbone to patch14 + pooler_output before restructure"
```

- [ ] **Step 3: Confirm a clean tree**

Run: `git status --short`
Expected: no output (clean).

---

### Task 1: Create the uv-workspace skeleton

Stand up the two empty packages and the virtual workspace root, and prove `uv sync` produces an environment where both import. No source moves yet.

**Files:**
- Modify: `pyproject.toml` (repo root → virtual workspace root)
- Create: `packages/silva/pyproject.toml`
- Create: `packages/silva/src/silva/__init__.py`
- Create: `packages/silva/src/silva/models/__init__.py`
- Create: `packages/silva/README.md`
- Create: `packages/silva-train/pyproject.toml`
- Create: `packages/silva-train/src/silva_train/__init__.py`
- Create: `packages/silva-train/src/silva_train/data/__init__.py`

- [ ] **Step 1: Replace the root `pyproject.toml` with a virtual workspace root**

Full new contents of `pyproject.toml`:

```toml
[tool.uv.workspace]
members = ["packages/*"]

[tool.uv.sources]
silva = { workspace = true }
torch = { index = "pytorch-cu132" }

# CUDA build of torch (driver supports CUDA 13.2 -> cu132). Pinning the index here
# keeps `uv sync` from falling back to the CPU wheel. Applies workspace-wide.
[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true

[dependency-groups]
dev = ["pytest>=8", "ruff"]

[tool.ruff]
exclude = [".venv", "wandb", "outputs", "scripts/sweep.py"]  # sweep.py: throwaway experiment harness
line-length = 160

[tool.ruff.lint]
select = ["ALL"]
ignore = [
    "RUF001", "ANN401", "PGH", "RUF003", "BLE001", "ERA001", "FIX002",
    "TD002", "TD003", "D", "A004", "ANN201", "B008", "FAST002", "INP001",
    "PLR2004", "C901", "PLR0912", "PLR0913", "PLR0915", "N812", "COM812",
    "T201", "TRY003", "EM102", "G004", "FBT001", "FBT002", "FBT003",
]
fixable = ["F401"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402", "F401"]
"**/tests/**" = ["S101", "E402", "F401", "ANN", "PLR2004", "PLC0415", "PD008"]
"scripts/*" = ["PLC0415", "EM101"]

[tool.pytest.ini_options]
testpaths = ["packages/silva/tests", "packages/silva-train/tests"]
addopts = "--import-mode=importlib"
```

- [ ] **Step 2: Create `packages/silva/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "silva"
version = "0.1.0"
description = "SILVA: SigLIP-based Illustration Visual Aesthetic Scorer (inference library)"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.4",
    "huggingface-hub>=0.24",
]

[project.optional-dependencies]
# End-to-end image -> score and the `silva` CLI. Core install does NOT need these.
backbone = [
    "transformers>=4.45",
    "pillow>=10",
]

[project.scripts]
silva = "silva.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/silva"]
```

- [ ] **Step 3: Create `packages/silva-train/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "silva-train"
version = "0.1.0"
description = "SILVA training pipeline (private, not published)"
requires-python = ">=3.12"
dependencies = [
    "silva",
    "torch>=2.4",
    "accelerate>=0.34",
    "pandas>=2.2",
    "pyarrow>=17",
    "numpy>=1.26",
    "scipy>=1.13",
    "pydantic>=2",
    "pyyaml>=6.0.2",
]

[project.optional-dependencies]
# Only the pictoria SQLite adapter (scripts/export_manifest.py) needs sqlite-vec.
export = ["sqlite-vec>=0.1"]
# Optional experiment tracking (silva_train.train report_to: wandb).
wandb = ["wandb>=0.17"]

[tool.hatch.build.targets.wheel]
packages = ["src/silva_train"]
```

- [ ] **Step 4: Create the package `__init__.py` files and library README**

`packages/silva/src/silva/__init__.py`:
```python
"""SILVA: SigLIP-based Illustration Visual Aesthetic Scorer (inference library)."""
```

`packages/silva/src/silva/models/__init__.py`: (empty file)

`packages/silva-train/src/silva_train/__init__.py`:
```python
"""SILVA training pipeline (private)."""
```

`packages/silva-train/src/silva_train/data/__init__.py`: (empty file)

`packages/silva/README.md`:
```markdown
# SILVA

A small ordinal-regression head that scores illustrations for one person's aesthetic
taste, on top of frozen `google/siglip2-so400m-patch14-384` embeddings.

```bash
pip install silva                # embedding -> score
pip install "silva[backbone]"    # image -> score + `silva` CLI
```

Weights live on the Hugging Face Hub; load with `silva.hub.HubAestheticModel.from_pretrained`.
```

- [ ] **Step 5: Sync the workspace**

Run: `uv sync --all-packages`
Expected: resolves and installs both `silva` and `silva-train` editable; exit 0.

- [ ] **Step 6: Verify both packages import**

Run: `uv run python -P -c "import silva, silva_train; print('skeleton ok')"`
Expected: `skeleton ok`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml packages/
git commit -m "build: uv-workspace skeleton (silva library + silva-train)"
```

---

### Task 2: Move the model into the library + split out `scoring.py`

Relocate the model modules into `silva`, create `scoring.py` (the score-reconstruction half of the old `losses.py`), and repoint the model imports. Library model tests go green.

**Files:**
- Move: `silva/models/aesthetic.py` → `packages/silva/src/silva/models/aesthetic.py`
- Move: `silva/models/ordinal_head.py` → `packages/silva/src/silva/models/ordinal_head.py`
- Create: `packages/silva/src/silva/scoring.py`
- Modify: `packages/silva/src/silva/models/aesthetic.py` (import line)
- Modify: `packages/silva/src/silva/models/ordinal_head.py` (import line)
- Move: `tests/test_aesthetic_model.py` → `packages/silva/tests/test_aesthetic_model.py`
- Move: `tests/test_ordinal_head.py` → `packages/silva/tests/test_ordinal_head.py`
- Create: `packages/silva/tests/test_scoring.py`

- [ ] **Step 1: Move the model modules**

```bash
git mv silva/models/aesthetic.py packages/silva/src/silva/models/aesthetic.py
git mv silva/models/ordinal_head.py packages/silva/src/silva/models/ordinal_head.py
```

- [ ] **Step 2: Create `packages/silva/src/silva/scoring.py`**

```python
"""Score reconstruction from ordinal threshold logits.

Shared by the model's forward pass (inference) and by the training losses, so
published weights and the training loop agree on exactly what a logit means. No
``torch.nn``, no loss functions — just the logit -> score maps. The loss
functions that consume these live in ``silva_train.losses``.
"""

from __future__ import annotations

import torch

# 5 ordinal levels (scores 1..5) -> 4 binary "score > k" thresholds.
NUM_THRESHOLDS = 4


def unit_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Canonical output in ``[0, 1]``: the mean of the threshold probabilities."""
    return torch.sigmoid(logits).mean(dim=-1)


def ordinal_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Label-space score in ``[1, 5]``: ``1 + sum`` of the threshold probabilities."""
    return 1.0 + torch.sigmoid(logits).sum(dim=-1)
```

- [ ] **Step 3: Repoint the model imports to `silva.scoring`**

In `packages/silva/src/silva/models/aesthetic.py`, change the import line:
```python
from silva.losses import ordinal_score_from_logits, unit_score_from_logits
```
to:
```python
from silva.scoring import ordinal_score_from_logits, unit_score_from_logits
```

In `packages/silva/src/silva/models/ordinal_head.py`, change:
```python
from silva.losses import NUM_THRESHOLDS
```
to:
```python
from silva.scoring import NUM_THRESHOLDS
```

- [ ] **Step 4: Move the model tests**

```bash
git mv tests/test_aesthetic_model.py packages/silva/tests/test_aesthetic_model.py
git mv tests/test_ordinal_head.py packages/silva/tests/test_ordinal_head.py
```
(Their import lines — `from silva.models.aesthetic import ...` and `from silva.models.ordinal_head import ...` — are unchanged: those modules still live under the `silva` package.)

- [ ] **Step 5: Create `packages/silva/tests/test_scoring.py`** (extracted from the old `tests/test_losses.py`)

```python
import pytest
import torch

from silva.scoring import ordinal_score_from_logits, unit_score_from_logits


def test_unit_score_is_zero_based_mean_of_threshold_probs():
    # logits 0 -> sigmoid 0.5 -> mean = 0.5
    assert unit_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(0.5)


def test_unit_score_bounds():
    assert unit_score_from_logits(torch.full((1, 4), 20.0)).item() == pytest.approx(1.0, abs=1e-6)
    assert unit_score_from_logits(torch.full((1, 4), -20.0)).item() == pytest.approx(0.0, abs=1e-6)


def test_ordinal_score_is_unit_rescaled_to_1_5():
    # logits 0 -> sum sigmoid = 2 -> 1 + 2 = 3.0
    assert ordinal_score_from_logits(torch.zeros(1, 4)).item() == pytest.approx(3.0)
```

- [ ] **Step 6: Run the library tests (shielded)**

Run: `uv run python -P -m pytest packages/silva/tests -v`
Expected: PASS — `test_scoring.py` (3), `test_aesthetic_model.py` (5), `test_ordinal_head.py` (3).

- [ ] **Step 7: Commit**

```bash
git add packages/silva silva/models
git commit -m "refactor: move model into silva library + split scoring out of losses"
```

---

### Task 3: Move `hub.py` into the library + round-trip test

`HubAestheticModel` ships in the library (downloaders call `from_pretrained`). Add an offline save/load round-trip test so we don't need the Hub to verify it.

**Files:**
- Move: `silva/hub.py` → `packages/silva/src/silva/hub.py`
- Create: `packages/silva/tests/test_hub.py`

- [ ] **Step 1: Move `hub.py`**

```bash
git mv silva/hub.py packages/silva/src/silva/hub.py
```
(Its imports — `from huggingface_hub import PyTorchModelHubMixin` and `from silva.models.aesthetic import EmbeddingAestheticModel` — are unchanged and resolve within the `silva` package.)

- [ ] **Step 2: Create `packages/silva/tests/test_hub.py`**

```python
import torch

from silva.hub import HubAestheticModel


def test_save_pretrained_writes_safetensors_and_config(tmp_path):
    model = HubAestheticModel(embedding_dim=16, hidden_dims=[32])
    model.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()


def test_from_pretrained_round_trips_weights(tmp_path):
    model = HubAestheticModel(embedding_dim=16, hidden_dims=[32]).eval()
    model.save_pretrained(tmp_path)
    loaded = HubAestheticModel.from_pretrained(tmp_path).eval()

    x = torch.randn(4, 16)
    assert torch.allclose(model(x)["score"], loaded(x)["score"], atol=1e-6)


def test_config_persists_constructor_args(tmp_path):
    HubAestheticModel(embedding_dim=16, dropout=0.2, hidden_dims=[8]).save_pretrained(tmp_path)
    loaded = HubAestheticModel.from_pretrained(tmp_path)
    # the LayerNorm width reflects embedding_dim recovered from config.json
    assert loaded.norm.normalized_shape == (16,)
```

- [ ] **Step 3: Run the library tests (shielded)**

Run: `uv run python -P -m pytest packages/silva/tests -v`
Expected: PASS, including the 3 new `test_hub.py` tests. (Runs fully offline — no Hub access.)

- [ ] **Step 4: Commit**

```bash
git add packages/silva silva/hub.py
git commit -m "refactor: move HubAestheticModel into silva library + offline round-trip test"
```

---

### Task 4: Add `backbone.py` + CLI + `[backbone]` extra

The image→embedding gatekeeper (pinned to patch14 + `pooler_output`) and the `silva score` CLI. Both must import cleanly WITHOUT `transformers` installed; the heavy import happens only when `Embedder` is constructed.

**Files:**
- Create: `packages/silva/src/silva/backbone.py`
- Create: `packages/silva/src/silva/cli.py`
- Create: `packages/silva/tests/test_backbone.py`

- [ ] **Step 1: Write the failing tests**

`packages/silva/tests/test_backbone.py`:
```python
import torch

from silva.backbone import BACKBONE, score_images
from silva.hub import HubAestheticModel


def test_backbone_is_pinned_to_patch14():
    # Must match pictoria ai/siglip_embed.py — patch14, NOT patch16.
    assert BACKBONE == "google/siglip2-so400m-patch14-384"


def test_score_images_runs_head_on_embedder_output():
    head = HubAestheticModel(embedding_dim=16).eval()

    class DummyEmbedder:
        def embed(self, image):
            return torch.randn(1, 16)

    results = score_images(["fake-image"], head, DummyEmbedder())
    assert len(results) == 1
    assert 0.0 <= results[0]["score"] <= 1.0
    assert 1.0 <= results[0]["ordinal_score"] <= 5.0


def test_cli_main_is_importable_without_backbone_extra():
    # Importing the CLI entry point must not require transformers/pillow.
    from silva.cli import main

    assert callable(main)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -P -m pytest packages/silva/tests/test_backbone.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'silva.backbone'`.

- [ ] **Step 3: Write `packages/silva/src/silva/backbone.py`**

```python
"""Image -> 1152-d SigLIP2 embedding, matching how the training vectors were made.

This is the single gatekeeper for train/inference consistency. It pins the
backbone to ``patch14-384`` and takes the raw pooled feature (``pooler_output``) —
exactly the path verified against the stored training embeddings at cosine 0.9998
(see ``scripts/verify_embedding.py``). A different backbone / patch size / pooling
produces vectors the published head was never trained on.

Needs the ``[backbone]`` extra (``transformers`` + ``pillow``). The heavy import is
deferred to ``Embedder.__init__`` so this module — and the CLI entry point — import
cleanly in a core-only install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

BACKBONE = "google/siglip2-so400m-patch14-384"

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL.Image import Image
    from torch import nn


class Embedder:
    """Loads the frozen SigLIP2 backbone once and turns images into embeddings."""

    def __init__(self, device: str | None = None) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            msg = 'silva image scoring needs the backbone extra: pip install "silva[backbone]"'
            raise ImportError(msg) from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(BACKBONE)
        self.model = AutoModel.from_pretrained(BACKBONE).to(self.device).eval()

    @torch.no_grad()
    def embed(self, image: Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(pixel_values=inputs.pixel_values)
        # newer transformers wraps the result; the pooled [1, 1152] vector is .pooler_output
        return (feats.pooler_output if hasattr(feats, "pooler_output") else feats).float()


@torch.no_grad()
def score_images(images: Sequence[Image], head: nn.Module, embedder: Embedder) -> list[dict[str, float]]:
    """Embed each image and run the head, returning ``{"score", "ordinal_score"}`` per image."""
    head.eval()
    results: list[dict[str, float]] = []
    for image in images:
        out = head(embedder.embed(image))
        results.append({"score": float(out["score"].item()), "ordinal_score": float(out["ordinal_score"].item())})
    return results
```

- [ ] **Step 4: Write `packages/silva/src/silva/cli.py`**

```python
"""`silva score IMG [IMG ...]` — end-to-end image aesthetic scoring.

Requires the ``[backbone]`` extra. The transformers/pillow imports live inside
``main`` so the registered entry point imports cleanly in a core-only install and
fails with a clear message only when actually run.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="silva", description="Score images for personal aesthetic appeal.")
    parser.add_argument("command", choices=["score"], help="only 'score' is supported")
    parser.add_argument("images", nargs="+", help="image file paths")
    parser.add_argument("--repo-id", default="<user>/silva-aesthetic", help="Hugging Face repo of the published head")
    parser.add_argument("--device", default=None, help="torch device override (default: auto)")
    args = parser.parse_args()

    from PIL import Image

    from silva.backbone import Embedder, score_images
    from silva.hub import HubAestheticModel

    head = HubAestheticModel.from_pretrained(args.repo_id)
    embedder = Embedder(device=args.device)
    images = [Image.open(path) for path in args.images]
    for path, res in zip(args.images, score_images(images, head, embedder), strict=True):
        print(f"{path}\tscore={res['score']:.4f}\tordinal={res['ordinal_score']:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -P -m pytest packages/silva/tests/test_backbone.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint the new library code**

Run: `uv run ruff check packages/silva/src`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add packages/silva
git commit -m "feat: backbone embedder (patch14/pooler_output) + silva score CLI + [backbone] extra"
```

---

### Task 5: Move the training package + trim `losses.py`

Relocate every training-only module into `silva_train`, drop the score-reconstruction defs from `losses.py` (now imported from the library), and repoint imports. Training tests go green.

**Files:**
- Move: `silva/losses.py` → `packages/silva-train/src/silva_train/losses.py`
- Move: `silva/config.py` → `packages/silva-train/src/silva_train/config.py`
- Move: `silva/metrics.py` → `packages/silva-train/src/silva_train/metrics.py`
- Move: `silva/evaluate.py` → `packages/silva-train/src/silva_train/evaluate.py`
- Move: `silva/train.py` → `packages/silva-train/src/silva_train/train.py`
- Move: `silva/data/dataset.py` → `packages/silva-train/src/silva_train/data/dataset.py`
- Move: `silva/data/manifest.py` → `packages/silva-train/src/silva_train/data/manifest.py`
- Modify: `silva_train/losses.py`, `silva_train/evaluate.py`, `silva_train/train.py`, `silva_train/data/dataset.py` (imports)
- Move + edit: the six training tests into `packages/silva-train/tests/`

- [ ] **Step 1: Move the training modules**

```bash
git mv silva/losses.py packages/silva-train/src/silva_train/losses.py
git mv silva/config.py packages/silva-train/src/silva_train/config.py
git mv silva/metrics.py packages/silva-train/src/silva_train/metrics.py
git mv silva/evaluate.py packages/silva-train/src/silva_train/evaluate.py
git mv silva/train.py packages/silva-train/src/silva_train/train.py
git mv silva/data/dataset.py packages/silva-train/src/silva_train/data/dataset.py
git mv silva/data/manifest.py packages/silva-train/src/silva_train/data/manifest.py
```

- [ ] **Step 2: Trim `silva_train/losses.py`** — remove the three symbols that moved to `silva.scoring` and import them instead.

Delete these from the file: the `NUM_THRESHOLDS = 4` assignment (and its comment), the `unit_score_from_logits` function, and the `ordinal_score_from_logits` function.

Replace the top-of-file imports:
```python
import torch
import torch.nn.functional as F

# 5 ordinal levels (scores 1..5) -> 4 binary "score > k" thresholds.
NUM_THRESHOLDS = 4
```
with:
```python
import torch
import torch.nn.functional as F

from silva.scoring import NUM_THRESHOLDS, ordinal_score_from_logits
```
(Keep `make_ordinal_targets`, `ordinal_loss`, `compute_pos_weight`, `pairwise_ranking_loss`, `soft_spearman_loss`, `listwise_loss`, `silva_loss`. They already call `ordinal_score_from_logits` / use `NUM_THRESHOLDS`, now satisfied by the import.)

- [ ] **Step 3: Repoint imports in the moved training modules**

In `silva_train/data/dataset.py`:
```python
from silva.data.manifest import validate_manifest
```
→
```python
from silva_train.data.manifest import validate_manifest
```

In `silva_train/evaluate.py`, change the three intra-package imports (leave the model import pointing at the library):
```python
from silva.config import Config
from silva.data.dataset import AestheticDataset
from silva.metrics import compute_metrics
from silva.models.aesthetic import EmbeddingAestheticModel
```
→
```python
from silva.models.aesthetic import EmbeddingAestheticModel  # library

from silva_train.config import Config
from silva_train.data.dataset import AestheticDataset
from silva_train.metrics import compute_metrics
```

In `silva_train/train.py`:
```python
from silva.config import Config
from silva.data.dataset import AestheticDataset
from silva.losses import compute_pos_weight, silva_loss
from silva.metrics import compute_metrics, is_improvement
from silva.models.aesthetic import EmbeddingAestheticModel
```
→
```python
from silva.models.aesthetic import EmbeddingAestheticModel  # library

from silva_train.config import Config
from silva_train.data.dataset import AestheticDataset
from silva_train.losses import compute_pos_weight, silva_loss
from silva_train.metrics import compute_metrics, is_improvement
```

(`silva_train/config.py`, `silva_train/metrics.py`, `silva_train/data/manifest.py` have no `silva` imports — no change.)

- [ ] **Step 4: Move and repoint the training tests**

```bash
git mv tests/test_losses.py packages/silva-train/tests/test_losses.py
git mv tests/test_dataset.py packages/silva-train/tests/test_dataset.py
git mv tests/test_manifest.py packages/silva-train/tests/test_manifest.py
git mv tests/test_metrics.py packages/silva-train/tests/test_metrics.py
git mv tests/test_split.py packages/silva-train/tests/test_split.py
git mv tests/test_train_smoke.py packages/silva-train/tests/test_train_smoke.py
```

Edit `packages/silva-train/tests/test_losses.py`: the score-reconstruction tests already moved to the library (`test_scoring.py`), so delete the three functions `test_unit_score_is_zero_based_mean_of_threshold_probs`, `test_unit_score_bounds`, and `test_ordinal_score_is_unit_rescaled_to_1_5`, and change the import block:
```python
from silva.losses import (
    compute_pos_weight,
    listwise_loss,
    make_ordinal_targets,
    ordinal_loss,
    ordinal_score_from_logits,
    pairwise_ranking_loss,
    silva_loss,
    soft_spearman_loss,
    unit_score_from_logits,
)
```
to:
```python
from silva_train.losses import (
    compute_pos_weight,
    listwise_loss,
    make_ordinal_targets,
    ordinal_loss,
    pairwise_ranking_loss,
    silva_loss,
    soft_spearman_loss,
)
```

Change the import line at the top of each remaining moved test:
- `test_dataset.py`: `from silva.data.dataset import AestheticDataset` → `from silva_train.data.dataset import AestheticDataset`
- `test_manifest.py`: `from silva.data.manifest import build_manifest, validate_manifest, write_manifest` → `from silva_train.data.manifest import build_manifest, validate_manifest, write_manifest`
- `test_split.py`: `from silva.data.manifest import assign_splits` → `from silva_train.data.manifest import assign_splits`
- `test_train_smoke.py`: `from silva.data.manifest import write_manifest` → `from silva_train.data.manifest import write_manifest`, and the inner `from silva.train import train` → `from silva_train.train import train`
- `test_metrics.py`: `from silva.metrics import (...)` → `from silva_train.metrics import (...)` (keep the imported names as-is)

- [ ] **Step 5: Re-sync (the `silva.scoring` dependency edge is new) and run training tests (shielded)**

```bash
uv sync --all-packages
uv run python -P -m pytest packages/silva-train/tests -v
```
Expected: PASS — losses (the loss-only subset), dataset, manifest, metrics, split, train smoke.

- [ ] **Step 6: Run the full suite (shielded) to confirm both packages are green**

Run: `uv run python -P -m pytest -v`
Expected: PASS across `packages/silva/tests` and `packages/silva-train/tests`.

- [ ] **Step 7: Commit**

```bash
git add packages silva tests
git commit -m "refactor: move training pipeline into silva_train; losses reuse silva.scoring"
```

---

### Task 6: Move `model_card.py` + repoint the scripts

`model_card.py` is publish-time only (embeds training metrics) → training package. Repoint every script that imported the old `silva.*` paths.

**Files:**
- Move: `silva/model_card.py` → `packages/silva-train/src/silva_train/model_card.py`
- Modify: `scripts/push_to_hub.py`, `scripts/export_manifest.py`, `scripts/eval_baselines.py`, `scripts/sweep.py`

- [ ] **Step 1: Move `model_card.py`**

```bash
git mv silva/model_card.py packages/silva-train/src/silva_train/model_card.py
```
(It has no `silva` imports — pure template — so no import edits inside it.)

- [ ] **Step 2: Repoint `scripts/push_to_hub.py`**

Change the top imports:
```python
from silva.hub import HubAestheticModel
from silva.model_card import render_model_card
```
→
```python
from silva.hub import HubAestheticModel               # library

from silva_train.model_card import render_model_card  # training
```
And the inline import inside `main`:
```python
        from silva.evaluate import evaluate
```
→
```python
        from silva_train.evaluate import evaluate
```

- [ ] **Step 3: Repoint the remaining scripts**

- `scripts/export_manifest.py`: `from silva.data.manifest import build_manifest, write_manifest` → `from silva_train.data.manifest import build_manifest, write_manifest`
- `scripts/eval_baselines.py`: `from silva.metrics import compute_metrics` → `from silva_train.metrics import compute_metrics`
- `scripts/sweep.py`: change `from silva.losses import (` → `from silva_train.losses import (`; change `from silva.metrics import compute_metrics` → `from silva_train.metrics import compute_metrics`; leave `from silva.models.aesthetic import EmbeddingAestheticModel` pointing at the library (no change).

- [ ] **Step 4: Verify every script imports under the workspace env**

Run:
```bash
uv run python -P -c "import ast,sys; [ast.parse(open(f,encoding='utf-8').read()) for f in ['scripts/push_to_hub.py','scripts/export_manifest.py','scripts/eval_baselines.py','scripts/sweep.py','scripts/verify_embedding.py']]; print('scripts parse ok')"
uv run python -P -c "import importlib.util as u; [u.spec_from_file_location('m','scripts/push_to_hub.py')]; from silva.hub import HubAestheticModel; from silva_train.model_card import render_model_card; from silva_train.evaluate import evaluate; print('push_to_hub imports ok')"
```
Expected: `scripts parse ok` then `push_to_hub imports ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts packages silva
git commit -m "refactor: model_card -> silva_train; repoint scripts to split packages"
```

---

### Task 7: Remove the legacy `silva/` dir + finalize repo config

The legacy root `silva/` is now empty of source (only stale `__init__.py` shells and empty dirs remain). Remove it so the import-shadow shield is no longer needed, then confirm the suite is green WITHOUT `-P`.

**Files:**
- Delete: `silva/` (whatever remains: `silva/__init__.py`, `silva/models/__init__.py`, `silva/data/__init__.py`, empty dirs)
- Delete: `tests/__init__.py` (if still present)
- Modify: `packages/silva/src/silva/__init__.py` (add public API exports)

- [ ] **Step 1: Confirm nothing of value remains in the legacy tree**

Run: `git ls-files silva tests`
Expected: only `silva/__init__.py`, `silva/models/__init__.py`, `silva/data/__init__.py`, and possibly `tests/__init__.py` — all shells. If any real module is still listed, STOP and move it (a Task 2/5/6 step was missed).

- [ ] **Step 2: Remove the legacy shells**

```bash
git rm -r silva
git rm tests/__init__.py
```
(If `git rm tests/__init__.py` reports the path is gone, the `tests/` dir is already empty — fine.)

- [ ] **Step 3: Add public API exports to the library `__init__.py`**

Full new contents of `packages/silva/src/silva/__init__.py`:
```python
"""SILVA: SigLIP-based Illustration Visual Aesthetic Scorer (inference library)."""

from silva.hub import HubAestheticModel
from silva.models.aesthetic import EmbeddingAestheticModel
from silva.scoring import ordinal_score_from_logits, unit_score_from_logits

__all__ = [
    "EmbeddingAestheticModel",
    "HubAestheticModel",
    "ordinal_score_from_logits",
    "unit_score_from_logits",
]
```

- [ ] **Step 4: Run the full suite the normal way (no shield)**

Run: `uv run pytest -v`
Expected: PASS. The legacy `silva/` dir is gone, so there is nothing left to shadow the installed package.

- [ ] **Step 5: Lint everything**

Run: `uv run ruff check .`
Expected: `All checks passed!` (Fix any findings the move introduced, e.g. import ordering, then re-run.)

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: drop legacy silva/ tree; export library public API"
```

---

### Task 8: Verify dependency isolation + publishing dry-run

Prove the core library does not drag in `transformers`, and that the publishing path produces correct artifacts without touching the Hub.

**Files:** none (verification only).

- [ ] **Step 1: Build the library wheel and inspect its metadata**

```bash
uv build --package silva
```
Expected: writes `dist/silva-0.1.0-py3-none-any.whl` (and sdist).

- [ ] **Step 2: Assert `transformers` is gated behind the `backbone` extra only**

Run:
```bash
uv run python -P - <<'PY'
import zipfile, glob, email.parser
whl = sorted(glob.glob("dist/silva-*.whl"))[-1]
with zipfile.ZipFile(whl) as z:
    meta = next(n for n in z.namelist() if n.endswith("METADATA"))
    msg = email.parser.Parser().parsestr(z.read(meta).decode())
reqs = msg.get_all("Requires-Dist") or []
core = [r for r in reqs if "extra ==" not in r]
backbone = [r for r in reqs if 'extra == "backbone"' in r]
assert not any("transformers" in r for r in core), f"transformers leaked into core: {core}"
assert any("transformers" in r for r in backbone), f"transformers missing from backbone extra: {backbone}"
print("core deps:", core)
print("backbone extra:", backbone)
print("isolation OK")
PY
```
Expected: prints core deps (`torch`, `huggingface-hub`), the backbone extra (`transformers`, `pillow`), then `isolation OK`.

- [ ] **Step 3: Confirm a core-only import never imports transformers**

Run:
```bash
uv run python -P -c "import sys; import silva.hub, silva.scoring, silva.cli, silva.backbone; assert 'transformers' not in sys.modules, 'transformers imported at module load'; print('core import clean')"
```
Expected: `core import clean` (importing the modules must not pull `transformers`; it loads only when `Embedder()` is constructed).

- [ ] **Step 4: Publishing dry-run (only if a checkpoint exists)**

If `outputs/v1_stage1_head/best.pt` exists locally:
```bash
uv run --extra hub python scripts/push_to_hub.py --repo-id local/dry-run --dry-run
```
Expected: writes `hub_export/` containing `model.safetensors`, `config.json`, `README.md`, and prints the test-split metrics JSON. No network upload.
If no checkpoint exists, skip — the import wiring is already covered by Task 6 Step 4 and Task 3's round-trip test.

- [ ] **Step 5: Clean build artifacts and commit any config fixes**

```bash
git status --short
# dist/ and hub_export/ should be gitignored; if not, add them to .gitignore
git add -A && git commit -m "chore: verify dependency isolation (no commit if nothing changed)" || echo "nothing to commit"
```

---

## Self-Review

**Spec coverage:**
- §1 monorepo layout (uv workspace, src/) → Tasks 1–7. ✓
- §2 losses.py split (scoring → library, losses → training) → Task 2 (scoring.py) + Task 5 (trim). ✓
- §3 library layering + `[backbone]` extra + CLI + ImportError guard → Task 1 (pyproject extra) + Task 4. ✓
- §4 embedding consistency (patch14 + pooler_output, gatekeeper) → Task 0 (preserve fixes) + Task 4 (`backbone.py` pins both, test asserts patch14). ✓
- §5 publishing flow + file placement (hub→library, model_card→training, scripts at root) → Tasks 3, 6, 8. ✓
- §4 "carry the uncommitted fixes through the move" → Task 0. ✓
- Dependency direction (silva-train → silva, never reverse) → enforced by pyproject (Task 1) and verified (Task 8). ✓

**Placeholder scan:** No TBD/TODO. Every code step shows full content; every import edit shows before→after; every command shows expected output. The CLI's `--repo-id` default `<user>/silva-aesthetic` is an intentional user-supplied placeholder at runtime, not a plan gap.

**Type/name consistency:** `BACKBONE`, `Embedder`, `score_images(images, head, embedder)`, `HubAestheticModel(embedding_dim, dropout, hidden_dims)`, `unit_score_from_logits` / `ordinal_score_from_logits` / `NUM_THRESHOLDS` (in `silva.scoring`), `render_model_card(repo_id, backbone, model_cfg, metrics)`, `evaluate(checkpoint, manifest_path, split, embedding_dim, dropout, hidden_dims, ...)` are used consistently across tasks. Library-vs-training import boundaries are spelled out per file (model import stays `silva.models.aesthetic` even inside training code).
