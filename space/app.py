"""SILVA aesthetic scorer — Gradio demo for Hugging Face Spaces.

Upload an illustration, get its aesthetic score in ``[0, 1]``.

The published head (~7 MB) loads at import; the heavy SigLIP2 backbone is loaded
lazily on the first ``score`` call, so the first request after a cold start is slow
(tens of seconds on a free CPU Space) and every request after it is fast.

The head is calibrated on a single rater's 1-5 labels — one consistent taste,
not a universal quality measure. An optional domain gate (``domain_reference.safetensors``
published next to the head, see ``scripts/fit_domain.py``) flags inputs far from anything
the model was trained on — photos, 3D, memes — where the score is not meaningful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from silva import SilvaScorer

if TYPE_CHECKING:
    from PIL.Image import Image

REPO_ID = "Jannchie/silva-aesthetic"

# Load the head once at startup. The SigLIP2 backbone is pulled in lazily on the
# first scored image (see silva.scorer.SilvaScorer.embedder).
scorer = SilvaScorer.from_pretrained(REPO_ID)


def _load_gate() -> dict | None:
    """Domain gate: subsampled training embeddings + calibrated kNN-distance ceiling.

    Mirrors silva_train.coverage.DomainReference inference (the Space installs only the
    silva-scorer SDK). Missing artifact -> gate off, the demo still scores everything.
    """
    try:
        data = load_file(hf_hub_download(REPO_ID, "domain_reference.safetensors"))
    except Exception:
        return None
    emb = torch.nn.functional.normalize(data["embeddings"].float(), dim=1)
    return {"emb": emb, "k": int(data["k"][0]), "threshold": float(data["threshold"][0])}


gate = _load_gate()


def score_image(image: Image | None) -> str:
    """Score one PIL image and render the result card. Returns HTML."""
    if image is None:
        return _result_html(None)
    with torch.no_grad():
        emb = scorer.embedder.embed(image)  # [1, 1152]; embedder pins the head to its device
        score = float(scorer.head(emb)["calibrated_score"].item())
        out_of_domain = False
        if gate is not None:
            q = torch.nn.functional.normalize(emb.float().cpu(), dim=1)
            cos, _ = (q @ gate["emb"].T).topk(gate["k"], dim=1)
            out_of_domain = float(1.0 - cos.mean()) > gate["threshold"]
    return _result_html(score, out_of_domain=out_of_domain)


def _result_html(score: float | None, out_of_domain: bool = False) -> str:
    if score is None:
        return "<div class='silva-card silva-empty'>No image scored yet.</div>"
    warning = ""
    if out_of_domain:
        warning = (
            "<div class='silva-warning'>Out of scope — this image sits far from the"
            " illustrations the model was trained on, so the score is unreliable.</div>"
        )
    pct = round(score * 100)
    # Friendly 1–5 estimate. calibrated_score is aligned to the 1–5 label
    # distribution, so a linear map back is a reasonable display approximation.
    stars = 1 + 4 * score
    return f"""
    <div class="silva-card">
      <div class="silva-label">Score</div>
      <div class="silva-score">{score:.3f}</div>
      <div class="silva-track"><div class="silva-fill" style="width:{pct}%"></div></div>
      <div class="silva-scale"><span>0</span><span class="silva-stars">&asymp; {stars:.1f} / 5</span><span>1</span></div>
      {warning}
    </div>
    """


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,340;9..144,500&family=Schibsted+Grotesk:wght@400;500&display=swap');

:root {
  --silva-paper: #f7f4ee;
  --silva-ink: #211d19;
  --silva-faint: #8d8579;
  --silva-line: #e2dccf;
}

/* Editorial heading over Gradio's default sans. */
.silva-intro h1 {
  font-family: 'Fraunces', serif;
  font-weight: 500;
  letter-spacing: 0.05em;
  margin-bottom: 0.1em;
}

/* Gallery-label result card: paper in both light and dark mode. */
.silva-card {
  border: 1px solid var(--silva-line);
  border-radius: 4px;
  padding: 40px 32px 28px;
  background: var(--silva-paper);
  color: var(--silva-ink);
  text-align: center;
}
.silva-card > * { animation: silva-rise 0.5s ease both; }
.silva-empty {
  color: var(--silva-faint);
  font-size: 0.95rem;
  padding: 56px 32px;
  border-style: dashed;
}
.silva-label {
  font-size: 0.72rem;
  font-weight: 500;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  color: var(--silva-faint);
}
.silva-score {
  font-family: 'Fraunces', serif;
  font-weight: 340;
  font-size: 4.4rem;
  line-height: 1.15;
  letter-spacing: -0.01em;
}
.silva-track {
  margin: 14px auto 10px;
  height: 2px;
  background: var(--silva-line);
}
.silva-fill {
  height: 100%;
  background: var(--silva-ink);
  transform-origin: left;
  animation: silva-grow 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.silva-scale {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.78rem;
  color: var(--silva-faint);
  font-variant-numeric: tabular-nums;
}
.silva-stars { font-size: 0.88rem; }
.silva-warning {
  margin-top: 18px;
  padding: 10px 14px;
  border: 1px solid #d9b88a;
  border-radius: 4px;
  background: #f6ead7;
  color: #7a5b2b;
  font-size: 0.85rem;
  text-align: left;
}
@keyframes silva-rise {
  from { opacity: 0; transform: translateY(6px); }
}
@keyframes silva-grow {
  from { transform: scaleX(0); }
}
"""

DESCRIPTION = """
# SILVA

**Aesthetic scoring for illustrations.** Upload an image to get a score in `[0, 1]` —
an ordinal-regression head reading a frozen
[SigLIP2](https://huggingface.co/google/siglip2-so400m-patch14-384) embedding.

The head is calibrated on a single rater's 1–5 labels, so read the score as one
consistent taste rather than a universal measure of quality. Inputs far from the
training distribution (photos, 3D, memes) are flagged as out of scope instead of
trusted. The first score after a cold start takes a while; later ones are quick.
"""

FOOTER = """
Head: [Jannchie/silva-aesthetic](https://huggingface.co/Jannchie/silva-aesthetic)
&nbsp;&middot;&nbsp; Source: [github.com/Jannchie/silva](https://github.com/Jannchie/silva)
"""

theme = gr.themes.Monochrome(
    font=[gr.themes.GoogleFont("Schibsted Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="SILVA Aesthetic Scorer", css=CSS, theme=theme) as demo:
    gr.Markdown(DESCRIPTION, elem_classes="silva-intro")
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Illustration", height=420)
            btn = gr.Button("Score", variant="primary")
        with gr.Column(scale=1):
            result = gr.HTML(_result_html(None))
    gr.Markdown(FOOTER, elem_classes="silva-footer")
    btn.click(score_image, inputs=image, outputs=result)

if __name__ == "__main__":
    demo.launch()
