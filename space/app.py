"""SILVA aesthetic scorer — Gradio demo for Hugging Face Spaces.

Upload an illustration, get the score this *one person's* taste model assigns it.

The published head (~7 MB) loads at import; the heavy SigLIP2 backbone is loaded
lazily on the first ``score`` call, so the first request after a cold start is slow
(tens of seconds on a free CPU Space) and every request after it is fast.

This is NOT a universal quality model -- the score reflects one specific person's
1-5 ratings and matches nobody else's preferences.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from silva import SilvaScorer

if TYPE_CHECKING:
    from PIL.Image import Image

REPO_ID = "Jannchie/silva-aesthetic"

# Load the head once at startup. The SigLIP2 backbone is pulled in lazily on the
# first scored image (see silva.scorer.SilvaScorer.embedder).
scorer = SilvaScorer.from_pretrained(REPO_ID)


def score_image(image: Image | None) -> str:
    """Score one PIL image and render the result card. Returns HTML."""
    if image is None:
        return _result_html(None)
    score = float(scorer.score(image))  # calibrated_score in [0, 1]
    return _result_html(score)


def _result_html(score: float | None) -> str:
    if score is None:
        return (
            "<div class='silva-card silva-empty'>"
            "Upload an illustration and press <b>Score</b>."
            "</div>"
        )
    pct = round(score * 100)
    # Friendly 1–5 estimate. calibrated_score is aligned to the 1–5 label
    # distribution, so a linear map back is a reasonable display approximation.
    stars = 1 + 4 * score
    return f"""
    <div class="silva-card">
      <div class="silva-score">{score:.3f}<span class="silva-range"> / 1.0</span></div>
      <div class="silva-track"><div class="silva-fill" style="width:{pct}%"></div></div>
      <div class="silva-stars">&#8776; {stars:.1f} / 5 &#9733; <span class="silva-note">(estimate)</span></div>
    </div>
    """


CSS = """
.silva-card {
  border-radius: 16px;
  padding: 28px 24px;
  background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
  color: #fff;
  text-align: center;
  box-shadow: 0 8px 30px rgba(79, 70, 229, 0.25);
}
.silva-empty {
  background: #1f2330;
  color: #9aa3b2;
  font-size: 0.95rem;
}
.silva-score {
  font-size: 3.2rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}
.silva-range { font-size: 1.1rem; font-weight: 500; opacity: 0.7; }
.silva-track {
  margin: 18px auto 12px;
  height: 10px;
  width: 88%;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.22);
  overflow: hidden;
}
.silva-fill {
  height: 100%;
  border-radius: 999px;
  background: #fff;
  transition: width 0.4s ease;
}
.silva-stars { font-size: 1.05rem; font-weight: 600; }
.silva-note { font-weight: 400; opacity: 0.7; font-size: 0.85rem; }
"""

DESCRIPTION = """
# 🎨 SILVA — Personal Aesthetic Scorer

Upload an illustration and see the score that **one specific person's** taste model
gives it. Output is a single number in `[0, 1]` — higher means *more to this person's
liking*.

> ⚠️ This is **not** a universal quality model. It was fit on one person's private
> 1–5 ratings, so the score reflects *their* preferences and won't match anyone else's.

Backbone: frozen [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384).
Head: [`Jannchie/silva-aesthetic`](https://huggingface.co/Jannchie/silva-aesthetic).
The first score after a cold start loads the backbone and takes a few tens of seconds.
"""

with gr.Blocks(title="SILVA Aesthetic Scorer", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="pil", label="Illustration", height=420)
            btn = gr.Button("Score", variant="primary")
        with gr.Column(scale=1):
            result = gr.HTML(_result_html(None))
    btn.click(score_image, inputs=image, outputs=result)

if __name__ == "__main__":
    demo.launch()
