# silva-scorer

Scores an illustration by **one person's** aesthetic taste — an ordinal-regression head on top
of frozen `google/siglip2-so400m-patch14-384` embeddings. Only the head ships (~7 MB); it is not
a universal quality model and won't match anyone else's preferences.

## Install

The PyPI name is `silva-scorer`; it imports as `silva`.

```bash
pip install silva-scorer              # embedding -> score (torch + huggingface-hub only)
pip install "silva-scorer[backbone]"  # image -> score, adds the SigLIP2 backbone + `silva` CLI
```

## Score an image

With the `[backbone]` extra, `silva` loads SigLIP2 for you — give it a path (or a list of
them) and get a score back:

```python
from silva import AestheticScorer

scorer = AestheticScorer.from_pretrained("Jannchie/silva-aesthetic")
scorer.score("image1.jpg")                  # 0.7421
scorer.score(["image1.jpg", "image2.jpg"])  # [0.7421, 0.3128]
```

Or from the CLI:

```bash
silva score image1.jpg image2.jpg --repo-id Jannchie/silva-aesthetic
# image1.jpg  score=0.7421
```

## Score from an embedding

Already running `google/siglip2-so400m-patch14-384` yourself? The core install (no
`transformers`) scores a 1152-d embedding directly:

```python
import torch
from silva import HubAestheticModel

head = HubAestheticModel.from_pretrained("Jannchie/silva-aesthetic").eval()
emb = torch.randn(1, 1152)        # raw pooler_output from the backbone above
print(head(emb)["score"].item())  # [0, 1] — fraction of quality bars cleared
```

The embedding must be the raw `pooler_output` of that exact backbone — it's what the head was
trained against.

## Training your own head

`silva` is inference-only. To fit a head on your own 1–5 ratings, see the
[silva-train](https://github.com/Jannchie/silva/tree/main/packages/silva-train) package.

## License

MIT
