# silva

Scores an illustration by **one person's** aesthetic taste — an ordinal-regression head on top
of frozen `google/siglip2-so400m-patch14-384` embeddings. Only the head ships (~7 MB); it is not
a universal quality model and won't match anyone else's preferences.

## Install

```bash
pip install silva              # embedding -> score (torch + huggingface-hub only)
pip install "silva[backbone]"  # image -> score, adds the SigLIP2 backbone + `silva` CLI
```

## Score from an embedding

The core install takes a 1152-d SigLIP2 embedding you computed yourself and returns a score:

```python
import torch
from silva import HubAestheticModel

head = HubAestheticModel.from_pretrained("Jannchie/silva-aesthetic").eval()
emb = torch.randn(1, 1152)          # your SigLIP2-SO400M-384 pooler_output
out = head(emb)
print(out["score"].item())          # [0, 1] — fraction of quality bars cleared
print(out["ordinal_score"].item())  # [1, 5] — label space
```

## Score from an image

With the `[backbone]` extra, `silva` runs SigLIP2 for you. From the CLI:

```bash
silva score image1.jpg image2.jpg --repo-id Jannchie/silva-aesthetic
# image1.jpg  score=0.7421  ordinal=3.9684
```

Or in Python:

```python
from PIL import Image
from silva import HubAestheticModel
from silva.backbone import Embedder, score_images

head = HubAestheticModel.from_pretrained("Jannchie/silva-aesthetic")
embedder = Embedder()  # loads frozen google/siglip2-so400m-patch14-384
images = [Image.open("image1.jpg")]
print(score_images(images, head, embedder))  # [{"score": ..., "ordinal_score": ...}]
```

The backbone must be exactly `google/siglip2-so400m-patch14-384` with the raw `pooler_output` —
that is what the head was trained against. Anything else scores wrong.

## Training your own head

`silva` is inference-only. To fit a head on your own 1–5 ratings, see the
[silva-train](https://github.com/Jannchie/silva/tree/main/packages/silva-train) package.

## License

MIT
