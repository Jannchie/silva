---
title: SILVA Aesthetic Scorer
emoji: 🎨
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
models:
  - Jannchie/silva-aesthetic
tags:
  - aesthetic
  - siglip2
  - ordinal-regression
---

# SILVA — Personal Aesthetic Scorer

Upload an illustration and see the score that **one specific person's** taste model
gives it. Output is a single number in `[0, 1]` — higher means *more to this person's
liking*.

> ⚠️ This is **not** a universal quality model. It was fit on one person's private
> 1–5 ratings, so the score reflects *their* preferences and won't match anyone else's.

## How it works

- **Backbone** (frozen): [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384)
  turns the image into a 1152-d embedding.
- **Head** (~7 MB): [`Jannchie/silva-aesthetic`](https://huggingface.co/Jannchie/silva-aesthetic),
  an ordinal-regression head that maps the embedding to a calibrated score.

The first score after a cold start loads the backbone and takes a few tens of seconds
on the free CPU hardware; subsequent scores are fast.

## Run it yourself

```bash
pip install "silva-scorer[backbone]"
silva score image.jpg --repo-id Jannchie/silva-aesthetic
```

Source & training code: <https://github.com/Jannchie/silva>.
