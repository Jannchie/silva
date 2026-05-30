# SILVA

A small ordinal-regression head that scores illustrations for one person's aesthetic
taste, on top of frozen `google/siglip2-so400m-patch14-384` embeddings.

```bash
pip install silva                # embedding -> score
pip install "silva[backbone]"    # image -> score + `silva` CLI
```

Weights live on the Hugging Face Hub; load with `silva.hub.HubAestheticModel.from_pretrained`.
