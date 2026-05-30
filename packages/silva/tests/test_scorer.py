import torch

from silva import AestheticScorer
from silva.models.aesthetic import EmbeddingAestheticModel


class DummyEmbedder:
    """Stands in for the SigLIP2 backbone: ignores the image, returns a fixed-shape vector."""

    def embed(self, image):
        return torch.randn(1, 16)


def make_scorer():
    scorer = AestheticScorer(EmbeddingAestheticModel(embedding_dim=16))
    scorer._embedder = DummyEmbedder()  # bypass lazy backbone load  # noqa: SLF001
    return scorer


def test_single_image_returns_a_float_in_unit_range():
    out = make_scorer().score(object())  # a single, non-list input
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0


def test_list_input_returns_list_of_floats():
    out = make_scorer().score([object(), object(), object()])
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in out)


def test_does_not_load_backbone_until_first_score():
    # Constructing the scorer must not touch transformers/pillow.
    scorer = AestheticScorer(EmbeddingAestheticModel(embedding_dim=16))
    assert scorer._embedder is None  # noqa: SLF001


def test_ordinal_score_is_not_in_public_api():
    import silva

    assert not hasattr(silva, "ordinal_score_from_logits")
