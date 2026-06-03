from silva.backbone import BACKBONE


def test_backbone_is_pinned_to_patch14():
    # Must match pictoria ai/siglip_embed.py — patch14, NOT patch16.
    assert BACKBONE == "google/siglip2-so400m-patch14-384"


def test_cli_main_is_importable_without_backbone_extra():
    # Importing the CLI entry point must not require transformers/pillow.
    from silva.cli import main

    assert callable(main)
