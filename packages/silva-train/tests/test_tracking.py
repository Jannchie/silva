from silva_train.tracking import build_log_with


def test_none_disables_tracking():
    assert build_log_with("none", "silva") is None


def test_builtin_tracker_name_passes_through():
    assert build_log_with("tensorboard", "silva") == "tensorboard"  # accelerate handles built-ins by string


def test_pandm_returns_official_tracker_instance():
    from pandm.integrations.accelerate import PandmTracker  # pandm's own accelerate tracker, not a reimplementation

    log_with = build_log_with("pandm", "silva", "run-1")

    assert isinstance(log_with, list)
    tracker = log_with[0]
    assert isinstance(tracker, PandmTracker)
    assert tracker.name == "pandm"  # accelerate must pass a custom tracker as an instance, not a string
