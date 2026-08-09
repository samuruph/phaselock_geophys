"""Drawing computed signals under the videos. No GPU, no weights, no recomputation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phaselock.analysis import signal_overlay, video


def rows(sample_id: str, source: str = "hidden_states", block: int = 3, scale: float = 1.0):
    """Minimal `statistics.csv` rows for one clip at one probe location."""
    return [
        {
            "sample_id": sample_id, "source": source, "block": str(block),
            "step": str(step), "tau": str(1.0 - step / 10),
            "phi_speed": str(scale * (step + 1)), "phi_curv": str(scale * 2.0),
            "phi_ang": str(scale * 0.3), "phi_accel": str(scale * 5.0),
            "phi_perr": str(scale * 0.8),
        }
        for step in range(4)
    ]


def test_per_step_selects_one_clip_at_one_location():
    everything = rows("a") + rows("b", scale=2.0) + rows("a", source="latent", block=-1)
    series = signal_overlay.per_step(everything, "a", "hidden_states", 3)
    assert set(series) == {"speed", "curv", "ang", "accel", "perr"}
    assert series["speed"] == [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)]


def test_per_step_returns_steps_in_order_whatever_the_row_order():
    shuffled = list(reversed(rows("a")))
    assert [s for s, _ in signal_overlay.per_step(shuffled, "a", "hidden_states", 3)["speed"]] == [0, 1, 2, 3]


def test_per_step_is_empty_for_a_location_that_was_not_probed():
    assert signal_overlay.per_step(rows("a"), "a", "hidden_states", 99) == {}


def test_choose_location_defaults_to_a_middle_hidden_state_block():
    everything = sum((rows("a", block=b) for b in (0, 5, 10, 15)), [])
    assert signal_overlay.choose_location(everything) == ("hidden_states", 10)


def test_choose_location_accepts_an_explicit_block():
    assert signal_overlay.choose_location(rows("a"), "hidden_states/b22") == ("hidden_states", 22)


def test_choose_location_falls_back_when_hidden_states_were_not_recorded():
    only_latent = rows("a", source="latent", block=-1)
    assert signal_overlay.choose_location(only_latent) == ("latent", -1)


def test_strip_is_the_requested_width():
    strip = signal_overlay.signal_strip(
        signal_overlay.per_step(rows("a"), "a", "hidden_states", 3),
        signal_overlay.per_step(rows("b", scale=2.0), "b", "hidden_states", 3),
        width_px=900,
    )
    assert strip.shape[1] == 900
    assert strip.ndim == 3 and strip.shape[2] == 3


def test_strip_needs_something_to_draw():
    with pytest.raises(ValueError, match="no statistics"):
        signal_overlay.signal_strip({}, {}, width_px=400)


def test_shading_marks_violated_above_plausible_as_correct():
    """Every statistic is oriented so larger means less regular.

    So violated above plausible is the signal calling that pair correctly, and the panel
    label must count it that way round -- getting this backwards would invert the reading
    of every strip.
    """
    import matplotlib.pyplot as plt

    low = signal_overlay.per_step(rows("a", scale=1.0), "a", "hidden_states", 3)
    high = signal_overlay.per_step(rows("b", scale=2.0), "b", "hidden_states", 3)

    captured: list[str] = []
    original = plt.Axes.set_xlabel

    def record(self, label, *args, **kwargs):
        captured.append(str(label))
        return original(self, label, *args, **kwargs)

    plt.Axes.set_xlabel = record
    try:
        signal_overlay.signal_strip(low, high, width_px=800)   # violated is the larger one
        signal_overlay.signal_strip(high, low, width_px=800)   # and now inverted
    finally:
        plt.Axes.set_xlabel = original

    correct, inverted = captured[:5], captured[5:10]
    assert all("correct at 4/4" in label for label in correct), correct
    assert all("correct at 0/4" in label for label in inverted), inverted


def test_attach_stacks_the_strip_under_every_frame(tmp_path):
    source = video.write_grid({"a": torch.full((5, 3, 40, 60), 0.5)}, tmp_path / "v.mp4")
    strip = np.full((30, 60, 3), 200, dtype=np.uint8)

    out = signal_overlay.attach(source, tmp_path / "v_signals.mp4", strip)
    import cv2

    capture = cv2.VideoCapture(str(out))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    assert len(frames) == 5
    original_height = 40 + video._band_height(60)
    assert frames[0].shape[0] == original_height + 30
    # The strip is light and sits at the bottom.
    assert frames[0][-15:].mean() > 150


def test_attach_rescales_a_strip_that_does_not_match_the_video_width(tmp_path):
    source = video.write_grid({"a": torch.full((4, 3, 40, 60), 0.5)}, tmp_path / "v.mp4")
    out = signal_overlay.attach(
        source, tmp_path / "w.mp4", np.full((20, 300, 3), 180, dtype=np.uint8)
    )
    import cv2

    capture = cv2.VideoCapture(str(out))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    capture.release()
    assert width == 60, "strip should be resized to the video, not the reverse"


def test_attach_rejects_a_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):
        signal_overlay.attach(
            tmp_path / "nope.mp4", tmp_path / "o.mp4", np.zeros((10, 10, 3), dtype=np.uint8)
        )
