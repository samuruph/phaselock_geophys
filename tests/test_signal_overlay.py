"""Drawing computed signals under the videos. No GPU, no weights, no recomputation."""

from __future__ import annotations

import pathlib

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


def _frames(value, n=24, spike_from=None, spike=0.0):
    series = np.full(n, float(value))
    if spike_from is not None:
        series[spike_from:] += spike
    return series


def test_timeline_figure_draws_one_panel_per_per_frame_signal(tmp_path):
    low = {name: _frames(2.0) for name in signal_overlay.PER_FRAME}
    high = {name: _frames(2.0, spike_from=12, spike=0.8) for name in signal_overlay.PER_FRAME}
    path = signal_overlay.timeline_figure(low, high, tmp_path / "t.png", title="x")
    assert path is not None and path.is_file()


def test_timeline_figure_returns_none_without_per_frame_signals(tmp_path):
    # `ang` is a std over the clip and has no per-frame form, so it alone is not enough.
    assert signal_overlay.timeline_figure({"ang": _frames(1.0)}, {}, tmp_path / "t.png") is None


def test_timeline_limits_leave_room_below_for_the_delta_fill():
    """The gap is filled against the panel floor, so the floor must sit below the data."""
    low = {"speed": _frames(2.0)}
    high = {"speed": _frames(2.4)}
    limits = signal_overlay._timeline_limits(["speed"], low, high)
    floor, ceiling = limits["speed"]
    assert floor < 2.0, "no headroom for the fill"
    assert ceiling > 2.4


def test_timeline_strips_produce_one_frame_each_with_a_playhead(tmp_path):
    low = {name: _frames(2.0, n=10) for name in signal_overlay.PER_FRAME}
    high = {name: _frames(2.3, n=10) for name in signal_overlay.PER_FRAME}
    strips = signal_overlay.timeline_strips(low, high, width_px=800)
    assert len(strips) == 10
    # The playhead moves, so consecutive frames must differ.
    assert not np.array_equal(strips[0], strips[5])


def test_per_video_frame_holds_each_latent_across_its_video_frames():
    """21 latent measurements stretched over 81 video frames, not 81 measurements."""
    import torch

    from phaselock.backends import get_spec
    from phaselock.probes import LATENT, ProbeRecord, TrajectoryKey

    spec = get_spec("wan21_t2v_1_3b")
    torch.manual_seed(0)
    record = ProbeRecord(
        trajectories={TrajectoryKey(LATENT, -1, 0): torch.randn(21, 8)},
        taus={0: 1.0},
        provenance={},
    )
    series = signal_overlay.per_video_frame(record, spec, source=LATENT)
    assert set(series) <= set(signal_overlay.PER_FRAME)
    speed = series["speed"]
    assert speed.size == spec.default_num_frames == 81
    # Distinct values cannot exceed the latent count, since each is held across its frames.
    assert len(set(speed[~np.isnan(speed)].tolist())) <= 21


def test_intermediates_are_placed_on_the_latents_they_describe():
    """Right-aligning every series lags the second differences by a whole latent.

    Curvature and acceleration are centred on the middle of the three latents they
    touch, not the last, so right-aligning them puts the curve four video frames after
    the motion that caused it -- exactly wrong for a figure whose job is to show *when*
    the signal fires.
    """
    span = signal_overlay._latent_span
    assert list(span(0, "speed", 3)) == [0, 1]
    assert list(span(0, "curv", 3)) == [0, 1, 2]
    assert list(span(0, "accel", 3)) == [0, 1, 2]
    # The residual predicts one specific frame from the `order` before it.
    assert list(span(0, "perr", 3)) == [3]
    assert list(span(5, "perr", 2)) == [7]


def test_spread_averages_where_spans_overlap():
    """Consecutive accelerations share two of their three latents."""
    from phaselock.backends import get_spec

    spec = get_spec("wan21_t2v_1_3b")
    values = np.array([0.0, 3.0])  # spans 0..2 and 1..3
    out = signal_overlay._spread_over_span(values, "accel", 21, spec, order=3)
    # Latent 0 sees only the first value; latent 1 sees both and must average them.
    assert out[0] == pytest.approx(0.0)
    from phaselock.analysis.latent_motion import frames_for_latent

    shared = list(frames_for_latent(1, spec))[0]
    assert out[shared] == pytest.approx(1.5)


def test_spread_leaves_uncovered_frames_as_nan():
    from phaselock.backends import get_spec

    spec = get_spec("wan21_t2v_1_3b")
    out = signal_overlay._spread_over_span(np.array([1.0]), "speed", 21, spec, order=3)
    assert not np.isnan(out[0]), "latent 0 is covered"
    assert np.isnan(out[-1]), "the tail has no value and must not be invented"


def test_timeline_accepts_a_single_series_for_a_generated_clip():
    """A generation has no counterpart, so one curve is a legitimate input.

    Inventing a comparison, or labelling the lone series 'plausible', would both be worse
    than drawing one line and saying so.
    """
    import matplotlib.pyplot as plt

    one = {name: _frames(2.0, n=16) for name in signal_overlay.PER_FRAME}
    captured = []
    original = plt.Axes.legend

    def record(self, *args, **kwargs):
        handles, labels = self.get_legend_handles_labels()
        captured.extend(labels)
        return original(self, *args, **kwargs)

    plt.Axes.legend = record
    try:
        import tempfile

        path = signal_overlay.timeline_figure(
            one, None, pathlib.Path(tempfile.mkdtemp()) / "g.png",
            series_label="generated",
        )
    finally:
        plt.Axes.legend = original

    assert path is not None and path.is_file()
    assert "generated" in captured, captured
    assert "violated" not in captured, "no counterpart exists to draw"
