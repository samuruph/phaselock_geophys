"""The mp4 writers. No GPU, no weights -- synthetic tensors only."""

from __future__ import annotations

import cv2
import pytest
import torch

from phaselock.analysis import video


def clip(frames: int = 6, height: int = 32, width: int = 48, value: float = 0.5):
    return torch.full((frames, 3, height, width), value)


def read(path):
    capture = cv2.VideoCapture(str(path))
    try:
        out = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            out.append(frame)
    finally:
        capture.release()
    return out


def test_grid_writes_a_readable_mp4(tmp_path):
    path = video.write_grid({"a": clip(), "b": clip(value=0.2)}, tmp_path / "g.mp4")
    frames = read(path)
    assert len(frames) == 6
    # Two panels wide plus the gutter, one label band tall, rounded up to even so
    # the codec does not crop.
    assert frames[0].shape[1] == 48 * 2 + 4
    assert frames[0].shape[0] == 32 + video._band_height(48)


def test_short_clips_hold_their_last_frame(tmp_path):
    """A panel that ends early must freeze, not go black.

    Going black mid-playback reads as "the model collapsed" when it only means the clip
    was shorter, which is exactly the misreading these videos exist to prevent.
    """
    long, short = clip(frames=8, value=0.9), clip(frames=3, value=0.9)
    path = video.write_grid({"long": long, "short": short}, tmp_path / "h.mp4")
    frames = read(path)
    assert len(frames) == 8

    band = video._band_height(48)
    right = frames[-1][band:, 48 + 3 :]
    assert right.mean() > 200, "short panel went dark instead of holding"


def test_all_clips_must_share_a_frame_size(tmp_path):
    with pytest.raises(ValueError, match="share a frame size"):
        video.write_grid({"a": clip(), "b": clip(height=16)}, tmp_path / "x.mp4")


def test_empty_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no clips"):
        video.write_grid({}, tmp_path / "x.mp4")


def test_pair_video_puts_the_two_clips_side_by_side(tmp_path):
    path = video.pair_video(
        clip(value=1.0), clip(value=0.0), tmp_path / "p.mp4",
        scenario="ball_drop", violation="penetration",
    )
    frame = read(path)[0]
    band = video._band_height(48)
    left, right = frame[band:, :48], frame[band:, 48 + 4 :]
    assert left.mean() > 200 and right.mean() < 55, "panels are not in the expected order"


def test_inversion_video_has_one_panel_per_step_plus_the_original(tmp_path):
    steps = [(0.0, clip()), (0.5, clip()), (1.0, clip())]
    path = video.inversion_video(steps, tmp_path / "i.mp4", original=clip())
    frame = read(path)[0]
    assert frame.shape[1] == 48 * 4 + 4 * 3


def test_inversion_video_columns_wrap_into_rows(tmp_path):
    steps = [(t / 4, clip()) for t in range(4)]
    path = video.inversion_video(steps, tmp_path / "i.mp4", columns=2)
    frame = read(path)[0]
    assert frame.shape[1] == 48 * 2 + 4
    assert frame.shape[0] == (32 + video._band_height(48)) * 2 + 4


def test_panels_are_captioned_with_the_diffusion_timestep(tmp_path, monkeypatch):
    """tau is the shared coordinate (1 = clean), and panels show the timestep.

    Regression: the callback used to hand out the backend's raw scheduler level, which
    is a 0-1000 timestep for Wan, so the caption computed (1 - 1000) * 1000 and read
    't=-999000'. Anything captioned off tau must agree with what statistics.csv stores.
    """
    seen = {}
    monkeypatch.setattr(
        video, "write_grid",
        lambda clips, path, **kw: (seen.update(names=list(clips), subs=kw.get("subtitles")), path)[1],
    )
    video.inversion_video([(1.0, clip()), (0.007, clip())], tmp_path / "i.mp4")
    assert seen["names"] == ["t=0", "t=993"], seen["names"]
