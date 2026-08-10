"""The shared progress helper. No GPU."""

from __future__ import annotations

from phaselock.progress import DISABLE_ENV, track


def test_track_yields_every_item_unchanged():
    assert list(track(range(5), "x")) == [0, 1, 2, 3, 4]


def test_track_is_a_passthrough_when_disabled(monkeypatch):
    """A non-interactive log should not be polluted with carriage returns."""
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert list(track(iter([1, 2, 3]), "x")) == [1, 2, 3]


def test_track_handles_an_iterator_with_no_length():
    """A generator has no len(); the bar must degrade rather than raise."""
    assert list(track((i for i in range(4)), "x")) == [0, 1, 2, 3]


def test_track_survives_tqdm_being_absent(monkeypatch):
    import builtins

    real = builtins.__import__

    def fail(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    assert list(track(range(3), "x")) == [0, 1, 2]
