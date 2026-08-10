#!/usr/bin/env python
"""Try candidate trajectory statistics before committing one to the codebase.

    python scripts/candidate_statistics.py <search_run> --confirm <heldout_run>

`scripts/scale_control.py` showed that almost every statistic in the study is really
measuring one thing -- how far the pooled trajectory moves -- and that the two ways of
removing that (a standard deviation, or a coefficient of variation) land at chance and
below chance respectively. Both discard the *time ordering*: shuffle the frames and they
are unchanged. A physics violation is a localised event, so an order-sensitive statistic
has something left to find where a distributional one does not.

This scores a battery of candidates so that question is answered by measurement. The
protocol matters as much as the numbers: searching a battery on one dataset and reporting
the winner is the selection error this project keeps guarding against, so `--confirm`
rescores the same battery on a held-out run and prints both columns. A candidate that
wins on the search set and lands at chance on the held-out one has not been found, it has
been fitted.

CPU only, and needs `probe.save_trajectories` on both runs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as stats_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from phaselock.datasets import get_paired_dataset
from phaselock.progress import track

EPS = 1e-8


def candidates(z: np.ndarray) -> dict[str, float]:
    """Every candidate for one `(T, D)` trajectory.

    Grouped by what they are sensitive to, because that is the axis the result turns on.
    """
    v = z[1:] - z[:-1]
    s = np.linalg.norm(v, axis=-1)
    energy = 0.5 * s**2
    de = np.diff(energy)
    scale = float(s.mean()) + EPS

    out: dict[str, float] = {
        # -- controls, to anchor the comparison ------------------------------
        "mean_speed  [scale]": float(s.mean()),
        "std_E  [scale]": float(energy.std()),
        "cv_E  [current phi_energy]": float(energy.std() / (energy.mean() + EPS)),
        "cv_speed": float(s.std() / scale),
    }
    if len(energy) < 4:
        return out

    centred = energy - energy.mean()
    denominator = float((centred**2).sum()) + EPS
    absolute = np.abs(de)

    out.update({
        # -- order-sensitive, scale-free -------------------------------------
        # A smooth exchange of energy is strongly autocorrelated frame to frame; an
        # inserted discontinuity is not.
        "1 - autocorr(E)": float(1.0 - (centred[:-1] * centred[1:]).sum() / denominator),
        # Is the energy change concentrated in one jolt, or spread over the clip?
        "crest(dE)": float(absolute.max() / (absolute.mean() + EPS)),
        # Total variation against range: how much the energy series wiggles relative to
        # how far it actually travels.
        "wiggle(E)": float(absolute.sum() / (energy.max() - energy.min() + EPS)),
        # Share of the whole clip's energy sitting in its single largest frame.
        "max share(E)": float(energy.max() / (energy.sum() + EPS)),
        # The same idea on the speed series rather than the squared one.
        "crest(speed)": float(s.max() / (s.mean() + EPS)),
        # Asymmetry: a violation that injects energy looks different from one that
        # removes it, and a symmetric spread cannot tell them apart.
        "skew(E)": float(((energy - energy.mean()) ** 3).mean()
                         / ((energy.std() + EPS) ** 3)),
        # -- order-sensitive, scale-carrying, for contrast --------------------
        "mean |dE|": float(absolute.mean()),
    })

    # -- energy continuity ---------------------------------------------------
    # Real motion changes its energy smoothly; a teleport, a penetration or an inserted
    # impulse breaks that at one instant. A spread over the whole clip cannot see it --
    # the same set of energies in a different order gives the same answer. These are
    # *local*: each is a per-frame relative jump, then summarised.
    level = float(energy.mean()) + EPS
    local = absolute / (energy[:-1] + energy[1:] + EPS)   # jump relative to its own level
    second = np.abs(np.diff(de)) if len(de) > 1 else np.array([0.0])
    out.update({
        "mean |dE| / mean E": float(absolute.mean() / level),
        "max |dE| / mean E": float(absolute.max() / level),
        "std(dE) / mean E": float(de.std() / level),
        "mean local jump": float(local.mean()),
        "max local jump": float(local.max()),
        "mean |d2E| / mean E": float(second.mean() / level),
        # What fraction of frames carry an unusually large jump: one discontinuity in an
        # otherwise smooth clip gives a small number, uniformly jittery energy a large one.
        "burst fraction": float((absolute > 2.0 * np.median(absolute) + EPS).mean()),
    })
    return out


def score(run_dir: Path, source: str, block_stride: int
          ) -> dict[str, tuple[float, float, float, float]]:
    """Per candidate: accuracy, its spread across cells, the tie rate, and the
    tie-excluded accuracy."""
    config = json.loads((run_dir / "config.json").read_text())
    dataset = get_paired_dataset(config["data"]["name"])
    pairs = dataset.select(limit=config["data"]["limit"], seed=config["data"]["seed"])
    trajectories = run_dir / "trajectories"
    saved = {p.stem for p in trajectories.glob("*.npz")}
    usable = [p for p in pairs
              if p.plausible.sample_id.replace("/", "_") in saved
              and p.violated.sample_id.replace("/", "_") in saved]
    if not usable:
        raise SystemExit(f"no pairs with both clips saved under {run_dir}")

    load = lambda s: np.load(trajectories / (s.sample_id.replace("/", "_") + ".npz"))
    available = [k for k in load(usable[0].plausible).keys() if k.split("|")[0] == source]

    def block_of(key: str) -> int:
        part = key.split("|")[1]
        return int(part[1:]) if part.startswith("b") else -1

    blocks = sorted({block_of(k) for k in available})
    keep = set(blocks[:: max(1, block_stride)])
    cells = [k for k in available if block_of(k) in keep]
    print(f"  {len(usable)} pairs x {len(cells)} cells of {source}")

    # Three-way, not two. The scoring rule is `violated > plausible`, so a TIE counts as a
    # miss -- and a candidate that takes few distinct values ties constantly, which drags
    # it below 50% with no effect behind it. A burst-fraction candidate read 39% on both
    # datasets and looked like the one real cross-dataset finding here; it was 22-30% ties
    # and 49% once they were excluded. Report both, always.
    hits: dict[str, dict[str, list[int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0, 0, 0]))
    for pair in track(usable, f"scoring {run_dir.parent.name}", unit="pair"):
        low_record, high_record = load(pair.plausible), load(pair.violated)
        for key in cells:
            low = candidates(low_record[key].astype(np.float64))
            high = candidates(high_record[key].astype(np.float64))
            for name in low:
                counts = hits[name][key]
                if high[name] > low[name]:
                    counts[0] += 1
                elif high[name] < low[name]:
                    counts[1] += 1
                else:
                    counts[2] += 1

    summary = {}
    for name, per_cell in hits.items():
        scored = [100 * up / (up + down + tie) for up, down, tie in per_cell.values()]
        ties = [100 * tie / (up + down + tie) for up, down, tie in per_cell.values()]
        untied = [100 * up / (up + down) for up, down, _ in per_cell.values() if up + down]
        summary[name] = (
            stats_module.mean(scored),
            stats_module.stdev(scored) if len(scored) > 1 else 0.0,
            stats_module.mean(ties),
            stats_module.mean(untied) if untied else float("nan"),
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("search_run", type=Path)
    parser.add_argument("--confirm", type=Path, default=None,
                        help="held-out run to rescore the same battery on")
    parser.add_argument("--source", default="hidden_states")
    parser.add_argument("--block-stride", type=int, default=5)
    parser.add_argument("--out", default="candidate_statistics.csv")
    args = parser.parse_args()

    print(f"search set: {args.search_run}")
    search = score(args.search_run, args.source, args.block_stride)
    confirm = None
    if args.confirm is not None:
        print(f"\nheld-out set: {args.confirm}")
        confirm = score(args.confirm, args.source, args.block_stride)

    # Rank by the tie-excluded number: it is the one that cannot be faked by a coarse
    # candidate, and the raw column exists to show when the two disagree.
    order = sorted(search, key=lambda n: -abs(search[n][3] - 50))
    width = max(len(n) for n in order) + 2
    header = (f"{'candidate':<{width}}{'search':>10}{'ties':>7}{'no ties':>9}")
    if confirm:
        header += f"{'held out':>11}{'ties':>7}{'no ties':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name in order:
        mean, _, ties, untied = search[name]
        line = f"{name:<{width}}{mean:>9.1f}%{ties:>6.0f}%{untied:>8.1f}%"
        if confirm and name in confirm:
            c_mean, _, c_ties, c_untied = confirm[name]
            line += f"{c_mean:>10.1f}%{c_ties:>6.0f}%{c_untied:>8.1f}%"
        print(line)

    out = args.search_run / args.out
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "search_accuracy", "search_std", "search_tie_rate",
                         "search_accuracy_excluding_ties", "heldout_accuracy",
                         "heldout_std", "heldout_tie_rate",
                         "heldout_accuracy_excluding_ties"])
        for name in order:
            row = [name] + [round(v, 1) for v in search[name]]
            row += ([round(v, 1) for v in confirm[name]]
                    if confirm and name in confirm else ["", "", "", ""])
            writer.writerow(row)
    print(f"\nwrote {out}")
    print(
        "\nSorted by distance from chance on the tie-excluded column. Read that one: the\n"
        "scoring rule counts a tie as a miss, so a coarse candidate reads below 50 with no\n"
        "effect behind it, and a high tie rate next to a large raw gap is that artefact\n"
        "rather than a finding. A candidate is only interesting if the tie-excluded number\n"
        "is far from 50 on BOTH sets and on the same side; search-only is a fit."
    )


if __name__ == "__main__":
    main()
