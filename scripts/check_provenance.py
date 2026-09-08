#!/usr/bin/env python3
"""Classify every CRAX source file against upstream Brax and write PROVENANCE.md.

CRAX is a fork of Brax (Apache-2.0). This script keeps the provenance record
honest: it downloads the upstream Brax source distribution, compares it against
the working tree file by file, and regenerates PROVENANCE.md from what it finds.

Files are compared with comments, blank lines and indentation removed, and with
the CRAX renamings undone (`crax` -> `brax`, top-level `training` ->
`brax.training`), so that a file which differs only by the rename is reported as
verbatim rather than as a modification.

Usage:
    python scripts/check_provenance.py                 # regenerate PROVENANCE.md
    python scripts/check_provenance.py --check         # fail if it is out of date
    python scripts/check_provenance.py --brax-src DIR  # use a local brax checkout
"""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BRAX_VERSION = "0.12.3"
OUTPUT = "PROVENANCE.md"
PACKAGES = ("crax", "training")

# Changed-line count at or below which a modification is called "trivial".
TRIVIAL_THRESHOLD = 12


def fetch_brax(version: str, dest: Path) -> Path:
    """Downloads and unpacks the Brax sdist, returning the `brax` package dir."""
    print(f"downloading brax=={version} ...", file=sys.stderr)
    subprocess.run(
        [sys.executable, "-m", "pip", "download", f"brax=={version}",
         "--no-deps", "--no-binary", ":all:", "-d", str(dest)],
        check=True, stdout=subprocess.DEVNULL,
    )
    tarball = next(dest.glob(f"brax-{version}.tar.gz"))
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)
    return dest / f"brax-{version}" / "brax"


def significant_lines(path: Path, undo_rename: bool = False) -> List[str]:
    """Reads a file as comparable lines: no comments, blanks or indentation."""
    text = path.read_text(errors="replace")
    if undo_rename:
        text = re.sub(r"\bfrom training\b", "from brax.training", text)
        text = re.sub(r"\bimport training\b", "import brax.training", text)
        text = text.replace("crax", "brax").replace("CRAX", "BRAX").replace("Crax", "Brax")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def upstream_path(rel: str, brax: Path) -> Optional[Path]:
    """Maps a CRAX path to the Brax path it would have come from."""
    if rel.startswith("crax/"):
        return brax / rel[len("crax/"):]
    if rel.startswith("training/"):
        return brax / "training" / rel[len("training/"):]
    return None


def changed_lines(upstream: Path, ours: Path) -> int:
    a = significant_lines(upstream)
    b = significant_lines(ours, undo_rename=True)
    matcher = difflib.SequenceMatcher(None, a, b)
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    )


def classify(repo: Path, brax: Path) -> Dict[str, List[Tuple[str, int, int]]]:
    """Returns {bucket: [(relpath, loc, changed_lines), ...]}."""
    buckets: Dict[str, List[Tuple[str, int, int]]] = {
        "verbatim": [], "trivial": [], "substantial": [], "original": [],
    }
    for package in PACKAGES:
        for path in sorted((repo / package).rglob("*.py")):
            rel = str(path.relative_to(repo))
            loc = len(significant_lines(path))
            upstream = upstream_path(rel, brax)
            if upstream is None or not upstream.exists():
                buckets["original"].append((rel, loc, 0))
                continue
            changed = changed_lines(upstream, path)
            if changed == 0:
                bucket = "verbatim"
            elif changed <= TRIVIAL_THRESHOLD:
                bucket = "trivial"
            else:
                bucket = "substantial"
            buckets[bucket].append((rel, loc, changed))
    return buckets


LABELS = {
    "verbatim": "Verbatim Brax",
    "trivial": "Brax, small changes",
    "substantial": "Brax, substantially modified",
    "original": "CRAX original",
}


def render(buckets: Dict[str, List[Tuple[str, int, int]]]) -> str:
    total_files = sum(len(v) for v in buckets.values())
    total_loc = sum(loc for v in buckets.values() for _, loc, _ in v)

    out: List[str] = []
    w = out.append
    w("# Provenance")
    w("")
    w("CRAX is a fork of [Brax](https://github.com/google/brax) "
      f"{BRAX_VERSION}, used under the Apache License 2.0. This file records "
      "which parts of the source tree come from Brax and which are CRAX's own, "
      "so that the distinction is visible without reading diffs.")
    w("")
    w("**This file is generated.** Run `python scripts/check_provenance.py` to "
      "regenerate it; do not edit it by hand.")
    w("")
    w("## Summary")
    w("")
    w("| Origin | Files | Lines of code | Share |")
    w("|---|---:|---:|---:|")
    for key in ("verbatim", "trivial", "substantial", "original"):
        files = buckets[key]
        loc = sum(l for _, l, _ in files)
        w(f"| {LABELS[key]} | {len(files)} | {loc:,} | {100 * loc / total_loc:.0f}% |")
    w(f"| **Total** | **{total_files}** | **{total_loc:,}** | |")
    w("")
    w("Lines of code exclude comments and blank lines. Files are compared "
      "against the upstream Brax source with the CRAX renamings undone, so a "
      "file that differs only because `brax` was renamed to `crax`, or because "
      "`brax.training` became the top-level `training` package, counts as "
      "verbatim.")
    w("")
    w("## Why CRAX vendors Brax rather than depending on it")
    w("")
    w("Brax is vendored rather than imported for two reasons:")
    w("")
    w("1. **Dependency weight.** Brax's own install pulls in flask, flask_cors, "
      "jinja2, grpcio, gym, pytinyrenderer, scipy, optax and orbax-checkpoint. "
      "Depending on it would put all of those in CRAX's base install, when the "
      "point of the `crax` package is that it can be installed without an RL "
      "stack at all.")
    w("2. **Changes inside Brax internals.** Several modifications cannot be "
      "expressed as subclasses: `crax/io/mjcf.py` loads collision functions "
      "from `crax.mjx.collisions`, `crax/mjx/pipeline.py` changes the signature "
      "of `_reformat_contact`, and `crax/envs/base.py` alters `PipelineEnv` "
      "construction and `Wrapper.render`.")
    w("")
    w("Every file derived from Brax keeps the original copyright header and "
      "carries a notice describing how it was changed, as required by section "
      "4(b) of the Apache License. See also [NOTICE](NOTICE).")
    w("")

    sections = [
        ("substantial", "Files taken from Brax and substantially modified. Each "
                        "carries a header notice describing the changes."),
        ("trivial", "Files taken from Brax with small changes (at most "
                    f"{TRIVIAL_THRESHOLD} lines). Each carries a header notice."),
        ("verbatim", "Files used unchanged from Brax, apart from the package "
                     "rename. These keep their original Brax copyright header."),
        ("original", "Files written for CRAX, with no Brax counterpart."),
    ]
    for key, blurb in sections:
        files = buckets[key]
        w(f"## {LABELS[key]} ({len(files)} files)")
        w("")
        w(blurb)
        w("")
        if key in ("trivial", "substantial"):
            w("| File | LOC | Changed lines |")
            w("|---|---:|---:|")
            for rel, loc, changed in sorted(files, key=lambda r: -r[2]):
                w(f"| `{rel}` | {loc} | {changed} |")
        else:
            w("| File | LOC |")
            w("|---|---:|")
            for rel, loc, _ in sorted(files):
                w(f"| `{rel}` | {loc} |")
        w("")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brax-src", type=Path, default=None,
                        help="Path to an unpacked brax package directory "
                             "(default: download the sdist)")
    parser.add_argument("--check", action="store_true",
                        help="Exit non-zero if PROVENANCE.md is out of date")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        brax = args.brax_src or fetch_brax(BRAX_VERSION, Path(tmp))
        report = render(classify(repo, brax))

    target = repo / OUTPUT
    if args.check:
        current = target.read_text() if target.exists() else ""
        if current != report:
            print(f"{OUTPUT} is out of date; run python scripts/check_provenance.py",
                  file=sys.stderr)
            return 1
        print(f"{OUTPUT} is up to date")
        return 0

    target.write_text(report)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
