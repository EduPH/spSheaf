"""Regenerate requirements.txt from pyproject.toml.

pyproject.toml declares the dependencies; requirements.txt exists because
people and tools expect one. Two hand-maintained lists of the same thing drift,
so this generates the second from the first and ``tests/test_requirements.py``
fails the build if it is stale.

    python tools/gen_requirements.py
"""

from __future__ import annotations

import pathlib
import sys

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: Extras folded into requirements.txt. "benchmark" is deliberately excluded:
#: those comparators are heavy and prone to conflicts, and a broken install
#: there must never stop the method from running.
EXTRAS = ("figures",)

HEADER = """\
# Runtime dependencies, plus what the figure checks need.
#
#   pip install -r requirements.txt
#
# GENERATED from pyproject.toml by tools/gen_requirements.py -- do not edit by
# hand. tests/test_requirements.py fails if the two drift apart.
#
# For an environment pinned to versions known to work, see
# requirements-lock.txt.
"""


def dependencies() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    deps = list(data["dependencies"])
    for extra in EXTRAS:
        deps += list(data.get("optional-dependencies", {}).get(extra, []))
    return deps


def render() -> str:
    lines = [HEADER, ""]
    lines += dependencies()
    lines += ["", "# the package itself, in editable mode", "-e ."]
    return "\n".join(lines) + "\n"


def main() -> int:
    out = ROOT / "requirements.txt"
    text = render()
    if "--check" in sys.argv:
        current = out.read_text() if out.exists() else ""
        if current != text:
            print("requirements.txt is stale; run python tools/gen_requirements.py")
            return 1
        print("requirements.txt is up to date")
        return 0
    out.write_text(text)
    print(f"wrote {out} ({len(dependencies())} requirements)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
