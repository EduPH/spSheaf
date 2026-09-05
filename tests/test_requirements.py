"""requirements.txt must not drift from pyproject.toml.

The dependencies are declared once, in pyproject.toml. requirements.txt is
generated from it because people and tools expect one to exist. Two
hand-maintained copies of the same list diverge quietly -- someone adds a
dependency to one, a fresh clone installs from the other and fails on an
import -- so the generated file is checked in and this test guards it.
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_requirements_matches_pyproject():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "gen_requirements.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_declared_dependency_is_importable():
    """A declared dependency that cannot be imported is a packaging error."""
    import importlib

    # distribution name -> module name, where they differ
    modules = {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "scikit-learn": "sklearn", "matplotlib": "matplotlib",
        "pyarrow": "pyarrow", "anndata": "anndata", "h5py": "h5py",
    }
    missing = [d for d, m in modules.items() if not importlib.util.find_spec(m)]
    assert not missing, f"declared but not importable: {missing}"
