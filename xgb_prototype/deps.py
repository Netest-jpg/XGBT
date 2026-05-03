"""deps.py — N4: Dependency version checker at import time."""
from __future__ import annotations

import importlib.metadata
import logging

_HARD = [
    ("xgboost",      "xgboost",            "1.7.0"),
    ("scikit-learn", "sklearn",            "1.3.0"),
    ("numpy",        "numpy",              "1.23.0"),
    ("pandas",       "pandas",             "1.5.0"),
]
_SOFT = [
    ("optuna",              "optuna",             "3.0.0"),
    ("plotly",              "plotly",             "5.0.0"),
    ("omegaconf",           "omegaconf",          "2.3.0"),
    ("scipy",               "scipy",              "1.9.0"),
    ("joblib",              "joblib",             "1.2.0"),
    ("great-expectations",  "great_expectations", "0.18.0"),
]


def check_deps() -> None:
    from packaging.version import Version

    _dep_log = logging.getLogger(__name__)
    issues: list[str] = []
    soft_issues: list[str] = []

    for pkg, _imp, min_ver in _HARD:
        try:
            installed = importlib.metadata.version(pkg)
            if Version(installed) < Version(min_ver):
                issues.append(
                    f"HARD  {pkg}=={installed} < required {min_ver}. "
                    f"Upgrade: pip install '{pkg}>={min_ver}'"
                )
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"HARD  '{pkg}' not installed. Run: pip install '{pkg}>={min_ver}'")

    for pkg, _imp, min_ver in _SOFT:
        try:
            installed = importlib.metadata.version(pkg)
            if Version(installed) < Version(min_ver):
                soft_issues.append(
                    f"SOFT  {pkg}=={installed} < recommended {min_ver}. "
                    f"Consider: pip install '{pkg}>={min_ver}'"
                )
        except importlib.metadata.PackageNotFoundError:
            pass

    for msg in soft_issues:
        _dep_log.warning("[DepCheck] %s", msg)

    if issues:
        raise ImportError(
            "\n[DepCheck] Hard dependency requirement(s) not met:\n  "
            + "\n  ".join(issues)
            + "\nFix the above before running training."
        )
    if not issues and not soft_issues:
        _dep_log.info("[DepCheck] All dependency versions OK.")


try:
    check_deps()
except ImportError as _dep_err:
    if "packaging" in str(_dep_err):
        pass
    else:
        raise