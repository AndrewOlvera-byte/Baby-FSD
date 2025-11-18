from __future__ import annotations
import importlib, pkgutil
from typing import Iterable

def _import_all_from(package_name: str, subpackages: Iterable[str] | None = None):
    """
    Import all modules in a package (and optionally subpackages).
    This triggers module-level @register decorators without hardcoding imports.
    """
    try:
        pkg = importlib.import_module(package_name)
    except ModuleNotFoundError:
        return

    # import leaf modules directly in the package
    if hasattr(pkg, "__path__"):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.import_module(m.name)
            except Exception:
                continue

    # import selected subpackages and their descendants
    if subpackages:
        for sub in subpackages:
            try:
                subpkg = importlib.import_module(f"{package_name}.{sub}")
            except ModuleNotFoundError:
                continue
            if hasattr(subpkg, "__path__"):
                for m in pkgutil.walk_packages(subpkg.__path__, subpkg.__name__ + "."):
                    try:
                        importlib.import_module(m.name)
                    except Exception:
                        continue

def bootstrap():
    """
    Import everything that can register components/trainers.
    Keep this list stable; add new component families here when you create them.
    """
    # trainers
    _import_all_from("src.trainers")

    # components
    _import_all_from("src.components", subpackages=[
        "envs",
        "datasets",
        "datacolls",
        "models",
        "losses",
        "collectors",
        "optimizers",
        "evaluators",
        "loggers",
    ])
