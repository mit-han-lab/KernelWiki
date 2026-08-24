"""Load PyYAML when available, with an offline pure-Python fallback.

The skill's data and Markdown frontmatter are YAML.  Agents commonly invoke
the scripts with a system Python that has no site packages (or with
``python -S``), so importing PyYAML directly makes otherwise read-only query
commands fail before they can do any work.  Keep the dependency optional at
runtime: use the host package for normal installations and fall back to the
vendored pure-Python implementation shipped in ``_vendor``.

Only the module selection lives here; callers continue to use the familiar
``yaml.safe_load``/``yaml.dump`` API.  The fallback is the same PyYAML API, so
block scalars, dates, flow collections, and serializer behavior stay aligned
with maintenance tooling as well as query tooling.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _load_yaml_module():
    try:
        module = importlib.import_module("yaml")
        api = ("safe_load", "safe_dump", "YAMLError")
        if not all(hasattr(module, name) for name in api):
            raise ImportError("imported yaml module does not expose the PyYAML API")
        return module, False
    except (ImportError, OSError):
        # A partially installed PyYAML can leave a module entry behind.  Remove
        # it before importing the bundled package so the fallback is complete.
        for name in tuple(sys.modules):
            if name == "yaml" or name.startswith("yaml."):
                sys.modules.pop(name, None)

    vendor_root = Path(__file__).resolve().parent / "_vendor"
    sys.path.insert(0, str(vendor_root))
    try:
        return importlib.import_module("yaml"), True
    finally:
        sys.path.pop(0)


yaml, USING_BUNDLED = _load_yaml_module()

__all__ = ["yaml", "USING_BUNDLED"]
