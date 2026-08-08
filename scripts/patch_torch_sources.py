"""Idempotent fix for a known torch + PyInstaller incompatibility on Python 3.12.

Frozen torch imports can fail with ``NameError: name 'name' is not defined`` in
``torch/_numpy/_ufuncs.py`` (module-level ``for name in _binary/_unary`` loop
that does ``vars()[name] = ...`` collides with a list comprehension that also
uses ``name``; PEP 709 inlining + PyInstaller's bytecode handling break the
binding). Patching the installed source before PyInstaller compiles it makes
the frozen app self-contained regardless of how the Windows Python was set up.

This script is safe to run repeatedly: it only rewrites the file when the
original (broken) pattern is found and prints a warning otherwise.

Run (on the build machine, from the project root):
    python scripts\\patch_torch_sources.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

_BINARY_OLD = """_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul", "divmod", "ldexp"]
]"""

_BINARY_NEW = """_binary = [
    n
    for n in dir(_binary_ufuncs_impl)
    if not n.startswith("_") and n not in ["torch", "matmul", "divmod", "ldexp"]
]"""

_UNARY_OLD = """_unary = [
    name
    for name in dir(_unary_ufuncs_impl)
    if not name.startswith("_") and name != "torch"
]"""

_UNARY_NEW = """_unary = [
    n
    for n in dir(_unary_ufuncs_impl)
    if not n.startswith("_") and n != "torch"
]"""


def _find_ufuncs() -> str | None:
    try:
        spec = importlib.util.find_spec("torch._numpy")
    except (ImportError, ValueError):
        spec = None
    if spec is None or spec.origin is None:
        return None
    return os.path.join(os.path.dirname(spec.origin), "_ufuncs.py")


def main() -> int:
    target = _find_ufuncs()
    if target is None or not os.path.isfile(target):
        print("patch_torch_sources: torch._numpy not found, skipping.", flush=True)
        return 0

    try:
        with open(target, encoding="utf-8") as fh:
            content = fh.read()
    except OSError as exc:
        print(f"patch_torch_sources: cannot read {target}: {exc}", file=sys.stderr)
        return 1

    if _BINARY_NEW in content and _UNARY_NEW in content:
        print("patch_torch_sources: torch._numpy._ufuncs already patched.", flush=True)
        return 0

    updated = content
    for old, new, label in (
        (_BINARY_OLD, _BINARY_NEW, "binary"),
        (_UNARY_OLD, _UNARY_NEW, "unary"),
    ):
        if old in updated:
            updated = updated.replace(old, new)
            print(f"patch_torch_sources: patched {label} comprehension in {target}")
        else:
            print(
                f"patch_torch_sources: {label} pattern not found in {target}; "
                "torch version may differ from the expected one.",
                file=sys.stderr,
            )

    if updated == content:
        print(
            "patch_torch_sources: nothing to patch (expected pattern missing). "
            "Build will continue.",
            flush=True,
        )
        return 0

    try:
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(updated)
    except OSError as exc:
        print(f"patch_torch_sources: cannot write {target}: {exc}", file=sys.stderr)
        return 1

    print(f"patch_torch_sources: done, patched {target}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
