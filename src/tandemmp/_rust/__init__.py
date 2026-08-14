"""Native (Rust/PyO3) extension modules for tandemmp.

Per ``WORKLIST.md``'s Rust-acceleration convention, each accelerated hot
path lives under its own subdirectory here (e.g. ``drift_correction/``),
built independently via `maturin <https://www.maturin.rs/>`_ rather than
through the project's main setuptools build. Compiled extensions install
directly into this namespace (``tandemmp._rust.<name>``); callers should
lazy-import them and fall back to the pure-Python implementation when the
extension hasn't been built (see ``tandemmp.drift.corrector``'s
``use_rust`` parameter for the pattern).

This package intentionally has no other content — it exists only so
``tandemmp._rust`` is a real (non-namespace) package that compiled
extensions can be installed into.
"""

from __future__ import annotations
