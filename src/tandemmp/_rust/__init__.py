"""Native (Rust/PyO3) extension modules for tandemmp.

Extensions are built separately with maturin and installed into this
package. Drift correction falls back to Python when the extension is
unavailable; see ``tandemmp.drift.corrector``'s ``use_rust`` parameter.
"""

from __future__ import annotations
