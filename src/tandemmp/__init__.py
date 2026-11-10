"""TandemMP — ratiometric iSCAT microscopy analysis."""

import os

# The ghost model needs Keras 2 (tf-keras); must be set before tensorflow is
# imported. An explicit TF_USE_LEGACY_KERAS in the environment takes precedence.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__version__ = "0.8.0-dev"
