"""Model loading utilities for batch forecast runs.

The model is loaded once per batch run rather than per individual request, which keeps the
inference step efficient and consistent for multiple market and crop combinations.
"""

from __future__ import annotations

from typing import Any


def load_model(path: str) -> Any:
    """Load a serialized model from disk for a batch forecast run."""
    # TODO: implement model deserialization and return the loaded estimator.
    return None
