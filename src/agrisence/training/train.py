"""Training orchestrator for the crop price forecasting workflow."""

from __future__ import annotations


def main() -> None:
    """Run the training pipeline.

    Intended flow:
    - load features
    - split chronologically
    - train baselines
    - train ensemble
    - evaluate
    - serialize outputs
    """
    # TODO: implement the end-to-end training orchestration sequence.
    pass


if __name__ == "__main__":
    main()
