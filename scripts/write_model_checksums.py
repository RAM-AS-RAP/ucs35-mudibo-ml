"""Write SHA-256 checksums for the saved stack bundles.

Run manually once after validating the committed model files, and again only
when a model file is intentionally replaced. This script does not load or
modify model contents.
"""

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "results" / "artifacts" / "models"
MODEL_FILES = (
    "xgb_h1.json", "xgb_h2.json",
    "stack_h1_sklearn.joblib", "stack_h2_sklearn.joblib",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checksums = {}
    for filename in MODEL_FILES:
        path = MODEL_DIR / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file is missing: {path}")
        checksums[filename] = sha256_file(path)
    output = MODEL_DIR / "MODEL_CHECKSUMS.json"
    output.write_text(json.dumps(checksums, indent=2) + "\n", encoding="utf-8")
    for filename, digest in checksums.items():
        print(f"{filename}: {digest}")
    print(f"Wrote {output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()