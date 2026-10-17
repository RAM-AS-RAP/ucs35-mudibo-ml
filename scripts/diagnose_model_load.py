"""Diagnose local stacking pickle files without modifying model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import traceback
import warnings


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ("stacking_h1.pkl",)
SKIP_DIRS = {".venv", "venv", "node_modules", ".git"}
SYNC_MARKERS = ("onedrive", "dropbox", "google drive", "icloud")
REPORT_PATH = REPO_ROOT / "artifacts" / "model_load_diagnosis.json"
REPORT = {"environment": {}, "models": {}, "checks": [], "conclusions": []}


def emit(status: str, label: str, message: str, details=None) -> dict:
    entry = {"status": status, "label": label, "message": message}
    if details is not None:
        entry["details"] = details
    REPORT["checks"].append(entry)
    print(f"{status}: {label}: {message}")
    if details is not None:
        lines = details if isinstance(details, list) else [details]
        for line in lines:
            print(f"  {line}")
    return entry


def version_for(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except Exception as exception:
        return f"ERROR {type(exception).__name__}: {exception}"


def capture_environment() -> None:
    environment = {
        "python": sys.version,
        "pointer_bits": struct.calcsize("P") * 8,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "inside_virtual_environment": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        "versions": {
            "xgboost": version_for("xgboost"),
            "scikit-learn": version_for("scikit-learn"),
            "joblib": version_for("joblib"),
            "numpy": version_for("numpy"),
            "pandas": version_for("pandas"),
        },
    }
    REPORT["environment"] = environment
    emit("PASS", "Environment", "Captured interpreter and installed package metadata", environment)


def classify_header(header: bytes) -> tuple[str, str]:
    stripped = header.lstrip()
    if len(header) >= 2 and header[0] == 0x80 and 2 <= header[1] <= 5:
        return "pickle-like", "PASS"
    if header.startswith(b"version https://git-lfs"):
        return "Git LFS pointer, not a model", "FAIL"
    if stripped.startswith((b"<!DOCTYPE", b"<html", b"<HTML")):
        return "HTML page, not a model", "FAIL"
    return "unknown header", "FAIL"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path) -> dict:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            header = stream.read(16)
        kind, status = classify_header(header)
        try:
            digest = hash_file(path)
            digest_status = "PASS"
        except Exception as exception:
            digest = None
            digest_status = "FAIL"
            emit("FAIL", f"SHA-256 {path.name}", f"{type(exception).__name__}: {exception}")
        ascii_header = "".join(chr(value) if 32 <= value <= 126 else "." for value in header)
        info = {
            "path": str(path.resolve()),
            "size_bytes": size,
            "sha256": digest,
            "header_hex": header.hex(),
            "header_ascii": ascii_header,
            "classification": kind,
            "header_status": status,
            "hash_status": digest_status,
        }
        emit(
            status,
            f"File header {path.name}",
            kind,
            [
                f"path={info['path']}",
                f"size_bytes={size}",
                f"sha256={digest}",
                f"first_16_hex={header.hex()}",
                f"first_16_ascii={ascii_header}",
            ],
        )
        return info
    except Exception as exception:
        emit("FAIL", f"Inspect file {path}", f"{type(exception).__name__}: {exception}")
        return {"path": str(path), "error": f"{type(exception).__name__}: {exception}"}


def find_model_copies() -> dict[str, list[dict]]:
    found = {name: [] for name in MODEL_NAMES}
    for current, directories, files in os.walk(REPO_ROOT):
        directories[:] = [name for name in directories if name not in SKIP_DIRS]
        for name in MODEL_NAMES:
            if name in files:
                found[name].append(describe_file(Path(current) / name))
    for name, copies in found.items():
        REPORT["models"][name] = {"copies": copies}
        if not copies:
            emit("FAIL", f"Copies of {name}", "No copy found in the repository")
        elif len(copies) > 1:
            emit("WARN", f"Copies of {name}", f"Found {len(copies)} copies", [item["path"] for item in copies])
        else:
            emit("PASS", f"Copies of {name}", "Found one repository copy", copies[0]["path"])
    return found


def import_loader():
    loader_file = REPO_ROOT / "src" / "agrisence" / "inference" / "model_loader.py"
    loader = None
    try:
        source_root = str(REPO_ROOT / "src")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from agrisence.inference import model_loader as loader

        paths = {
            horizon: Path(path).expanduser().resolve()
            for horizon, path in loader.MODEL_PATHS.items()
        }
        REPORT["loader"] = {
            "module_path": str(Path(loader.__file__).resolve()),
            "model_paths": {horizon: str(path) for horizon, path in paths.items()},
            "checksum_path": str(Path(loader.CHECKSUM_PATH).resolve()),
        }
        emit("PASS", "Import model loader", "Imported package loader and resolved MODEL_PATHS", REPORT["loader"])
        for horizon, path in paths.items():
            matching = [
                copy for copy in REPORT["models"].get(path.name, {}).get("copies", [])
                if Path(copy["path"]).resolve() == path
            ]
            details = [f"resolved_path={path}", f"exists={path.is_file()}"]
            if matching:
                details.append(f"matches measured copy={matching[0]['path']}")
                status = "PASS" if path.is_file() else "FAIL"
                message = "Loader uses the measured repository copy" if path.is_file() else "Loader path does not exist"
            else:
                status = "FAIL"
                message = "Loader path differs from all measured copies"
                details.extend(copy["path"] for copy in REPORT["models"].get(path.name, {}).get("copies", []))
            REPORT["models"].setdefault(path.name, {})["loader_path"] = str(path)
            REPORT["models"][path.name]["loader_path_matches_copy"] = bool(matching)
            emit(status, f"Loader path {horizon}", message, details)
        return loader, paths
    except Exception as exception:
        text = str(exception)
        emit("FAIL", "Import model loader", f"{type(exception).__name__}: {text}")
        context_lines = []
        try:
            lines = loader_file.read_text(encoding="utf-8").splitlines()
            hits = [index for index, line in enumerate(lines) if "MODEL_PATHS" in line or ".pkl" in line]
            included = set()
            for index in hits:
                for position in range(max(0, index - 3), min(len(lines), index + 4)):
                    if position not in included:
                        context_lines.append(f"{position + 1}: {lines[position]}")
                        included.add(position)
        except Exception as read_exception:
            context_lines.append(f"Could not read loader source: {type(read_exception).__name__}: {read_exception}")
        emit("WARN", "Loader source fallback", "Relevant loader source lines", context_lines)
        REPORT["loader"] = {"module_path": str(loader_file), "import_error": text, "source_context": context_lines}
        return loader, {}


def checksum_checks(loader, paths, copies) -> dict:
    if loader is None or not hasattr(loader, "CHECKSUM_PATH"):
        emit("FAIL", "Checksum manifest path", "Could not obtain checksum path from loader")
        return {}
    manifest_path = Path(loader.CHECKSUM_PATH).resolve()
    if not manifest_path.is_file():
        emit("FAIL", "Checksum manifest", f"Missing: {manifest_path}")
        REPORT["checksum_manifest"] = {"path": str(manifest_path), "exists": False}
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        REPORT["checksum_manifest"] = {
            "path": str(manifest_path), "exists": True,
            "modified": manifest_path.stat().st_mtime,
        }
        print(f"PASS: Checksum manifest: {manifest_path}; modified={manifest_path.stat().st_mtime}")
    except Exception as exception:
        emit("FAIL", "Read checksum manifest", f"{type(exception).__name__}: {exception}")
        return {}
    results = {}
    for horizon, name in (("h1", "stacking_h1.pkl"),):
        path = paths.get(horizon)
        stored = manifest.get(name) if isinstance(manifest, dict) else None
        measured = None
        modified = None
        if path is not None and path.is_file():
            try:
                measured = hash_file(path)
                modified = path.stat().st_mtime
            except Exception as exception:
                emit("FAIL", f"Checksum current {name}", f"{type(exception).__name__}: {exception}")
        status = "PASS" if isinstance(stored, str) and measured == stored.lower() else "FAIL"
        message = "Stored manifest digest matches loader file" if status == "PASS" else "Stored digest missing or differs from loader file"
        details = [f"stored_digest={stored}", f"loader_file_digest={measured}", f"model_modified={modified}"]
        emit(status, f"Manifest checksum {name}", message, details)
        results[horizon] = {"stored_digest": stored, "loader_digest": measured, "model_modified": modified, "matches": status == "PASS"}
        REPORT["models"].setdefault(name, {}).update(results[horizon])
    return results


def reference_hash_checks(args, paths, checksums) -> None:
    expected = {"h1": args.expected_h1}
    snippets = []
    for horizon in ("h1",):
        reference = expected[horizon]
        path = paths.get(horizon)
        if not reference:
            emit(
                "WARN", f"Original Colab hash {horizon}",
                "No reference hash supplied; the local copy cannot be proven identical to the original",
                "Colab: hashlib.sha256(open(path, 'rb').read()).hexdigest()",
            )
            snippets.append("hashlib.sha256(open(path, 'rb').read()).hexdigest()")
            continue
        actual = None
        try:
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Loader path missing for {horizon}")
            actual = hash_file(path)
            matches = actual.lower() == reference.strip().lower()
            status = "PASS" if matches else "FAIL"
            emit(status, f"Original Colab hash {horizon}", "Reference hash matches" if matches else "Reference hash differs", [f"reference={reference}", f"local={actual}"])
            REPORT["models"].setdefault(path.name, {}).update({"reference_digest": reference, "reference_matches": matches})
        except Exception as exception:
            emit("FAIL", f"Original Colab hash {horizon}", f"{type(exception).__name__}: {exception}")
    REPORT["colab_hash_snippet"] = snippets


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def path_and_sync_checks(paths) -> dict:
    root_text = str(REPO_ROOT.resolve()).lower()
    user_profile = Path(os.environ.get("USERPROFILE", str(Path.home())))
    redirected_parents = [user_profile / name for name in ("Desktop", "Documents")]
    under_redirected = any(_is_under(REPO_ROOT, parent) for parent in redirected_parents)
    sync = {}
    for horizon, path in paths.items():
        text = str(path.resolve()).lower()
        markers = [marker for marker in SYNC_MARKERS if marker in text]
        sync[horizon] = bool(markers)
        emit(
            "WARN" if markers else "PASS",
            f"Sync-folder path {horizon}",
            f"Matched sync folder markers: {markers}" if markers else "No named sync folder in model path",
            [f"model_path={path.resolve()}", f"repository_under_Desktop_or_Documents={under_redirected}"],
        )
        if os.name != "nt":
            emit("PASS", f"Cloud placeholder {horizon}", "not applicable")
        else:
            try:
                attributes = path.stat().st_file_attributes
                offline = bool(attributes & 0x1000)
                recall = bool(attributes & 0x400000)
                placeholder = offline or recall
                sync[horizon] = sync[horizon] or placeholder
                emit(
                    "WARN" if placeholder else "PASS",
                    f"Cloud placeholder {horizon}",
                    f"FILE_ATTRIBUTE_OFFLINE={offline}; FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS={recall}; attributes=0x{attributes:x}",
                )
            except Exception as exception:
                emit("FAIL", f"Cloud placeholder {horizon}", f"{type(exception).__name__}: {exception}")

        try:
            first = path.read_bytes()
            second = path.read_bytes()
            first_hash = hashlib.sha256(first).hexdigest()
            second_hash = hashlib.sha256(second).hexdigest()
            size = os.path.getsize(path)
            stable = len(first) == len(second) and first_hash == second_hash and len(first) == size
            emit(
                "PASS" if stable else "FAIL",
                f"Repeated read {horizon}",
                "Both full reads have equal length/hash and match stat size" if stable else "Repeated reads or stat size differ",
                [f"read1_length={len(first)}", f"read2_length={len(second)}", f"stat_size={size}", f"read1_sha256={first_hash}", f"read2_sha256={second_hash}"],
            )
        except Exception as exception:
            emit("FAIL", f"Repeated read {horizon}", f"{type(exception).__name__}: {exception}")
    if under_redirected:
        emit("WARN", "Repository location", "Repository is under Desktop or Documents, which Windows may redirect into cloud sync", str(REPO_ROOT))
    else:
        emit("PASS", "Repository location", "Repository is not under the user's Desktop or Documents", str(REPO_ROOT))
    REPORT["sync_or_placeholder"] = sync
    REPORT["repository_under_desktop_or_documents"] = under_redirected
    return sync


def git_and_line_ending_checks(paths) -> dict:
    results = {}
    for horizon, path in paths.items():
        try:
            relative = str(path.resolve().relative_to(REPO_ROOT.resolve())).replace(os.sep, "/")
        except ValueError:
            relative = str(path)
        commands = [
            ("core.autocrlf", ["git", "config", "--get", "core.autocrlf"]),
            ("check-attr", ["git", "check-attr", "-a", relative]),
            ("ls-files --eol", ["git", "ls-files", "--eol", relative]),
        ]
        outputs = {}
        for label, command in commands:
            try:
                result = subprocess.run(
                    command, cwd=REPO_ROOT, capture_output=True, text=True,
                    check=False, timeout=15,
                )
                output = (result.stdout or "").strip()
                error = (result.stderr or "").strip()
                outputs[label] = {"returncode": result.returncode, "stdout": output, "stderr": error}
                emit(
                    "PASS" if result.returncode == 0 else "WARN",
                    f"Git {label} {horizon}",
                    output or ("no output" if result.returncode == 0 else f"exit {result.returncode}"),
                    error if error else None,
                )
            except Exception as exception:
                outputs[label] = {"error": f"{type(exception).__name__}: {exception}"}
                emit("FAIL", f"Git {label} {horizon}", outputs[label]["error"])
        try:
            contents = path.read_bytes()
            crlf_count = contents.count(b"\r\n")
            lf_count = contents.count(b"\n")
            autocrlf = outputs.get("core.autocrlf", {}).get("stdout", "").strip().lower()
            eol_output = outputs.get("ls-files --eol", {}).get("stdout", "").lower()
            attr_output = outputs.get("check-attr", {}).get("stdout", "").lower()
            converted = "w/crlf" in eol_output or "text" in attr_output
            warn = autocrlf == "true" or converted
            emit(
                "WARN" if warn else "PASS",
                f"Line endings {horizon}",
                "Git may have rewritten line endings in a binary file" if warn else "No automatic text conversion indicated",
                [f"CRLF occurrences={crlf_count}", f"LF occurrences including CRLF={lf_count}", f"core.autocrlf={autocrlf or '(unset)'}", f"ls-files --eol={eol_output or '(no output)'}"],
            )
            outputs["line_endings"] = {"crlf_count": crlf_count, "lf_count": lf_count, "conversion_warning": warn}
            results[horizon] = outputs
        except Exception as exception:
            emit("FAIL", f"Line endings {horizon}", f"{type(exception).__name__}: {exception}")
            results[horizon] = outputs
    REPORT["git_checks"] = results
    return results


def print_traceback(status: str, label: str, exception: BaseException) -> None:
    rendered = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    emit(status, label, f"{type(exception).__name__}: {exception}", rendered.rstrip().splitlines())


def inspect_bundle(bundle, label: str) -> None:
    try:
        if not isinstance(bundle, dict):
            emit("FAIL", f"Bundle keys {label}", f"Expected dict, got {type(bundle).__name__}")
            return
        keys = sorted(str(key) for key in bundle)
        emit("PASS", f"Bundle keys {label}", "Loaded dictionary bundle", keys)
        required = ("rf_model", "xgb_model", "ridge_meta")
        present = all(key in bundle for key in required)
        emit("PASS" if present else "FAIL", f"Required estimators {label}", str(present), required)
        if not present:
            return
        try:
            rounds = bundle["xgb_model"].get_booster().num_boosted_rounds()
            emit("PASS", f"XGBoost booster {label}", f"num_boosted_rounds={rounds}")
        except Exception as exception:
            print_traceback("FAIL", f"XGBoost booster {label}", exception)
    except Exception as exception:
        print_traceback("FAIL", f"Inspect bundle {label}", exception)


def direct_load_tests(paths, loader) -> dict:
    results = {}
    for horizon in ("h1",):
        path = paths.get(horizon)
        result = {"direct_succeeded": False, "crlf_succeeded": None, "project_loader_succeeded": None}
        REPORT["models"].setdefault(f"stacking_{horizon}.pkl", {})["load_tests"] = result
        if path is None:
            exception = FileNotFoundError(f"No loader path available for {horizon}")
            print_traceback("FAIL", f"Direct joblib load {horizon}", exception)
            results[horizon] = result
            continue
        first_exception = None
        bundle = None
        try:
            import joblib

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                bundle = joblib.load(path)
            result["direct_succeeded"] = True
            result["direct_keys"] = sorted(str(key) for key in bundle) if isinstance(bundle, dict) else None
            emit("PASS", f"Direct joblib load {horizon}", "Loaded model bundle")
            for item in captured:
                warning = {
                    "category": item.category.__name__, "message": str(item.message),
                    "filename": item.filename, "lineno": item.lineno,
                }
                result.setdefault("warnings", []).append(warning)
                emit("WARN", f"Load warning {horizon}", f"{warning['category']}: {warning['message']}", f"{warning['filename']}:{warning['lineno']}")
            inspect_bundle(bundle, f"direct {horizon}")
        except Exception as exception:
            first_exception = exception
            result["direct_error"] = f"{type(exception).__name__}: {exception}"
            print_traceback("FAIL", f"Direct joblib load {horizon}", exception)
            try:
                import joblib

                data = path.read_bytes().replace(b"\r\n", b"\n")
                fallback_bundle = joblib.load(io.BytesIO(data))
                result["crlf_succeeded"] = True
                emit("PASS", f"In-memory CRLF retry {horizon}", "joblib.load(BytesIO(data with CRLF replaced)) succeeded")
                inspect_bundle(fallback_bundle, f"CRLF retry {horizon}")
            except Exception as retry_exception:
                result["crlf_succeeded"] = False
                result["crlf_error"] = f"{type(retry_exception).__name__}: {retry_exception}"
                print_traceback("FAIL", f"In-memory CRLF retry {horizon}", retry_exception)
        if first_exception is None and loader is not None and hasattr(loader, "load_model"):
            try:
                project_bundle = loader.load_model(horizon)
                result["project_loader_succeeded"] = True
                emit("PASS", f"Project load_model {horizon}", "Loader validation and deserialization succeeded")
                inspect_bundle(project_bundle, f"project loader {horizon}")
            except Exception as exception:
                result["project_loader_succeeded"] = False
                result["project_loader_error"] = f"{type(exception).__name__}: {exception}"
                print_traceback("FAIL", f"Project load_model {horizon}", exception)
                chain = []
                seen = set()
                current = exception
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    cause = current.__cause__
                    context = current.__context__
                    chain.append({
                        "type": type(current).__name__, "message": str(current),
                        "cause": f"{type(cause).__name__}: {cause}" if cause else None,
                        "context": f"{type(context).__name__}: {context}" if context else None,
                    })
                    current = cause or context
                result["exception_chain"] = chain
                emit("WARN", f"Exception chain {horizon}", "Included loader cause/context chain", chain)
        results[horizon] = result
    REPORT["load_tests"] = results
    return results


def make_conclusions(copies, paths, checksums, references, sync, load_results) -> None:
    conclusions = []
    for name, info in REPORT["models"].items():
        if name not in MODEL_NAMES:
            continue
        for copy in info.get("copies", []):
            if copy.get("header_status") == "FAIL":
                conclusions.append(f"{name}: invalid header ({copy.get('classification')}); restore the original model file.")
    for horizon, name in (("h1", MODEL_NAMES[0]),):
        reference = REPORT["models"].get(name, {}).get("reference_matches")
        if reference is False:
            conclusions.append(f"{name}: reference hash differs; download the original again from the Drive web page in a browser.")
        path = paths.get(horizon)
        if path and not REPORT["models"].get(name, {}).get("loader_path_matches_copy", False):
            conclusions.append(f"{name}: loader path differs from the measured copy.")
        load = load_results.get(horizon, {})
        if load.get("direct_succeeded") is False and load.get("crlf_succeeded") is True:
            conclusions.append(f"{name}: in-memory CRLF replacement loads; Git or another tool may have converted binary line endings; add *.pkl binary to .gitattributes and restore the original.")
        if sync.get(horizon):
            conclusions.append(f"{name}: sync-folder path or cloud placeholder detected; keep it on device or move it outside the synced folder.")
        reference_matches = REPORT["models"].get(name, {}).get("reference_matches")
        if reference_matches is True and load.get("direct_succeeded") is True and load.get("project_loader_succeeded") is False:
            conclusions.append(f"{name}: reference hash matches and direct load works, but project load_model fails; inspect the captured loader traceback and line.")
        direct_error = load.get("direct_error", "")
        if reference_matches is True and load.get("direct_succeeded") is False and "input stream corrupted" in direct_error.lower():
            conclusions.append(f"{name}: reference hash matches but direct load reports XGBoost input stream corrupted; the pickle does not deserialize here and needs repackaging (XGBoost JSON export) in Colab.")
    if not any(REPORT["models"].get(name, {}).get("reference_matches") is True for name in MODEL_NAMES):
        conclusions.append("No original Colab hash was supplied; local checksum agreement cannot prove the copy matches the training originals.")
    conclusive = any((
        "invalid header" in line
        or "reference hash differs" in line
        or "loader path differs" in line
        or "CRLF replacement loads" in line
        or "sync-folder path" in line
        or "reference hash matches" in line
    for line in conclusions))
    if not conclusive:
        conclusions.append("No cause is conclusive yet. Original-file SHA-256 hashes from Colab are still needed; compare them with the local loader paths.")
    REPORT["conclusions"] = conclusions
    for index, conclusion in enumerate(conclusions, 1):
        print(f"{index}. {conclusion}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-h1", help="SHA-256 of the original H1 file computed in Colab")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    print("Model load diagnosis (read only)")
    capture_environment()
    copies = find_model_copies()
    loader, paths = import_loader()
    checksums = checksum_checks(loader, paths, copies)
    reference_hash_checks(args, paths, checksums)
    sync = path_and_sync_checks(paths)
    git_and_line_ending_checks(paths)
    load_results = direct_load_tests(paths, loader)
    print("\nVerdict")
    make_conclusions(copies, paths, checksums, {"h1": args.expected_h1}, sync, load_results)
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(REPORT, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"\nJSON report: {REPORT_PATH}")
    except Exception as exception:
        emit("FAIL", "Write JSON report", f"{type(exception).__name__}: {exception}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())