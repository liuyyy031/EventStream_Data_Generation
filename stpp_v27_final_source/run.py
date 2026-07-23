#!/usr/bin/env python3
"""Readable standalone launcher for the validated, unmodified v27 source."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


BUNDLE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = BUNDLE_DIR / "runtime"
DATA_GENERATION_DIR = BUNDLE_DIR.parent
ENV_PATH = DATA_GENERATION_DIR / ".env"
MANIFEST_PATH = BUNDLE_DIR / "manifest.json"


def _check_source_bundle() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    failures = []
    for module, metadata in manifest["modules"].items():
        path = RUNTIME_DIR / metadata["relative_path"]
        if not path.is_file():
            failures.append(f"{module}: missing {path}")
            continue
        content = path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != metadata["sha256"]:
            failures.append(f"{module}: SHA-256 mismatch")
            continue
        compile(content, str(path), "exec", dont_inherit=True)
    if failures:
        raise RuntimeError("Source-bundle check failed:\n- " + "\n- ".join(failures))
    print(
        "STPP v27 final source check passed: "
        f"{len(manifest['modules'])} modules, "
        f"manifest={manifest['bundle_sha256']}"
    )
    return 0


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required") from exc
    if not ENV_PATH.is_file():
        raise RuntimeError(f"Expected .env file was not found: {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError(f"LLM_API_KEY is missing after loading {ENV_PATH}")


def _load_entrypoint():
    _load_env()
    sys.path.insert(0, str(RUNTIME_DIR))
    from demo_sts_stpp_v27 import main as v27_main
    return v27_main


def main() -> int:
    if "--check-source-bundle" in sys.argv:
        return _check_source_bundle()
    if "--check-env" in sys.argv:
        _load_env()
        print(f"Loaded environment file: {ENV_PATH}")
        print("LLM_API_KEY: set")
        print(
            "LLM_BASE_URL: "
            + (os.environ.get("LLM_BASE_URL") or "using llm_client default")
        )
        return 0
    if "--check-imports" in sys.argv:
        entrypoint = _load_entrypoint()
        print(f"STPP v27 final import check passed: {entrypoint.__module__}.main")
        return 0
    return int(_load_entrypoint()())


if __name__ == "__main__":
    raise SystemExit(main())
