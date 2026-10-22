"""Cache identity for model artifacts and inference dependencies."""

import hashlib
import os
import platform
from importlib import metadata
from pathlib import Path


def runtime_identity():
    packages = (
        "torch",
        "transformers",
        "accelerate",
        "qwen-vl-utils",
        "qwen-omni-utils",
        "flash-attn",
        "vllm",
        "lmdeploy",
        "av",
        "Pillow",
        "openai",
        "requests",
    )
    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def model_identity(model_path=None, revision=None):
    """Hash local artifacts once at run entry; remote revisions are declarations."""
    result = {"declared_revision": revision or os.getenv("OMNICOMPASS_MODEL_REVISION") or None}
    if not isinstance(model_path, (str, os.PathLike)):
        return result
    root = Path(model_path).expanduser()
    if not root.exists():
        return result
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    suffixes = {".safetensors", ".bin", ".pt", ".pth", ".gguf", ".json", ".model", ".jinja"}
    tokenizer_files = {"merges.txt", "vocab.txt", "tokenizer.tiktoken"}
    artifacts = {}
    for path in candidates:
        if not path.is_file() or (path.suffix not in suffixes and path.name not in tokenizer_files):
            continue
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        artifacts[path.name if root.is_file() else str(path.relative_to(root))] = digest.hexdigest()
    if not artifacts:
        raise ValueError("Local model path contains no recognized model/tokenizer artifacts")
    result["local_artifacts_sha256"] = artifacts
    return result
