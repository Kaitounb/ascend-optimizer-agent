"""Create a portable source + verified-evidence ZIP, without local caches or secrets."""

import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    target = root / "ascend-optimizer-agent.zip"
    manifest_path = root / "DELIVERY_MANIFEST.json"
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path in (target, manifest_path)
            or any(
                part in {".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache"}
                for part in relative.parts
            )
            or relative.parts[0].startswith("runs")
            or path.name in {".env", ".DS_Store", "api-key"}
            or path.suffix == ".pyc"
        ):
            continue
        files.append(path)
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest_path.write_text(
        json.dumps(
            {"version": "0.2.0", "files_sha256": manifest}, indent=2, ensure_ascii=False
        )
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, manifest_path]:
            archive.write(
                path, str(Path("ascend-optimizer-agent") / path.relative_to(root))
            )
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for name, sha in manifest.items():
            assert (
                hashlib.sha256(
                    archive.read("ascend-optimizer-agent/" + name)
                ).hexdigest()
                == sha
            )
    print(target)
    print(
        "files",
        len(files) + 1,
        "bytes",
        target.stat().st_size,
        "SHA256",
        hashlib.sha256(target.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()
