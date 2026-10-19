"""Checkpoint this notebook's real-model/GDN evidence without credential files."""

import hashlib
import json
from pathlib import Path
import zipfile


def main():
    project = Path(__file__).resolve().parent
    target = project.parent / "ascend-gdn-agent-evidence-20261016.zip"
    root = project / "runs-20261016"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(root.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != ".lock":
                archive.write(p, str(p.relative_to(project)))
        for p in sorted(project.glob("*.log")):
            archive.write(p, p.name)
    print(
        "CHECKPOINT",
        target.name,
        target.stat().st_size,
        hashlib.sha256(target.read_bytes()).hexdigest(),
        flush=True,
    )
    for p in sorted(root.glob("*/result.json")):
        result = json.loads(p.read_text())
        best = p.parent / "candidates" / f"{result['best']}.json"
        config = json.loads(best.read_text())["config"]
        print(
            p.parent.name,
            result["status"],
            config,
            result["final_comparison"]["speedup_geomean"],
            result["independent_improvement_established"],
            flush=True,
        )


if __name__ == "__main__":
    main()
