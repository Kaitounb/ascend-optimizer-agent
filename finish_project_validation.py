"""Wait for the selected serial run, then qualify and test the project export."""

from pathlib import Path
import subprocess
import sys
import time


def main():
    project = Path(__file__).resolve().parent
    pid = int(sys.argv[1])
    deadline = time.monotonic() + 3600
    command = Path(f"/proc/{pid}/cmdline")
    while command.exists() and b"validate_project_delivery.py" in command.read_bytes():
        if time.monotonic() > deadline:
            raise TimeoutError(
                "Previous serial run is still active; no competing NPU task started"
            )
        time.sleep(5)
    for script in (
        "qualify_project.py",
        "build_project_candidate.py",
        "validate_project_integration.py",
    ):
        print("SERIAL_STAGE", script, flush=True)
        subprocess.run([sys.executable, "-u", script], cwd=project, check=True)
        subprocess.run(
            [sys.executable, "export_project_evidence.py"], cwd=project, check=True
        )
    print("ALL_PROJECT_STAGES_FINISHED", flush=True)


if __name__ == "__main__":
    main()
