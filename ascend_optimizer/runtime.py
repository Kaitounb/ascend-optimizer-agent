"""Killable persistent worker transport. Keep LLM secrets out of the NPU process."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

PREFIX = b"@@ASCEND_RESULT@@"


class Worker:
    def __init__(self, root: Path, timeout=240):
        self.root, self.timeout = root, timeout
        self.process = None
        self.buffer = b""
        self.stderr = None
        self.secret_values = set()

    def exclude_secret(self, value):
        if value and value not in self.secret_values:
            self.secret_values.add(value)
            self.close()  # Recreate any existing worker with the stricter environment.

    def start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(
                word in k.upper()
                for word in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
            )
            and not any(secret in v for secret in self.secret_values)
        }
        env["PYTHONUNBUFFERED"] = "1"
        package_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = package_root
        self.stderr = (self.root / "worker.stderr.log").open("ab")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "ascend_optimizer.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            env=env,
            cwd=self.root,
            start_new_session=True,
            bufsize=0,
        )
        self.buffer = b""

    def call(self, request):
        self.start()
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                index = line.find(PREFIX)
                if index >= 0:
                    payload = json.loads(line[index + len(PREFIX) :])
                    if not payload.get("ok"):
                        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
                    return payload["result"]
                with (self.root / "worker.stdout.log").open("ab") as out:
                    out.write(line + b"\n")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError(
                    f"NPU worker exceeded {self.timeout}s; process group terminated; best candidate retained"
                )
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    code = self.process.poll()
                    self.close()
                    raise RuntimeError(
                        f"NPU worker exited ({code}); inspect worker.stderr.log"
                    )
                self.buffer += chunk
                if len(self.buffer) > 16 * 1024 * 1024:
                    self.close()
                    raise RuntimeError("Worker output exceeded 16 MiB without a result")

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait(timeout=3)
                except ProcessLookupError:
                    pass
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self.stderr:
            self.stderr.close()
            self.stderr = None
