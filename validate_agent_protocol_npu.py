"""A deterministic local API fixture drives the actual agent + real NPU tools.

This is an integration test, NOT a real model experiment or a performance claim.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from ascend_optimizer.agent import run_agent
from ascend_optimizer.engine import Engine, atomic_json, initialize, workspace_lock
from ascend_optimizer.llm import Client
from ascend_optimizer.tasks import digest, render


def main():
    root = Path(__file__).resolve().parent / "runs" / "http_agent_npu"
    config = {"block": 4096, "unroll": 1}
    cid = digest(render("vector_add", config))[:16]
    calls = [
        (
            "propose_candidate",
            {
                "config": config,
                "reason": "Local API fixture tests the full tool path, not model intelligence",
            },
        ),
        *[
            (name, {"candidate_id": cid})
            for name in (
                "compile_kernel",
                "validate_kernel",
                "bench_kernel",
                "accept_candidate",
            )
        ],
    ]
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(observed)
            observed.append(
                {"tools": len(payload["tools"]), "messages": len(payload["messages"])}
            )
            if index < len(calls):
                name, arguments = calls[index]
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"fixture_{index}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            else:
                message = {
                    "role": "assistant",
                    "content": "本地 API 替身的工具调用流程完成，不代表真实模型推理。",
                }
            data = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with workspace_lock(root):
            if not (root / "task.json").exists():
                initialize(root, "vector_add")
            engine = Engine(root, timeout=300)
            try:
                client = Client(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="local-protocol-fixture",
                    key_env="ASCEND_PROTOCOL_FIXTURE_NO_KEY",
                )
                result = run_agent(engine, client, max_steps=8)
                assert result["status"] == "complete", result
                assert len(observed) == 6, observed
                atomic_json(
                    root / "protocol_check.json",
                    {
                        "passed": True,
                        "real_npu": True,
                        "model_is_fixture": True,
                        "requests": observed,
                    },
                )
                print("HTTP_AGENT_REAL_NPU_PASS", flush=True)
            finally:
                engine.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
