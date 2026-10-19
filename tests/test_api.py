"""Actual HTTP protocol round trips against a local deterministic model fixture."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ascend_optimizer.agent import run_agent
from ascend_optimizer.engine import Engine, initialize
from ascend_optimizer.llm import Client
from test_contract import FakeWorker


class ProtocolTests(unittest.TestCase):
    def exercise(self, mode):
        received = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initialize(root, "vector_add")
            engine = Engine(root, worker=FakeWorker())
            cid = engine.propose_candidate({"block": 1024, "unroll": 1}, "fixture")[
                "candidate_id"
            ]
            actions = [
                "compile_kernel",
                "validate_kernel",
                "bench_kernel",
                "accept_candidate",
            ]

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass

                def do_POST(self):
                    request = json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                    received.append(request)
                    index = len(received) - 1
                    if mode == "chat":
                        message = {"role": "assistant", "content": None}
                        if index < len(actions):
                            message["tool_calls"] = [
                                {
                                    "id": f"call_{index}",
                                    "type": "function",
                                    "function": {
                                        "name": actions[index],
                                        "arguments": json.dumps({"candidate_id": cid}),
                                    },
                                }
                            ]
                        else:
                            message["content"] = (
                                "Completed fixture, not a real model benchmark."
                            )
                        payload = {"choices": [{"message": message}]}
                    elif index < len(actions):
                        payload = {
                            "output": [
                                {
                                    "type": "function_call",
                                    "call_id": f"call_{index}",
                                    "name": actions[index],
                                    "arguments": json.dumps({"candidate_id": cid}),
                                }
                            ]
                        }
                    else:
                        payload = {
                            "output": [
                                {
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "Completed fixture.",
                                        }
                                    ],
                                }
                            ]
                        }
                    data = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with (
                    patch.dict(os.environ, {"TEST_AGENT_KEY": "fixture-secret"}),
                    patch("sys.stdout"),
                ):
                    client = Client(
                        base_url=f"http://127.0.0.1:{server.server_port}/v1",
                        model="fixture",
                        api_mode=mode,
                        key_env="TEST_AGENT_KEY",
                    )
                    result = run_agent(engine, client, max_steps=6)
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["best"], cid)
                self.assertEqual(len(received), 5)
                if mode == "chat":
                    observation = received[1]["messages"][-1]
                    self.assertEqual(observation["tool_call_id"], "call_0")
                    self.assertEqual(observation["role"], "tool")
                else:
                    observation = received[1]["input"][-1]
                    self.assertEqual(observation["call_id"], "call_0")
                    self.assertEqual(observation["type"], "function_call_output")
                for path in root.glob("*.json*"):
                    self.assertNotIn("fixture-secret", path.read_text())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_chat_protocol(self):
        self.exercise("chat")

    def test_responses_protocol(self):
        self.exercise("responses")

    def test_reject_http_nonloopback_and_credentials_in_url(self):
        for url in (
            "http://example.com/v1",
            "https://key@example.com/v1",
            "https://example.com/v1?key=secret",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Client(base_url=url, model="fixture")


if __name__ == "__main__":
    unittest.main()
