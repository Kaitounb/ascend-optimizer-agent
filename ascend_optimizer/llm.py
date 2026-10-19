"""Dependency-free OpenAI-compatible Chat Completions and Responses adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError(
            "API redirect refused; configure the final endpoint explicitly"
        )


class Client:
    def __init__(
        self,
        *,
        base_url,
        model,
        api_mode="chat",
        key_env="OPENAI_API_KEY",
        key_file=None,
        timeout=120,
        request_options=None,
    ):
        parsed = urlsplit(base_url)
        if (
            parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not parsed.hostname
        ):
            raise ValueError(
                "API base URL must not contain credentials, query, or fragment"
            )
        if parsed.scheme != "https" and not (
            parsed.scheme == "http"
            and parsed.hostname in ("localhost", "127.0.0.1", "::1")
        ):
            raise ValueError("Use HTTPS, except for a loopback API server")
        if api_mode not in ("chat", "responses"):
            raise ValueError("API mode must be chat or responses")
        if not model:
            raise ValueError("Set --model or ASCEND_OPT_MODEL")
        self.key = os.environ.get(key_env, "")
        if not self.key and key_file:
            path = Path(key_file).expanduser()
            info = path.stat()
            if stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid():
                raise ValueError(
                    "API key file must be owned by this user and private (0600)"
                )
            self.key = path.read_text().strip()
        self.request_options = request_options or {}
        if set(self.request_options) - {
            "max_tokens",
            "max_output_tokens",
            "thinking",
            "reasoning_effort",
        }:
            raise ValueError("Unsupported request options")
        self.last_metadata = {}
        if not self.key and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"Set {key_env} in the agent process environment; do not put keys in task files"
            )
        self.url = base_url.rstrip("/") + (
            "/chat/completions" if api_mode == "chat" else "/responses"
        )
        self.model, self.api_mode, self.timeout = model, api_mode, timeout
        self.opener = build_opener(NoRedirect())

    def redact(self, text):
        return text.replace(self.key, "[REDACTED]") if self.key else text

    def post(self, payload):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        request = Request(self.url, json.dumps(payload).encode(), headers)
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    raw = response.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise RuntimeError("API response exceeds 8 MiB")
                    return json.loads(raw)
            except HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1 + attempt)
                    continue
                # Do not log response bodies that might echo request headers or credentials.
                raise RuntimeError(
                    f"Model API returned HTTP {exc.code}; check endpoint, model, access, or quotas"
                ) from None
            except URLError as exc:
                raise RuntimeError(
                    f"Model API connection failed: {self.redact(str(exc.reason))}"
                ) from None
        raise RuntimeError("API retry budget exhausted")

    def turn(self, history, schema, instructions):
        if self.api_mode == "chat":
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": instructions}, *history],
                "tools": [{"type": "function", "function": s} for s in schema],
                "tool_choice": "auto",
                "store": False,
            }
            payload.update(self.request_options)
            response = self.post(payload)
            self.last_metadata = {
                "returned_model": response.get("model"),
                "usage": response.get("usage"),
                "id": response.get("id"),
            }
            message = response["choices"][0]["message"]
            # Keep provider-specific reasoning fields when supplied, for compatible reasoning APIs.
            history.append(message)
            calls = [
                {
                    "id": c["id"],
                    "name": c["function"]["name"],
                    "arguments": c["function"]["arguments"],
                }
                for c in message.get("tool_calls", [])
            ]
            return calls, message.get("content") or ""
        payload = {
            "model": self.model,
            "instructions": instructions,
            "input": history,
            "tools": [{"type": "function", **s, "strict": False} for s in schema],
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        payload.update(self.request_options)
        response = self.post(payload)
        self.last_metadata = {
            "returned_model": response.get("model"),
            "usage": response.get("usage"),
            "id": response.get("id"),
        }
        output = response.get("output", [])
        history.extend(output)
        calls = [
            {
                "id": item["call_id"],
                "name": item["name"],
                "arguments": item["arguments"],
            }
            for item in output
            if item.get("type") == "function_call"
        ]
        text = "\n".join(
            part.get("text", "")
            for item in output
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        return calls, text

    def observation(self, history, call, value):
        content = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if self.api_mode == "chat":
            history.append(
                {"role": "tool", "tool_call_id": call["id"], "content": content}
            )
        else:
            history.append(
                {
                    "type": "function_call_output",
                    "call_id": call["id"],
                    "output": content,
                }
            )
