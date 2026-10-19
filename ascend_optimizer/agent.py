"""ReAct controller and offline search controller sharing exactly the same tools."""

from __future__ import annotations

import json

SYSTEM = """You optimize predefined operators on Huawei Ascend NPU using Triton-Ascend.
Follow measured evidence: get_task/get_environment, read_candidate(best), propose_candidate(config, reason),
compile_kernel -> validate_kernel -> bench_kernel -> accept_candidate. Inspect errors and try another configuration.
Use profile_kernel when diagnosis is useful. Every successfully benchmarked candidate must receive an accept_candidate decision.
Never invent timing, change task shapes/dtypes/epsilon, assume CUDA warps map to NPU cores, or claim HBM bandwidth saturation.
Config changes are rendered by trusted templates. You cannot run arbitrary source/shell or edit the evaluator.
The task contains search_configs as suggestions. You may propose other configurations within the tool schema.
Stop after a useful series of accepted/rejected hypotheses; finding no improvement is a legitimate outcome.
Return a short Chinese explanation of observed results and limitations; host generates the authoritative report and independent final comparison.
"""


def tool_schema(task):
    operator = task["operator"]
    if operator.startswith("gdn_"):
        config = {
            "type": "object",
            "properties": {
                "value_tile": {"type": "integer", "enum": [0, 8, 16, 32, 64, 128]},
                "unroll": {
                    "type": "boolean",
                    "description": "Statically unroll windows <=4, original loop otherwise",
                },
            },
            "required": ["value_tile"],
            "additionalProperties": False,
        }
    elif operator == "vector_add":
        config = {
            "type": "object",
            "properties": {
                "block": {
                    "type": "integer",
                    "enum": [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                },
                "unroll": {"type": "integer", "enum": [1, 2, 4, 8]},
            },
            "required": ["block", "unroll"],
            "additionalProperties": False,
        }
    else:
        config = {
            "type": "object",
            "properties": {
                "implementation": {"type": "string", "enum": ["torch_eager", "triton"]},
                "rows": {"type": "integer", "enum": [1, 2, 4, 8]},
            },
            "required": ["implementation", "rows"],
            "additionalProperties": False,
        }
    descriptions = {
        "get_task": "Read frozen contract, search suggestions and current best ID.",
        "get_environment": "Read actual NPU/CANN/compiler environment; no secrets.",
        "get_history": "Read up to thirty recent measurements, failures and candidate decisions.",
        "read_candidate": "Read an immutable candidate manifest and generated source.",
        "propose_candidate": "Create source from supported configuration and record your hypothesis.",
        "compile_kernel": "Compile and launch candidate on the real NPU; not just Python syntax validation.",
        "validate_kernel": "Compare against trusted CPU reference, check guards, NaNs and unchanged inputs.",
        "bench_kernel": "Revalidate both versions, measure ABBA/BAAB paired NPU events against current best.",
        "accept_candidate": "Deterministically decide promotion using a fresh benchmark for the exact candidate/current best.",
        "profile_kernel": "Collect actual CPU/NPU profiler trace; profiler timings are not acceptance evidence.",
    }
    schema = []
    for name, description in descriptions.items():
        if name in ("get_task", "get_environment", "get_history"):
            props = {}
        elif name == "propose_candidate":
            props = {"config": config, "reason": {"type": "string", "maxLength": 2000}}
        else:
            props = {"candidate_id": {"type": "string", "pattern": "^[0-9a-f]{16}$"}}
        schema.append(
            {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props),
                    "additionalProperties": False,
                },
            }
        )
    return schema


def search(engine, *, max_candidates=6, profile=False):
    engine.preflight()
    configs = engine.task["search_configs"]
    attempted = set()
    trajectory = engine.root / "trajectory.jsonl"
    history = (
        [json.loads(line) for line in trajectory.read_text().splitlines()]
        if trajectory.exists()
        else []
    )
    for row in history:
        if row["event"] == "candidate_decision":
            attempted.add(row["result"]["candidate_id"])
    count = 0
    for config in configs:
        if count >= max_candidates:
            break
        proposal = engine.call(
            "propose_candidate",
            {"config": config, "reason": "Search hypothesis: " + json.dumps(config)},
        )
        if not proposal["ok"]:
            continue
        cid = proposal["result"]["candidate_id"]
        if cid == engine.state["best"] or cid in attempted:
            continue
        count += 1
        for name in (
            "compile_kernel",
            "validate_kernel",
            "bench_kernel",
            "accept_candidate",
        ):
            result = engine.call(name, {"candidate_id": cid})
            if not result["ok"]:
                break
    return engine.finalize("search", "complete", profile=profile)


def run_agent(engine, client, *, max_steps=24, max_tools=64, profile=False):
    if hasattr(engine.worker, "exclude_secret"):
        engine.worker.exclude_secret(client.key)
    engine.preflight()
    task = engine.get_task()
    history = [
        {
            "role": "user",
            "content": "Optimize this frozen task using the tools.\n"
            + json.dumps(task, ensure_ascii=False)
            + "\nPrevious evidence:\n"
            + json.dumps(engine.get_history(), ensure_ascii=False),
        }
    ]
    calls_used, early_stops = 0, 0
    pending = set()
    status, final_text = "budget_exhausted", ""
    transcript = engine.root / f"conversation-{engine.session}.jsonl"
    for step in range(max_steps):
        try:
            calls, text = client.turn(history, tool_schema(engine.task), SYSTEM)
            with transcript.open("a") as stream:
                stream.write(
                    client.redact(
                        json.dumps(
                            {
                                "step": step,
                                "calls": calls,
                                "text": text,
                                "api": getattr(client, "last_metadata", {}),
                            },
                            ensure_ascii=False,
                        )
                    )
                    + "\n"
                )
            if not calls:
                if (
                    engine.decisions >= engine.task.get("min_decisions", 1)
                    and not pending
                ):
                    status, final_text = "complete", text
                    break
                early_stops += 1
                if early_stops > 2:
                    status, final_text = "incomplete_early_stop", text
                    break
                history.append(
                    {
                        "role": "user",
                        "content": f"Need {engine.task.get('min_decisions', 1)} completed decisions; have {engine.decisions}, pending {sorted(pending)}. Try other suggested configurations. Use compile -> validate -> bench -> accept; prose alone is not completion.",
                    }
                )
                continue
            for call in calls:
                if calls_used >= max_tools:
                    observation = {"ok": False, "error": "Tool budget exhausted"}
                else:
                    calls_used += 1
                    try:
                        arguments = (
                            json.loads(call["arguments"])
                            if isinstance(call["arguments"], str)
                            else call["arguments"]
                        )
                    except (TypeError, json.JSONDecodeError):
                        observation = {
                            "ok": False,
                            "error": "Malformed tool arguments; supply a JSON object",
                        }
                    else:
                        observation = engine.call(call["name"], arguments)
                        if observation["ok"] and call["name"] == "bench_kernel":
                            pending.add(arguments["candidate_id"])
                        if observation["ok"] and call["name"] == "accept_candidate":
                            pending.discard(arguments["candidate_id"])
                client.observation(history, call, observation)
                with transcript.open("a") as stream:
                    stream.write(
                        client.redact(
                            json.dumps(
                                {
                                    "step": step,
                                    "tool": call["name"],
                                    "observation": observation,
                                },
                                ensure_ascii=False,
                            )
                        )
                        + "\n"
                    )
            if calls_used >= max_tools:
                break
        except Exception as exc:
            status = "api_error"
            final_text = client.redact(str(exc))
            engine.event("model_error", {"error": final_text})
            break
    # Finish pending decisions using the same authoritative path. Stale evidence remains an explicit error.
    for cid in sorted(pending):
        decision = engine.call("accept_candidate", {"candidate_id": cid})
        if not decision["ok"]:
            status = "incomplete_pending_candidate"
    return engine.finalize(
        client.api_mode + ":" + client.model,
        status,
        model_text=client.redact(final_text),
        profile=profile,
    )
