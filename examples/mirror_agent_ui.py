"""Browser UI for the AMOS Mirror Agent demo.

Run with:

    PYTHONPATH=src python examples/mirror_agent_ui.py --host 127.0.0.1 --port 8787

The UI uses local Codex for conversational self-awareness and explanation by
default. Deterministic AMOS maintenance remains non-LLM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping

from amos import Amos

try:
    from examples.mirror_agent_demo import AGENT_ID, SCOPE, USER_ID, MirrorAgentDemo
except ModuleNotFoundError:  # direct execution with PYTHONPATH=src
    from mirror_agent_demo import AGENT_ID, SCOPE, USER_ID, MirrorAgentDemo


REPO_ROOT = Path(__file__).resolve().parents[1]


class CodexLLMClient:
    provider_name = "local_codex"

    def __init__(self, *, timeout_seconds: int = 90):
        self.timeout_seconds = timeout_seconds
        self.codex_path = shutil.which("codex")

    def available(self) -> bool:
        return self.codex_path is not None

    def generate(self, prompt: str) -> str:
        if not self.codex_path:
            raise RuntimeError("codex executable was not found on PATH")
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as output:
            output_path = Path(output.name)
        try:
            command = [
                self.codex_path,
                "exec",
                "--cd",
                str(REPO_ROOT),
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--output-last-message",
                str(output_path),
                "-",
            ]
            subprocess.run(
                command,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=True,
            )
            answer = output_path.read_text(encoding="utf-8").strip()
            if not answer:
                raise RuntimeError("codex produced an empty answer")
            return answer
        finally:
            output_path.unlink(missing_ok=True)


class OfflineLLMClient:
    provider_name = "offline_test_renderer"

    def available(self) -> bool:
        return True

    def generate(self, prompt: str) -> str:
        context = json.loads(prompt.split("<AMOS_CONTEXT>", 1)[1].split("</AMOS_CONTEXT>", 1)[0])
        refs = ", ".join(item["atom_ref"] for item in context["packet_items"][:4])
        if not refs:
            refs = "no high-relevance memory refs"
        return (
            "Offline renderer: AMOS retrieved bounded self-model and task memory "
            f"before answering. Relevant memory refs: {refs}. "
            "Maintenance remains deterministic and non-LLM."
        )


class MirrorAgentUIState:
    def __init__(self, db_path: Path, *, lm_mode: str = "codex"):
        self.db_path = db_path
        self.amos = Amos(db_path)
        self.demo = MirrorAgentDemo(self.amos, db_path=str(db_path))
        self.report = self.demo.run()
        self.lock = threading.RLock()
        self.lm_client = OfflineLLMClient() if lm_mode == "offline" else CodexLLMClient()
        if lm_mode == "codex" and not self.lm_client.available():
            self.lm_client = OfflineLLMClient()
        self.lm_mode = self.lm_client.provider_name

    def close(self) -> None:
        self.amos.close()

    def provider_status(self) -> dict[str, Any]:
        return {
            "provider": self.lm_mode,
            "available": self.lm_client.available(),
            "chat_lm_backed": self.lm_mode == "local_codex",
            "maintenance_uses_llm": False,
        }

    def current_report(self) -> dict[str, Any]:
        self.report = self.demo.report()
        self.report["lm"] = self.provider_status()
        return self.report

    def compile_reasoning(self, need: str) -> dict[str, Any]:
        need = need.strip()
        if not need:
            raise ValueError("reasoning need is required")
        frame = self.amos.compile_memory_frame(
            need=need,
            purpose=(
                "render a grounded Mirror Agent response using coherent history, "
                "active conclusions, constraints, conflicts, and commitments"
            ),
            depth="working_frame",
            task_context={
                "human_id": USER_ID,
                "project_id": "amos",
                "project_thread_id": "mirror-agent-demo",
                "phase": "interactive_chat",
            },
            scope=SCOPE,
            requester="reasoner",
            target_processor="reasoner",
            token_or_byte_budget={"tokens": 1800},
            run_policy=False,
        )
        self.demo.reasoning_frames["interactive_chat"] = frame
        self.demo.loaded_reasoning_pages.pop("interactive_chat", None)
        return frame

    def _load_reasoning_page(self, page_id: str) -> dict[str, Any]:
        frame = self.demo.reasoning_frames.get("interactive_chat")
        if not frame:
            raise ValueError("compile a reasoning frame before loading a page")
        descriptor = next(
            (
                item
                for item in frame.get("page_index", [])
                if item.get("page_id") == page_id
            ),
            None,
        )
        if descriptor is None:
            raise ValueError("unknown reasoning page id")
        loaded = self.amos.load_memory_page(
            frame_id=frame["frame_id"],
            revision=frame["revision"],
            page=descriptor,
            need="load selected coherent memory detail",
            purpose="inspect the trusted continuation for the active reasoning frame",
            depth="supporting",
            scope=SCOPE,
            requester="reasoner",
            target_processor="reasoner",
            token_or_byte_budget={"tokens": 1600},
            run_policy=False,
        )
        self.demo.loaded_reasoning_pages["interactive_chat"] = loaded
        return loaded

    def load_reasoning_page(self, page_id: str) -> dict[str, Any]:
        loaded = self._load_reasoning_page(page_id.strip())
        return {
            "page": loaded,
            "report": self.current_report(),
        }

    def chat(self, message: str) -> dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("message is required")
        turn_ref = f"ui/chat/{self.amos.store.graph_version() + 1}"
        evidence = self.demo.capture(
            "user_message",
            turn_ref,
            {"text": message},
        )
        frame = self.compile_reasoning(message)
        loaded_page: dict[str, Any] = {}
        if frame.get("page_index"):
            loaded_page = self._load_reasoning_page(
                str(frame["page_index"][0]["page_id"])
            )
        exact = self.amos.retrieve_atom(
            "mirror_reasoning_design_current",
            scope=SCOPE,
            requester="reasoner",
            target_processor="reasoner",
            include_conflicts=True,
            include_low_health=True,
            include_superseded=True,
            run_policy=False,
        )
        self.demo.exact_lookups["interactive_chat"] = exact
        packet = self.amos.retrieve_packet(
            cues=[message],
            scope=SCOPE,
            requester="reasoner",
            target_processor="reasoner",
            include_conflicts=True,
            include_low_health=True,
            max_items=8,
            # Interactive reads must not synchronously rebuild every derived
            # index. The service/background worker and explicit maintenance
            # endpoint own that work.
            run_policy=False,
        )
        self_view = self.amos.retrieve_self_awareness(agent_id=AGENT_ID, scope=SCOPE)
        recall = self.amos.retrieve_agentic_recall(
            agent_id=AGENT_ID,
            cues=[message],
            scope=SCOPE,
            target_processor="reasoner",
        )
        capacity = self.amos.health_capacity()
        prompt = build_lm_prompt(
            message=message,
            packet=packet,
            self_view=self_view,
            recall=recall,
            capacity=capacity,
            frame=frame,
            loaded_page=loaded_page,
        )
        answer = self.lm_client.generate(prompt)
        retrieved_refs = [item["atom_ref"] for item in packet["items"]]
        cited_refs = used_memory_refs(answer, packet)
        outcome: dict[str, Any] = {
            "label": "useful" if cited_refs else "observed",
            "use_status": "materially_used" if cited_refs else "context_only",
            "user_message_ref": evidence["evidence_id"],
            "lm_provider": self.lm_mode,
        }
        if cited_refs:
            outcome["used_item_refs"] = cited_refs
        feedback = self.amos.record_retrieval_outcome(
            packet_id=packet["packet_id"],
            request=packet["request"],
            outcome=outcome,
        )
        self.demo.retrieval_feedback.append(feedback)
        self.amos.record_agentic_trace(
            agent_id=AGENT_ID,
            task="mirror agent chat",
            action="rendered LM-backed self-aware answer",
            outcome="success",
            lesson=(
                "bounded AMOS context should drive conversational explanation; "
                f"turn={turn_ref}; packet={packet['packet_id']}; evidence={evidence['evidence_id']}"
            ),
            scope=SCOPE,
            actor="reasoner",
        )
        turn = {
            "scenario": "interactive_chat",
            "user": message,
            "agent": answer,
            "memory_packet_id": packet["packet_id"],
            "reasoning_frame_id": frame["frame_id"],
            "loaded_page_id": loaded_page.get("page_id"),
            "retrieved_memory_refs": retrieved_refs,
            "cited_memory_refs": cited_refs,
            "lm_provider": self.lm_mode,
        }
        self.demo.chat.append(turn)
        self.demo.packets["interactive_chat"] = packet
        self.demo.service_views["reasoner"] = {
            "graph_version": packet["graph_version"],
            "packet_id": packet["packet_id"],
            "retrieved_item_refs": retrieved_refs,
            "materially_used_item_refs": cited_refs,
            "frame_id": frame["frame_id"],
            "loaded_page_id": loaded_page.get("page_id"),
            "lm_provider": self.lm_mode,
        }
        return {
            "turn": turn,
            "packet": packet,
            "self_awareness": self_view,
            "agentic_recall": recall,
            "reasoning_frame": frame,
            "loaded_reasoning_page": loaded_page,
            "exact_lookup": exact,
            "retrieval_feedback": feedback,
            "capacity": capacity,
            "lm": self.provider_status(),
            "report": self.current_report(),
        }

    def run_maintenance(self) -> dict[str, Any]:
        policy = self.amos.run_memory_policy(
            force=True,
            trigger="ui_maintenance_button",
            scope=SCOPE,
            actor="svc:memory_policy",
        )
        results = policy.get("results", {})
        smp = results.get("smp", {"status": "skipped", "outputs": []})
        steward = results.get("steward", {"status": "skipped", "actions": []})
        distiller = results.get(
            "maintenance_distiller",
            {"status": "skipped", "proposals": [], "committed": [], "deferred": []},
        )
        distillation = results.get("distillation", {"status": "skipped"})
        index = results.get("index", {"status": "skipped", "indexes": []})
        cache = results.get("packet_cache", {"status": "skipped"})
        self.demo.maintenance = {
            "policy": policy,
            "smp": smp,
            "steward": steward,
            "distillation": distillation,
            "maintenance_distiller": distiller,
            "index": index,
            "packet_cache": cache,
            "lm_used": False,
        }
        return {
            "policy": policy,
            "smp": smp,
            "steward": steward,
            "distillation": distillation,
            "maintenance_distiller": distiller,
            "index": index,
            "packet_cache": cache,
            "lm_used": False,
            "report": self.current_report(),
        }


def build_lm_prompt(
    *,
    message: str,
    packet: Mapping[str, Any],
    self_view: Mapping[str, Any],
    recall: Mapping[str, Any],
    capacity: Mapping[str, Any],
    frame: Mapping[str, Any],
    loaded_page: Mapping[str, Any],
) -> str:
    context = {
        "user_message": message,
        "agent_id": AGENT_ID,
        "instruction": (
            "Answer as the Amos Mirror Agent. Do not claim consciousness or sentience. "
            "Explain operational self-awareness using only the AMOS context below. "
            "Cite an exact atom ref only when it materially shapes the answer; do not "
            "cite merely exposed context. Keep the answer concise and human-friendly."
        ),
        "packet_id": packet["packet_id"],
        "packet_items": [packet_item_summary(item) for item in packet["items"][:10]],
        "packet_omissions": packet.get("omissions", [])[:12],
        "reasoning_frame": reasoning_frame_summary(frame),
        "loaded_reasoning_page": reasoning_page_summary(loaded_page),
        "self_model": {
            "capabilities": [packet_item_summary(item) for item in self_view["capabilities"]],
            "limitations": [packet_item_summary(item) for item in self_view["limitations"]],
            "open_commitments": [
                packet_item_summary(item) for item in self_view["open_commitments"]
            ],
            "runtime_state": self_view.get("runtime_state"),
            "calibration": self_view.get("calibration"),
        },
        "agentic_recall": {
            "successes": [packet_item_summary(item) for item in recall["successes"][:5]],
            "failures": [packet_item_summary(item) for item in recall["failures"][:5]],
            "blocked": [packet_item_summary(item) for item in recall["blocked"][:5]],
            "corrections": [packet_item_summary(item) for item in recall["corrections"][:5]],
            "external_constraints": recall["external_constraints"],
        },
        "capacity": capacity,
        "maintenance_policy": {
            "routine_maintenance_uses_llm": False,
            "smp": "deterministic non-generative",
            "distiller": "processor-pack proposals with policy-gated low-risk commits",
            "reviewer_authority": "draft_only",
        },
    }
    return (
        "You are a local Codex LM used only for Mirror Agent conversational explanation.\n"
        "Do not edit files. Do not run commands. Return only the answer text.\n"
        "<AMOS_CONTEXT>\n"
        + json.dumps(context, indent=2, sort_keys=True)
        + "\n</AMOS_CONTEXT>\n"
    )


def packet_item_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", {})
    return {
        "atom_ref": item.get("atom_ref"),
        "type": item.get("type"),
        "score": item.get("score"),
        "health_status": item.get("health_status"),
        "evidence_refs": item.get("evidence_refs", []),
        "text": payload.get("claim")
        or payload.get("description")
        or payload.get("name")
        or payload.get("limitation")
        or payload.get("capability")
        or payload.get("desired_state")
        or payload.get("promised_action")
        or json.dumps(payload, sort_keys=True)[:240],
    }


def reasoning_frame_summary(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": frame.get("frame_id"),
        "revision": frame.get("revision", {}),
        "truncated": frame.get("truncated", False),
        "unknowns": list(frame.get("unknowns", []))[:8],
        "units": [
            {
                "unit_id": unit.get("unit_id"),
                "unit_type": unit.get("unit_type"),
                "title": unit.get("title"),
                "summary": unit.get("summary"),
                "active_conclusion_refs": unit.get("active_conclusion_refs", []),
                "constraint_refs": unit.get("constraint_refs", []),
                "commitment_refs": unit.get("commitment_refs", []),
                "conflict_refs": unit.get("conflict_refs", []),
                "source_atom_refs": unit.get("source_atom_refs", []),
                "compression": unit.get("compression", {}),
            }
            for unit in frame.get("units", [])[:8]
        ],
        "page_index": [
            {
                "page_id": item.get("page_id"),
                "title": item.get("title"),
                "summary": item.get("summary"),
                "relevance": item.get("relevance"),
                "focus_atom_refs": item.get("focus_atom_refs", []),
                "source_atom_refs": item.get("source_atom_refs", []),
            }
            for item in frame.get("page_index", [])[:12]
        ],
    }


def reasoning_page_summary(page: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "page_id": page.get("page_id"),
        "status": page.get("status"),
        "depth": page.get("depth"),
        "source_atom_refs": page.get("source_atom_refs", []),
        "active_conclusion_refs": page.get("active_conclusion_refs", []),
        "constraint_refs": page.get("constraint_refs", []),
        "commitment_refs": page.get("commitment_refs", []),
        "conflict_refs": page.get("conflict_refs", []),
        "truncated": page.get("truncated", False),
    }


def used_memory_refs(answer: str, packet: Mapping[str, Any]) -> list[str]:
    """Return only packet atoms explicitly cited by the rendered answer."""

    return [
        str(item["atom_ref"])
        for item in packet.get("items", [])
        if item.get("atom_ref") and str(item["atom_ref"]) in answer
    ]


class MirrorAgentUIServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        db_path: Path,
        *,
        lm_mode: str = "codex",
    ):
        self.state = MirrorAgentUIState(db_path, lm_mode=lm_mode)
        super().__init__(server_address, make_handler())

    def server_close(self) -> None:
        try:
            self.state.close()
        finally:
            super().server_close()


def make_handler() -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AmosMirrorUI/1.0"

        def do_GET(self) -> None:
            self._handle("GET")

        def do_POST(self) -> None:
            self._handle("POST")

        def log_message(self, _format: str, *args: Any) -> None:
            return

        def _handle(self, method: str) -> None:
            try:
                state = self.server.state  # type: ignore[attr-defined]
                if method == "GET" and self.path.split("?", 1)[0] == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                if method == "GET" and self.path.split("?", 1)[0] == "/":
                    return self._write_html(INDEX_HTML)
                if method == "GET" and self.path.split("?", 1)[0] == "/api/report":
                    with state.lock:
                        return self._write_json(state.current_report())
                if method == "GET" and self.path.split("?", 1)[0] == "/api/status":
                    return self._write_json(state.provider_status())
                if method == "POST" and self.path.split("?", 1)[0] == "/api/chat":
                    body = self._read_json()
                    with state.lock:
                        return self._write_json(state.chat(str(body.get("message", ""))))
                if (
                    method == "POST"
                    and self.path.split("?", 1)[0] == "/api/reasoning/compile"
                ):
                    body = self._read_json()
                    with state.lock:
                        frame = state.compile_reasoning(str(body.get("need", "")))
                        return self._write_json(
                            {
                                "frame": frame,
                                "report": state.current_report(),
                            }
                        )
                if (
                    method == "POST"
                    and self.path.split("?", 1)[0] == "/api/reasoning/page"
                ):
                    body = self._read_json()
                    with state.lock:
                        return self._write_json(
                            state.load_reasoning_page(
                                str(body.get("page_id", ""))
                            )
                        )
                if method == "POST" and self.path.split("?", 1)[0] == "/api/maintenance/run":
                    with state.lock:
                        return self._write_json(state.run_maintenance())
                self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            except Exception as exc:  # pragma: no cover - exercised via UI smoke
                self._write_json(
                    {"error": str(exc), "type": type(exc).__name__},
                    status=HTTPStatus.BAD_REQUEST,
                )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            if not raw:
                return {}
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("expected JSON object")
            return data

        def _write_html(self, html: str) -> None:
            raw = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _write_json(
            self, payload: Mapping[str, Any], *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMOS Mirror Agent</title>
  <style>
    :root {
      --bg: #f7f8f7;
      --surface: #ffffff;
      --surface-2: #f1f4f2;
      --text: #18201d;
      --muted: #68736f;
      --border: #dce3df;
      --accent: #157f6e;
      --accent-2: #0d5d51;
      --amber: #b7791f;
      --red: #ba3a35;
      --green: #15803d;
      --shadow: 0 10px 28px rgba(19, 31, 27, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input { font: inherit; }
    .app { display: grid; grid-template-columns: 236px 1fr; min-height: 100vh; }
    .sidebar {
      background: #17211e;
      color: #e8eeee;
      padding: 18px 14px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }
    .brand { display: flex; align-items: center; gap: 10px; padding: 5px 6px 14px; border-bottom: 1px solid rgba(255,255,255,0.12); }
    .mark { width: 30px; height: 30px; border-radius: 6px; background: linear-gradient(135deg, #34d399, #0f766e); display: grid; place-items: center; font-weight: 800; color: #06231d; }
    .brand strong { display: block; font-size: 14px; }
    .brand span { display: block; color: #a8b7b2; font-size: 12px; margin-top: 2px; }
    .nav { display: grid; gap: 5px; }
    .nav button {
      border: 0;
      color: #cbd8d4;
      background: transparent;
      display: flex;
      align-items: center;
      gap: 9px;
      padding: 10px 10px;
      border-radius: 6px;
      cursor: pointer;
      text-align: left;
      font-size: 13px;
    }
    .nav button.active, .nav button:hover { background: rgba(255,255,255,0.09); color: #fff; }
    .dot { width: 7px; height: 7px; border-radius: 50%; background: #7f918b; }
    .nav button.active .dot { background: #34d399; }
    .main { min-width: 0; display: grid; grid-template-rows: auto 1fr; }
    .topbar {
      height: 58px;
      background: rgba(255,255,255,0.82);
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 22px;
      position: sticky;
      top: 0;
      z-index: 2;
      backdrop-filter: blur(12px);
    }
    .topbar h1 { font-size: 17px; margin: 0; font-weight: 700; }
    .status { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
    .chip {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
      white-space: nowrap;
    }
    .chip.good { color: var(--green); border-color: #b9dec8; background: #f0faf4; }
    .chip.warn { color: var(--amber); border-color: #ead4a8; background: #fffaf0; }
    .chip.bad { color: var(--red); border-color: #efc0bd; background: #fff5f4; }
    .content { padding: 18px 22px 28px; overflow: auto; }
    .view { display: none; }
    .view.active { display: block; }
    .grid { display: grid; gap: 14px; }
    .chat-grid { grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr); }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .panel-head {
      min-height: 46px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .panel-head h2 { margin: 0; font-size: 14px; }
    .panel-body { padding: 13px 14px; }
    .transcript { display: grid; gap: 12px; max-height: calc(100vh - 220px); overflow: auto; padding-right: 4px; }
    .turn { border: 1px solid var(--border); border-radius: 8px; padding: 12px; background: #fcfdfc; }
    .turn .user { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
    .turn .agent { font-size: 14px; line-height: 1.45; }
    .refs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .ref { font-size: 11px; color: var(--accent-2); background: #eef8f5; border: 1px solid #cce9e2; border-radius: 6px; padding: 3px 6px; }
    .composer { display: grid; grid-template-columns: 1fr auto; gap: 8px; margin-top: 12px; }
    input {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 11px 12px;
      background: #fff;
      color: var(--text);
      min-width: 0;
    }
    input:disabled { background: #f4f7f5; color: var(--muted); }
    .primary {
      border: 0;
      background: var(--accent);
      color: #fff;
      border-radius: 6px;
      padding: 10px 13px;
      cursor: pointer;
      font-weight: 650;
      font-size: 13px;
    }
    .icon-button {
      width: 50px;
      height: 45px;
      min-width: 50px;
      padding: 0;
      display: inline-grid;
      place-items: center;
    }
    .icon-button:disabled {
      cursor: wait;
      background: var(--accent-2);
      opacity: 0.86;
    }
    .submit-icon {
      width: 19px;
      height: 19px;
      display: block;
    }
    .submit-icon path {
      fill: none;
      stroke: currentColor;
      stroke-width: 2.3;
      stroke-linecap: round;
      stroke-linejoin: round;
    }
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255,255,255,0.4);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: block;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    .secondary {
      border: 1px solid var(--border);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12px;
    }
    .rows { display: grid; gap: 8px; }
    .row {
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #fff;
      padding: 10px;
      display: grid;
      gap: 6px;
    }
    .row-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .row strong { font-size: 13px; overflow-wrap: anywhere; }
    .meta { color: var(--muted); font-size: 12px; line-height: 1.4; overflow-wrap: anywhere; }
    .score { color: var(--accent-2); font-size: 12px; font-weight: 700; }
    .kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }
    .kpi { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
    .kpi span { color: var(--muted); font-size: 12px; display: block; }
    .kpi strong { font-size: 20px; display: block; margin-top: 6px; }
    .two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .meter { height: 10px; border-radius: 99px; background: #e8eeeb; overflow: hidden; }
    .meter div { height: 100%; background: var(--red); width: 100%; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #101715;
      color: #d9efe8;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.5;
      max-height: 520px;
      overflow: auto;
    }
    @media (max-width: 980px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; }
      .nav { grid-template-columns: repeat(4, minmax(0, 1fr)); }
      .chat-grid, .two, .three { grid-template-columns: 1fr; }
      .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 640px) {
      .topbar {
        height: auto;
        min-height: 0;
        align-items: flex-start;
        flex-direction: column;
        gap: 10px;
        padding: 14px 22px;
        position: static;
      }
      .status {
        justify-content: flex-start;
        width: 100%;
      }
      .chip { white-space: normal; }
      .content { padding-top: 18px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand"><div class="mark">A</div><div><strong>AMOS Mirror Agent</strong><span>Operational self-awareness</span></div></div>
      <nav class="nav" id="nav"></nav>
    </aside>
    <main class="main">
      <header class="topbar">
        <h1 id="title">Chat</h1>
        <div class="status" id="status"></div>
      </header>
      <section class="content" id="content"></section>
    </main>
  </div>
  <script>
    const tabs = ["Chat", "Self Model", "Memory Packet", "Reasoning", "Evidence", "Maintenance", "Capacity", "Graph"];
    let state = null;
    let active = "Chat";
    let scrollTranscriptAfterRender = false;
    let chatPending = false;
    let pendingMessage = "";

    const $ = (id) => document.getElementById(id);
    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    const refs = (items) => (items || []).map(ref => `<span class="ref">${esc(ref)}</span>`).join("");
    const atomText = (item) => {
      const p = item.payload || {};
      return p.claim || p.description || p.name || p.limitation || p.capability || p.desired_state || p.promised_action || item.label || item.atom_ref || item.id;
    };

    async function refreshReport() {
      const response = await fetch("/api/report");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "report refresh failed");
      state = data;
      return state;
    }

    function initNav() {
      $("nav").innerHTML = tabs.map(tab => `<button class="${tab === active ? "active" : ""}" data-tab="${tab}"><span class="dot"></span>${tab}</button>`).join("");
      document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", async () => {
        active = button.dataset.tab;
        try {
          await refreshReport();
        } catch (error) {
          alert(error.message || "report refresh failed");
        }
        render();
      }));
    }

    async function load() {
      await refreshReport();
      initNav();
      render();
    }

    function renderStatus() {
      const memory = state.verification.memory;
      const capacity = state.capacity.health;
      const lm = state.lm || {};
      $("status").innerHTML = [
        `<span class="chip">graph v${memory.graph_version}</span>`,
        `<span class="chip ${lm.chat_lm_backed ? "good" : "warn"}">LM ${esc(lm.provider || "unknown")}</span>`,
        `<span class="chip good">maintenance non-LLM</span>`,
        `<span class="chip ${capacity.pressure_mode === "red" ? "bad" : "good"}">capacity ${capacity.pressure_mode}</span>`,
        `<span class="chip">SQLite</span>`
      ].join("");
    }

    function render() {
      if (!state) return;
      initNav();
      $("title").textContent = active;
      renderStatus();
      const content = $("content");
      if (active === "Chat") content.innerHTML = viewChat();
      if (active === "Self Model") content.innerHTML = viewSelf();
      if (active === "Memory Packet") content.innerHTML = viewPacket(state.memory_packet);
      if (active === "Reasoning") content.innerHTML = viewReasoning();
      if (active === "Evidence") content.innerHTML = viewEvidence();
      if (active === "Maintenance") content.innerHTML = viewMaintenance();
      if (active === "Capacity") content.innerHTML = viewCapacity();
      if (active === "Graph") content.innerHTML = viewGraph();
      bindActions();
      const transcript = document.querySelector(".transcript");
      if (transcript && scrollTranscriptAfterRender) transcript.scrollTop = transcript.scrollHeight;
      scrollTranscriptAfterRender = false;
    }

    function viewChat() {
      const packet = state.memory_packet || {};
      const packetSource = state.memory_packet_source ? `<span class="chip">${esc(state.memory_packet_source)}</span>` : "";
      const messageValue = chatPending ? pendingMessage : "";
      const submitContent = chatPending ? `<span class="spinner" aria-hidden="true"></span>` : `<svg class="submit-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h13"></path><path d="m13 6 6 6-6 6"></path></svg>`;
      return `<div class="grid chat-grid">
        <section class="panel">
          <div class="panel-head"><h2>Conversation</h2><span class="chip ${state.lm.chat_lm_backed ? "good" : "warn"}">${state.lm.chat_lm_backed ? "LM-backed" : "offline fallback"}</span></div>
          <div class="panel-body">
            <div class="transcript">${state.chat.map(turn => `<div class="turn"><div class="user">${esc(turn.user)}</div><div class="agent">${esc(turn.agent)}</div><div class="refs">${refs(turn.cited_memory_refs || [turn.memory_packet_id])}</div></div>`).join("")}</div>
            <form class="composer" id="chat-form" aria-busy="${chatPending ? "true" : "false"}">
              <input id="message" value="${esc(messageValue)}" placeholder="Message" aria-label="Message" ${chatPending ? "disabled" : ""}>
              <button class="primary icon-button" type="submit" aria-label="${chatPending ? "Waiting for response" : "Submit prompt"}" title="${chatPending ? "Waiting for response" : "Submit prompt"}" ${chatPending ? "disabled" : ""}>${submitContent}</button>
            </form>
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><h2>Retrieved Memory</h2><div class="refs">${packetSource}<span class="chip">${esc(packet.packet_id)}</span></div></div>
          <div class="panel-body">${packetRows(packet, 8)}</div>
        </section>
      </div>`;
    }

    function viewSelf() {
      const canonical = state.current_self_model.canonical_self_atoms || [];
      const open = state.current_self_model.open_commitments || [];
      return `<div class="kpi-grid">
        <div class="kpi"><span>Self atoms</span><strong>${canonical.length}</strong></div>
        <div class="kpi"><span>Capabilities</span><strong>${canonical.filter(a => a.type === "capability").length}</strong></div>
        <div class="kpi"><span>Limitations</span><strong>${canonical.filter(a => a.type === "limitation").length}</strong></div>
        <div class="kpi"><span>Open commitments</span><strong>${open.length}</strong></div>
      </div>
      <div class="grid two">
        <section class="panel"><div class="panel-head"><h2>Identity, Goals, Commitments, Procedures</h2></div><div class="panel-body"><div class="rows">${canonical.map(atomRow).join("")}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Calibration And Runtime State</h2></div><div class="panel-body"><pre>${esc(JSON.stringify(state.current_self_model.self_awareness.calibration, null, 2))}</pre><pre>${esc(JSON.stringify(state.current_self_model.self_awareness.runtime_state?.payload || {}, null, 2))}</pre></div></section>
      </div>`;
    }

    function viewPacket(packet) {
      const history = state.memory_packets || [];
      return `<div class="grid two">
        <section class="panel"><div class="panel-head"><h2>Packet Items</h2><span class="chip">${esc(packet?.packet_id)}</span></div><div class="panel-body">${packetRows(packet, 20)}</div></section>
        <section class="panel"><div class="panel-head"><h2>Omissions And Degradation</h2></div><div class="panel-body"><pre>${esc(JSON.stringify({degradation: packet?.degradation, omissions: packet?.omissions}, null, 2))}</pre></div></section>
        <section class="panel"><div class="panel-head"><h2>Packet History</h2><span class="chip">${history.length} packets</span></div><div class="panel-body"><div class="rows">${history.map(item => `<div class="row"><div class="row-top"><strong>${esc(item.source)}</strong><span class="score">graph v${esc(item.graph_version)}</span></div><div class="meta">${esc(item.packet_id)} · ${esc(item.item_count)} items · ${esc(item.retrieval_mode)}</div></div>`).join("")}</div></div></section>
      </div>`;
    }

    function viewReasoning() {
      const reasoning = state.reasoning || {};
      const frame = reasoning.frame || {};
      const loaded = reasoning.loaded_page || {};
      const exact = reasoning.exact_lookup || {};
      const pages = frame.page_index || [];
      const units = frame.units || [];
      const revisionClass = reasoning.revision_current ? "good" : "warn";
      const revisionLabel = reasoning.revision_current ? "current revision" : "stale — recompile";
      return `<div class="kpi-grid">
        <div class="kpi"><span>Resident units</span><strong>${units.length}</strong></div>
        <div class="kpi"><span>Demand pages</span><strong>${pages.length}</strong></div>
        <div class="kpi"><span>Token estimate</span><strong>${esc(frame.token_estimate || 0)}</strong></div>
        <div class="kpi"><span>Truncated</span><strong>${frame.truncated ? "yes" : "no"}</strong></div>
      </div>
      <div class="grid two">
        <section class="panel">
          <div class="panel-head"><h2>Compile Coherent Memory</h2><span class="chip ${revisionClass}">${revisionLabel}</span></div>
          <div class="panel-body">
            <form class="composer" id="reasoning-form">
              <input id="reasoning-need" value="Why is the Mirror Agent specification first?" aria-label="Reasoning need">
              <button class="primary" type="submit">Compile</button>
            </form>
            <pre>${esc(JSON.stringify({frame_id: frame.frame_id, revision: frame.revision, orientation: frame.orientation, unknowns: frame.unknowns, compilation_trace: frame.compilation_trace}, null, 2))}</pre>
          </div>
        </section>
        <section class="panel"><div class="panel-head"><h2>Resident Coherent Units</h2><span class="chip">${units.length} units</span></div><div class="panel-body"><div class="rows">${units.map(unit => `<div class="row"><div class="row-top"><strong>${esc(unit.title || unit.unit_id)}</strong><span class="score">${esc(unit.unit_type)} · ${esc(unit.compression?.mode || "none")}</span></div><div class="meta">${esc(unit.summary || "")}</div><div class="refs">${refs(unit.source_atom_refs)}</div></div>`).join("") || `<div class="row"><div class="meta">No resident units; inspect the page index.</div></div>`}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Trusted Page Index</h2><span class="chip">${pages.length} descriptors</span></div><div class="panel-body"><div class="rows">${pages.map(page => `<div class="row"><div class="row-top"><strong>${esc(page.title || page.page_id)}</strong><button class="secondary load-reasoning-page" data-page-id="${esc(page.page_id)}" ${reasoning.revision_current ? "" : "disabled"}>Load page</button></div><div class="meta">${esc(page.summary || page.relevance || "")}</div><div class="refs">${refs(page.source_atom_refs)}</div></div>`).join("") || `<div class="row"><div class="meta">No deeper pages are needed for this frame.</div></div>`}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Loaded Page</h2><span class="chip">${esc(loaded.status || "not loaded")}</span></div><div class="panel-body"><pre>${esc(JSON.stringify(loaded, null, 2))}</pre></div></section>
        <section class="panel"><div class="panel-head"><h2>Exact-ID Lookup</h2><span class="chip">${esc(exact.retrieval_mode || "none")}</span></div><div class="panel-body"><pre>${esc(JSON.stringify(exact, null, 2))}</pre></div></section>
      </div>`;
    }

    function viewEvidence() {
      const ev = state.evidence.captured || [];
      return `<section class="panel"><div class="panel-head"><h2>Evidence Records</h2><span class="chip">${ev.length} captured</span></div><div class="panel-body"><div class="rows">${ev.map(e => `<div class="row"><div class="row-top"><strong>${esc(e.evidence_id)}</strong><span class="meta">${esc(e.source_type)}</span></div><div class="meta">${esc(e.source_ref)}</div><pre>${esc(JSON.stringify(e.payload, null, 2))}</pre></div>`).join("")}</div></div></section>`;
    }

    function viewMaintenance() {
      const latest = state.maintenance_journal.latest || {};
      const steward = latest.steward || {};
      const distiller = latest.maintenance_distiller || {};
      const committed = distiller.committed || [];
      const deferred = distiller.deferred || [];
      const proposals = distiller.proposals || [];
      const processors = (distiller.processors || []).map(p => p.processor_id).join(", ") || "none";
      const memoryPolicy = state.verification.memory.memory_policy || {};
      const policyTick = state.verification.memory.last_policy_tick || {};
      return `<div class="kpi-grid">
        <div class="kpi"><span>SMP outputs</span><strong>${latest.smp?.outputs?.length || 0}</strong></div>
        <div class="kpi"><span>Steward actions</span><strong>${steward.actions?.length || 0}</strong></div>
        <div class="kpi"><span>Processor proposals</span><strong>${proposals.length}</strong></div>
        <div class="kpi"><span>Committed</span><strong>${committed.filter(item => item.status === "committed" || item.status === "already_committed").length}</strong></div>
      </div>
      <div class="grid two">
        <section class="panel"><div class="panel-head"><h2>Automatic Memory Policy</h2><button class="secondary" id="run-maintenance">Run Now</button></div><div class="panel-body">
          <div class="refs"><span class="chip good">LM used: false</span><span class="chip ${memoryPolicy.due?.due ? "warn" : "good"}">${memoryPolicy.due?.due ? "due" : "scheduled"}</span><span class="chip">${esc(policyTick.status || "tick")}</span></div>
          <div class="rows">
            <div class="row"><div class="row-top"><strong>Policy State</strong><span class="score">graph v${esc(memoryPolicy.state?.last_graph_version || 0)}</span></div><div class="meta">Last trigger: ${esc(memoryPolicy.state?.last_trigger || "none")} · reasons: ${esc((memoryPolicy.state?.last_due_reasons || []).join(", ") || "none")}</div></div>
            <div class="row"><div class="row-top"><strong>Active Processor Packs</strong><span class="score">${esc(distiller.domain || "generic")}</span></div><div class="meta">${esc(processors)}</div></div>
            <div class="row"><div class="row-top"><strong>Reviewer</strong><span class="score">${esc(distiller.reviewer?.authority || "draft_only")}</span></div><div class="meta">${esc(distiller.reviewer?.status || "disabled")} · canonical mutation: ${esc(distiller.reviewer?.mutates_canonical_memory)}</div></div>
          </div>
        </div></section>
        <section class="panel"><div class="panel-head"><h2>Processor Pack Proposals</h2><span class="chip">${proposals.length} proposals</span></div><div class="panel-body"><div class="rows">${proposalRows(proposals)}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Committed Distillations</h2><span class="chip">${committed.length} commits</span></div><div class="panel-body"><div class="rows">${committedRows(committed)}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Deferred Review</h2><span class="chip">${deferred.length} deferred</span></div><div class="panel-body"><div class="rows">${deferredRows(deferred)}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Steward And Legacy Distillation</h2></div><div class="panel-body"><pre>${esc(JSON.stringify({smp_outputs: latest.smp?.outputs?.length || 0, steward_actions: steward.actions || [], deterministic_distillation: latest.distillation || {}, index: latest.index || {}, packet_cache: latest.packet_cache || {}, reviewer_policy: state.verification.llm_reviewer_policy}, null, 2))}</pre></div></section>
        <section class="panel"><div class="panel-head"><h2>Journal And Suppression</h2></div><div class="panel-body"><pre>${esc(JSON.stringify(state.maintenance_journal, null, 2))}</pre></div></section>
      </div>`;
    }

    function viewCapacity() {
      const health = state.capacity.health;
      return `<div class="grid two">
        <section class="panel"><div class="panel-head"><h2>Capacity Pressure</h2><span class="chip bad">${esc(health.pressure_mode)}</span></div><div class="panel-body"><div class="meter"><div></div></div><pre>${esc(JSON.stringify(state.capacity, null, 2))}</pre></div></section>
        <section class="panel"><div class="panel-head"><h2>Packet Impact</h2></div><div class="panel-body"><pre>${esc(JSON.stringify(state.capacity.degraded_packet, null, 2))}</pre></div></section>
      </div>`;
    }

    function viewGraph() {
      return `<div class="grid two">
        <section class="panel"><div class="panel-head"><h2>Selected Atoms</h2></div><div class="panel-body"><div class="rows">${(state.graph.selected_atoms || []).map(atomRow).join("")}</div></div></section>
        <section class="panel"><div class="panel-head"><h2>Associative Edges</h2></div><div class="panel-body"><div class="rows">${(state.graph.edges || []).map(edge => `<div class="row"><div class="row-top"><strong>${esc(edge.relation)}</strong><span class="meta">${esc(edge.health_status)} · v${esc(edge.version || "")}</span></div><div class="meta">${esc(edge.source_ref)} -> ${esc(edge.target_ref)}</div><div class="meta">derivation: ${esc(edge.derivation?.kind || "unknown")} · ${esc(edge.derivation?.processor_id || "intrinsic")} · confidence ${esc(edge.confidence?.score || "")}</div><div class="meta">feedback: used ${esc(edge.derivation?.retrieval_telemetry?.used_count || 0)} · corrected ${esc(edge.derivation?.retrieval_telemetry?.correction_count || 0)}</div><div class="refs">${refs(edge.evidence_refs)}</div></div>`).join("")}</div></div></section>
      </div>`;
    }

    function packetRows(packet, limit) {
      return `<div class="rows">${(packet?.items || []).slice(0, limit).map(item => `<div class="row"><div class="row-top"><strong>${esc(item.atom_ref)}</strong><span class="score">${esc(item.type)} ${esc(item.score)}</span></div><div class="meta">${esc(atomText(item))}</div><div class="refs">${refs(item.evidence_refs)}</div></div>`).join("")}</div>`;
    }

    function atomRow(atom) {
      return `<div class="row"><div class="row-top"><strong>${esc(atom.id || atom.atom_ref)}</strong><span class="score">${esc(atom.type)} v${esc(atom.version || "")}</span></div><div class="meta">${esc(atom.label || atomText(atom))}</div><div class="refs">${refs(atom.evidence_refs)}</div></div>`;
    }

    function proposalRows(proposals) {
      if (!proposals?.length) return `<div class="row"><div class="meta">No processor proposals in the latest tick.</div></div>`;
      return proposals.slice(0, 12).map(proposal => `<div class="row"><div class="row-top"><strong>${esc(proposal.title || proposal.action)}</strong><span class="score">${esc(proposal.risk_level)} ${esc(proposal.confidence)}</span></div><div class="meta">${esc(proposal.processor_id)} · ${esc(proposal.reason_code)}</div><div class="refs">${refs(proposal.source_refs)}</div></div>`).join("");
    }

    function committedRows(committed) {
      if (!committed?.length) return `<div class="row"><div class="meta">No low-risk proposals were committed in the latest tick.</div></div>`;
      return committed.map(item => `<div class="row"><div class="row-top"><strong>${esc(item.atom?.id || item.proposal_id)}</strong><span class="score">${esc(item.status)}</span></div><div class="meta">${esc(item.atom?.payload?.summary || item.reason || "")}</div><div class="refs">${refs(item.source_refs)}</div></div>`).join("");
    }

    function deferredRows(deferred) {
      if (!deferred?.length) return `<div class="row"><div class="meta">No proposals are waiting for review.</div></div>`;
      return deferred.map(item => `<div class="row"><div class="row-top"><strong>${esc(item.action)}</strong><span class="score">${esc(item.risk_level)}</span></div><div class="meta">${esc(item.reason)}</div><div class="refs">${refs(item.source_refs)}</div></div>`).join("");
    }

    function bindActions() {
      const form = $("chat-form");
      if (form) form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (chatPending) return;
        const message = $("message").value.trim();
        if (!message) return;
        chatPending = true;
        pendingMessage = message;
        render();
        try {
          const response = await fetch("/api/chat", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({message})});
          const data = await response.json();
          if (!response.ok) throw new Error(data.error || "chat failed");
          state = data.report || state;
          active = "Chat";
          scrollTranscriptAfterRender = true;
        } catch (error) {
          alert(error.message || "chat failed");
        } finally {
          chatPending = false;
          pendingMessage = "";
          render();
        }
      });
      const run = $("run-maintenance");
      if (run) run.addEventListener("click", async () => {
        run.disabled = true;
        const response = await fetch("/api/maintenance/run", {method: "POST"});
        const data = await response.json();
        if (!response.ok) alert(data.error || "maintenance failed");
        state = data.report || state;
        active = "Maintenance";
        render();
      });
      const reasoningForm = $("reasoning-form");
      if (reasoningForm) reasoningForm.addEventListener("submit", async (event) => {
        event.preventDefault();
        const need = $("reasoning-need").value.trim();
        if (!need) return;
        const response = await fetch("/api/reasoning/compile", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({need})});
        const data = await response.json();
        if (!response.ok) {
          alert(data.error || "reasoning compilation failed");
          return;
        }
        state = data.report || state;
        active = "Reasoning";
        render();
      });
      document.querySelectorAll(".load-reasoning-page").forEach(button => button.addEventListener("click", async () => {
        button.disabled = true;
        const response = await fetch("/api/reasoning/page", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({page_id: button.dataset.pageId})});
        const data = await response.json();
        if (!response.ok) {
          alert(data.error || "reasoning page load failed");
          return;
        }
        state = data.report || state;
        active = "Reasoning";
        render();
      }));
    }

    load();
  </script>
</body>
</html>
"""


def serve(host: str, port: int, db_path: Path, *, lm_mode: str) -> None:
    server = MirrorAgentUIServer((host, port), db_path, lm_mode=lm_mode)
    try:
        print(f"AMOS Mirror Agent UI: http://{host}:{server.server_address[1]}")
        print(f"LM provider: {server.state.lm_mode}")
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", default="/tmp/amos_mirror_ui.sqlite3")
    parser.add_argument("--lm", choices=["codex", "offline"], default="codex")
    args = parser.parse_args()
    serve(args.host, args.port, Path(args.db), lm_mode=args.lm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
