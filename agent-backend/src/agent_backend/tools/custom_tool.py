import json
import os
import re
from typing import Any, Type
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GenerateDiagramMCPToolInput(BaseModel):
    """Input schema for GenerateDiagramMCPTool."""
    prompt: str = Field(..., description="The comprehensive string prompt instructing MCP to build a diagram based on the analysis.")

class GenerateDiagramMCPTool(BaseTool):
    name: str = "generate_diagram"
    description: str = (
        "Sends the defined diagram prompt to the external Excalidraw MCP tool "
        "and returns an Excalidraw-compatible JSON payload."
    )
    args_schema: Type[BaseModel] = GenerateDiagramMCPToolInput

    @staticmethod
    def _safe_id(prefix: str, idx: int) -> str:
        return f"{prefix}_{idx}"

    @staticmethod
    def _normalize_label(value: Any) -> str:
        label = str(value or "").strip()
        if not label:
            return ""
        if label in {"-", "_"}:
            return ""
        if label.lower() in {"na", "n/a", "none", "null", "undefined"}:
            return ""
        return label[:80]

    @staticmethod
    def _parse_sse_or_json(raw_text: str) -> dict[str, Any]:
        text = raw_text.strip()
        if not text:
            return {}
        if text.startswith("data:"):
            json_chunks: list[str] = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk and chunk != "[DONE]":
                        json_chunks.append(chunk)
            for chunk in reversed(json_chunks):
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    continue
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", text)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _build_elements_from_prompt(prompt: str) -> list[dict[str, Any]]:
        parts = [p.strip() for p in prompt.split("->") if p.strip()]
        if not parts:
            normalized = re.sub(r"\bthen\b", ",", prompt, flags=re.IGNORECASE)
            parts = [s.strip() for s in re.split(r"[,.\n]", normalized) if s.strip()][:6]
        parts = [GenerateDiagramMCPTool._normalize_label(part) for part in parts]
        parts = [part for part in parts if part]
        if not parts:
            parts = ["Requirement", "No details provided"]

        elements: list[dict[str, Any]] = [
            {"type": "cameraUpdate", "width": 800, "height": 600, "x": 0, "y": 0}
        ]

        box_w = 220
        box_h = 96
        gap = 90
        y = 220
        x0 = 80

        for idx, label in enumerate(parts):
            x = x0 + idx * (box_w + gap)
            box_id = GenerateDiagramMCPTool._safe_id("box", idx + 1)
            elements.append(
                {
                    "type": "rectangle",
                    "id": box_id,
                    "x": x,
                    "y": y,
                    "width": box_w,
                    "height": box_h,
                    "roundness": {"type": 3},
                    "strokeColor": "#1f2937",
                    "backgroundColor": "#dbeafe" if idx % 2 else "#c3fae8",
                    "text": label[:52],
                    "textAlign": "center",
                    "verticalAlign": "middle",
                    "fillStyle": "solid",
                }
            )
            if idx > 0:
                elements.append(
                    {
                        "type": "arrow",
                        "id": GenerateDiagramMCPTool._safe_id("arrow", idx),
                        "x": x - gap,
                        "y": y + (box_h / 2),
                        "width": gap,
                        "height": 0,
                        "points": [[0, 0], [gap, 0]],
                        "strokeColor": "#0f766e",
                        "strokeWidth": 2,
                        "endArrowhead": "arrow",
                    }
                )
        return elements

    @staticmethod
    def _post_jsonrpc(url: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = Request(
            url=url,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        with urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
        return GenerateDiagramMCPTool._parse_sse_or_json(body)

    def _run(self, prompt: str) -> str:
        mcp_url = os.getenv("MCP_HTTP_URL", "http://localhost:3001/mcp").strip()
        elements = self._build_elements_from_prompt(prompt)

        try:
            self._post_jsonrpc(
                mcp_url,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "crewai-agent-backend", "version": "0.1.0"},
                    },
                },
            )

            rpc = self._post_jsonrpc(
                mcp_url,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "create_view",
                        "arguments": {"elements": json.dumps(elements)},
                    },
                },
            )

            checkpoint_id = (
                rpc.get("result", {})
                .get("structuredContent", {})
                .get("checkpointId", "")
            )
            payload = {
                "type": "excalidraw",
                "version": 2,
                "source": "http://localhost:3001/mcp",
                "elements": elements,
                "appState": {"viewBackgroundColor": "#ffffff"},
                "checkpoint_id": checkpoint_id,
                "original_prompt": prompt,
            }
            return json.dumps(payload, ensure_ascii=True)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            fallback = {
                "type": "excalidraw",
                "version": 2,
                "source": "fallback-local",
                "elements": elements,
                "appState": {"viewBackgroundColor": "#ffffff"},
                "original_prompt": prompt,
                "warning": f"MCP call failed: {exc}",
            }
            return json.dumps(fallback, ensure_ascii=True)
