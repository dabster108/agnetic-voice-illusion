#!/usr/bin/env python
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_backend.crew import AgentBackend
from agent_backend.tools.custom_tool import GenerateDiagramMCPTool


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUT_PATH = PROJECT_ROOT / "agent_input.json"


def _configure_litellm_runtime() -> None:
	"""Reduce LiteLLM shutdown logging noise without changing model behavior."""
	# Keep user-provided setting if present; otherwise avoid very verbose internal logs.
	os.environ.setdefault("LITELLM_LOG", "ERROR")
	os.environ.setdefault("LITELLM_TELEMETRY", "False")

	try:
		# Pre-import apscheduler so its atexit registration happens before python shutdown phase.
		import apscheduler.schedulers.asyncio  # noqa
	except ImportError:
		pass

	try:
		import litellm

		litellm.telemetry = False
		litellm.suppress_debug_info = True
		litellm.turn_off_message_logging = True
		litellm.callbacks = []
		litellm.success_callback = []
		litellm.failure_callback = []
		litellm.service_callback = []
	except Exception:
		# Never block crew execution if LiteLLM internals change.
		pass


_configure_litellm_runtime()


app = FastAPI(title="Agent Backend API", version="0.1.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


class GenerateRequest(BaseModel):
	requirement: str = Field(..., min_length=1, description="Natural language requirement")
	input_source: str = Field(default="text")


class WorkspaceGenerateRequest(BaseModel):
	prompt: str = Field(..., min_length=1, description="Prompt to convert into workspace graph")
	input_source: str = Field(default="text")


_RENDERABLE_EXCALIDRAW_TYPES = {
	"cameraUpdate",
	"rectangle",
	"ellipse",
	"diamond",
	"text",
	"arrow",
	"line",
	"freedraw",
	"image",
	"frame",
	"embeddable",
}

_WORKSPACE_ACTION_WORDS = {
	"send",
	"sends",
	"sending",
	"receive",
	"receives",
	"receiving",
	"process",
	"processes",
	"processing",
	"store",
	"stores",
	"storing",
	"save",
	"saves",
	"saving",
	"read",
	"reads",
	"reading",
	"write",
	"writes",
	"writing",
	"create",
	"creates",
	"creating",
	"generate",
	"generates",
	"generating",
	"transform",
	"transforms",
	"transforming",
	"validate",
	"validates",
	"validating",
	"trigger",
	"triggers",
	"triggering",
	"notify",
	"notifies",
	"notifying",
	"return",
	"returns",
	"returning",
	"output",
	"outputs",
	"outputting",
	"publish",
	"publishes",
	"publishing",
	"forward",
	"forwards",
	"forwarding",
	"sync",
	"syncs",
	"syncing",
	"map",
	"maps",
	"mapping",
	"route",
	"routes",
	"routing",
	"deliver",
	"delivers",
	"delivering",
	"ingest",
	"ingests",
	"ingesting",
	"analyze",
	"analyzes",
	"analyzing",
	"score",
	"scores",
	"scoring",
}

_ACTION_BASE_BY_TOKEN = {
	"sends": "send",
	"sending": "send",
	"receives": "receive",
	"receiving": "receive",
	"processes": "process",
	"processing": "process",
	"stores": "store",
	"storing": "store",
	"saves": "save",
	"saving": "save",
	"reads": "read",
	"reading": "read",
	"writes": "write",
	"writing": "write",
	"creates": "create",
	"creating": "create",
	"generates": "generate",
	"generating": "generate",
	"transforms": "transform",
	"transforming": "transform",
	"validates": "validate",
	"validating": "validate",
	"triggers": "trigger",
	"triggering": "trigger",
	"notifies": "notify",
	"notifying": "notify",
	"returns": "return",
	"returning": "return",
	"outputs": "output",
	"outputting": "output",
	"publishes": "publish",
	"publishing": "publish",
	"forwards": "forward",
	"forwarding": "forward",
	"syncs": "sync",
	"syncing": "sync",
	"maps": "map",
	"mapping": "map",
	"routes": "route",
	"routing": "route",
	"delivers": "deliver",
	"delivering": "deliver",
	"ingests": "ingest",
	"ingesting": "ingest",
	"analyzes": "analyze",
	"analyzing": "analyze",
	"scores": "score",
	"scoring": "score",
}

_ENTITY_TRIM_WORDS = {
	"a",
	"an",
	"the",
	"me",
	"as",
	"ans",
	"this",
	"that",
	"these",
	"those",
	"my",
	"our",
	"your",
	"their",
	"for",
	"to",
	"from",
	"in",
	"into",
	"on",
	"at",
	"with",
	"by",
	"of",
	"and",
	"then",
	"after",
	"before",
	"when",
	"where",
	"while",
	"through",
	"across",
	"over",
	"under",
	"via",
	"please",
	"create",
	"build",
	"design",
	"generate",
	"draw",
	"show",
	"make",
	"diagram",
	"flowchart",
	"architecture",
	"system",
	"workflow",
	"process",
	"required",
	"need",
	"needs",
	"want",
	"wants",
}

_GENERIC_ENTITY_WORDS = {
	"diagram",
	"flowchart",
	"architecture",
	"system",
	"workflow",
	"process",
	"component",
	"components",
	"thing",
	"things",
}

_INPUT_HINT_WORDS = {
	"input",
	"request",
	"user",
	"client",
	"frontend",
	"ui",
	"web",
	"browser",
	"mobile",
	"react",
	"source",
	"prompt",
	"sensor",
	"file",
	"event",
	"message",
	"command",
	"form",
	"upload",
}

_STORAGE_HINT_WORDS = {
	"database",
	"storage",
	"store",
	"repository",
	"cache",
	"table",
	"bucket",
	"archive",
	"memory",
	"log",
	"ledger",
}

_OUTPUT_HINT_WORDS = {
	"output",
	"result",
	"response",
	"report",
	"dashboard",
	"alert",
	"notification",
	"view",
	"summary",
	"insight",
	"export",
	"email",
	"screen",
}

_DEFAULT_TEMPLATE_TERMS = {
	"api gateway",
	"fastapi",
	"backend",
	"core service",
	"primary database",
	"auth service",
	"postgres",
	"microservice",
}

_WORKSPACE_LAYER_COLORS = {
	0: "rgba(251, 191, 36, 0.24)",
	1: "rgba(59, 130, 246, 0.24)",
	2: "rgba(34, 197, 94, 0.24)",
	3: "rgba(20, 184, 166, 0.24)",
}

_WORKSPACE_NODE_WIDTH = 220.0
_WORKSPACE_NODE_HEIGHT = 108.0
_WORKSPACE_X_STEP = 300.0
_WORKSPACE_Y_STEP = 180.0
_WORKSPACE_MAX_NODES = 18


def _is_number(value: Any) -> bool:
	return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_label(value: Any) -> str:
	label = str(value or "").strip()
	if not label:
		return ""
	if label in {"-", "_"}:
		return ""
	if label.lower() in {"na", "n/a", "none", "null", "undefined"}:
		return ""
	return label[:80]


def _labels_from_requirement(requirement: str) -> list[str]:
	if not requirement:
		return []

	if "->" in requirement:
		parts = [p.strip() for p in requirement.split("->") if p.strip()]
	else:
		parts = [p.strip() for p in re.split(r"[,.;\n]", requirement) if p.strip()]

	return [_normalize_label(part) for part in parts if _normalize_label(part)]


def _ensure_rectangle_text_fields(payload: dict[str, Any], requirement: str) -> dict[str, Any]:
	elements = payload.get("elements")
	if not isinstance(elements, list) or not elements:
		return payload

	label_candidates: list[str] = []

	entities = payload.get("entities")
	if isinstance(entities, list):
		for entity in entities:
			if not isinstance(entity, dict):
				continue
			name_label = _normalize_label(entity.get("name"))
			type_label = _normalize_label(entity.get("type"))
			if name_label:
				label_candidates.append(name_label)
			elif type_label:
				label_candidates.append(type_label)

	for element in elements:
		if isinstance(element, dict) and element.get("type") == "text":
			text_label = _normalize_label(element.get("text"))
			if text_label:
				label_candidates.append(text_label)

	label_candidates.extend(_labels_from_requirement(requirement))

	candidate_idx = 0
	rectangle_idx = 0
	updated_elements: list[dict[str, Any]] = []

	for element in elements:
		if not isinstance(element, dict):
			continue

		if element.get("type") != "rectangle":
			updated_elements.append(element)
			continue

		rectangle_idx += 1
		label = _normalize_label(element.get("text"))
		if not label and candidate_idx < len(label_candidates):
			label = _normalize_label(label_candidates[candidate_idx])
			candidate_idx += 1
		if not label:
			label = f"Component {rectangle_idx}"

		new_element = dict(element)
		new_element["text"] = label
		new_element["textAlign"] = "center"
		new_element["verticalAlign"] = "middle"
		updated_elements.append(new_element)

	payload["elements"] = updated_elements
	return payload



def _rectangle_count(elements: Any) -> int:
	if not isinstance(elements, list):
		return 0
	return sum(1 for el in elements if isinstance(el, dict) and el.get("type") == "rectangle")


def _add_text_overlays_for_rectangles(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
	"""Add explicit text elements for rectangle labels for Excalidraw compatibility."""
	updated: list[dict[str, Any]] = []

	for idx, element in enumerate(elements, start=1):
		if not isinstance(element, dict) or element.get("type") != "rectangle":
			updated.append(element)
			continue

		label = _normalize_label(element.get("text"))
		if not label:
			updated.append(element)
			continue

		x = element.get("x")
		y = element.get("y")
		width = element.get("width")
		height = element.get("height")
		if not (_is_number(x) and _is_number(y) and _is_number(width) and _is_number(height)):
			updated.append(element)
			continue

		rect_id = element.get("id") or f"rect_{idx}"
		text_id = f"label_overlay_{idx}"

		rect_element = dict(element)
		rect_element["id"] = rect_id
		bound = rect_element.get("boundElements", []) or []
		rect_element["boundElements"] = bound + [{"id": text_id, "type": "text"}]
		updated.append(rect_element)

		updated.append(
			{
				"type": "text",
				"id": text_id,
				"x": float(x) + 14,
				"y": float(y) + (float(height) / 2) - 12,
				"width": float(width) - 28,
				"height": float(height) - 24,
				"text": label,
				"fontSize": 20,
				"fontFamily": 1,
				"textAlign": "center",
				"verticalAlign": "middle",
				"strokeColor": "#0f172a",
				"containerId": rect_id,
			}
		)

	return updated


def _extract_json_from_text(text: str) -> dict[str, Any]:
	raw = text.strip()
	if not raw:
		return {}

	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		match = re.search(r"\{[\s\S]*\}", raw)
		if not match:
			return {"raw": raw}
		try:
			return json.loads(match.group(0))
		except json.JSONDecodeError:
			return {"raw": raw}


def _normalize_crew_output(result: Any) -> dict[str, Any]:
	if isinstance(result, dict):
		return result
	if isinstance(result, str):
		return _extract_json_from_text(result)

	raw = getattr(result, "raw", None)
	if isinstance(raw, str):
		return _extract_json_from_text(raw)

	text = str(result)
	return _extract_json_from_text(text)


def _build_fallback_result(requirement: str, error_message: str) -> dict[str, Any]:
	"""Return a stable Excalidraw payload when crew execution fails."""
	elements = GenerateDiagramMCPTool._build_elements_from_prompt(requirement)
	return {
		"type": "excalidraw",
		"version": 2,
		"source": "backend-fallback",
		"elements": elements,
		"appState": {"viewBackgroundColor": "#ffffff"},
		"diagram_type": "fallback",
		"entities": [],
		"relationships": [],
		"layout_suggestion": "linear",
		"warning": f"Crew fallback used: {error_message}",
	}


def _has_renderable_elements(payload: dict[str, Any]) -> bool:
	elements = payload.get("elements")
	if not isinstance(elements, list) or not elements:
		return False

	for element in elements:
		if not isinstance(element, dict):
			return False

		element_type = element.get("type")
		if element_type not in _RENDERABLE_EXCALIDRAW_TYPES:
			return False

		if element_type == "cameraUpdate":
			continue

		if element_type == "rectangle" and not _normalize_label(element.get("text")):
			return False

		if not _is_number(element.get("x")) or not _is_number(element.get("y")):
			return False

	return True


def _exc_rect_exit_toward(
	x: float, y: float, w: float, h: float, toward_x: float, toward_y: float
) -> tuple[float, float, tuple[float, float]]:
	"""Point where a ray from the rect center toward *toward* exits the rectangle (bounding box)."""
	cx = x + w / 2
	cy = y + h / 2
	dx = toward_x - cx
	dy = toward_y - cy
	if abs(dx) < 1e-9 and abs(dy) < 1e-9:
		dx, dy = 1.0, 0.0
	hw = w / 2
	hh = h / 2
	adx = abs(dx)
	ady = abs(dy)
	if adx * hh > ady * hw:
		t = (hw if dx > 0 else -hw) / dx
		px = cx + dx * t
		py = cy + dy * t
		fp_y = 0.5 if h <= 0 else min(1.0, max(0.0, (py - y) / h))
		fp = (1.0 if dx > 0 else 0.0, fp_y)
	else:
		t = (hh if dy > 0 else -hh) / dy
		px = cx + dx * t
		py = cy + dy * t
		fp_x = 0.5 if w <= 0 else min(1.0, max(0.0, (px - x) / w))
		fp = (fp_x, 1.0 if dy > 0 else 0.0)
	return px, py, fp


def _exc_shape_exit_toward(
	shape_type: str,
	x: float,
	y: float,
	w: float,
	h: float,
	toward_x: float,
	toward_y: float,
) -> tuple[float, float, tuple[float, float]]:
	shape = (shape_type or "rectangle").lower()
	cx = x + w / 2
	cy = y + h / 2
	dx = toward_x - cx
	dy = toward_y - cy
	if abs(dx) < 1e-9 and abs(dy) < 1e-9:
		dx, dy = 1.0, 0.0

	hw = max(w / 2, 1e-6)
	hh = max(h / 2, 1e-6)

	if shape == "ellipse":
		d = ((dx * dx) / (hw * hw) + (dy * dy) / (hh * hh)) ** 0.5
		if d <= 1e-9:
			return _exc_rect_exit_toward(x, y, w, h, toward_x, toward_y)
		t = 1.0 / d
		px = cx + dx * t
		py = cy + dy * t
		fp_x = 0.5 if w <= 0 else min(1.0, max(0.0, (px - x) / w))
		fp_y = 0.5 if h <= 0 else min(1.0, max(0.0, (py - y) / h))
		return px, py, (fp_x, fp_y)

	if shape == "diamond":
		d = abs(dx) / hw + abs(dy) / hh
		if d <= 1e-9:
			return _exc_rect_exit_toward(x, y, w, h, toward_x, toward_y)
		t = 1.0 / d
		px = cx + dx * t
		py = cy + dy * t
		fp_x = 0.5 if w <= 0 else min(1.0, max(0.0, (px - x) / w))
		fp_y = 0.5 if h <= 0 else min(1.0, max(0.0, (py - y) / h))
		return px, py, (fp_x, fp_y)

	return _exc_rect_exit_toward(x, y, w, h, toward_x, toward_y)


def _exc_arrow_binding(
	element_id: str,
	fixed_point: tuple[float, float] | None = None,
) -> dict[str, Any]:
	binding: dict[str, Any] = {"elementId": element_id, "focus": 0.5, "gap": 0}
	if fixed_point is None:
		return binding

	fx = min(1.0, max(0.0, float(fixed_point[0])))
	fy = min(1.0, max(0.0, float(fixed_point[1])))
	binding["fixedPoint"] = [fx, fy]
	return binding


def _build_elements_from_entities_and_relationships(payload: dict[str, Any]) -> list[dict[str, Any]]:
	"""Build a renderable Excalidraw scene from entities/relationships when available."""
	entities_raw = payload.get("entities")
	if not isinstance(entities_raw, list) or not entities_raw:
		return []

	entities: list[dict[str, str]] = []
	for idx, item in enumerate(entities_raw, start=1):
		if not isinstance(item, dict):
			continue
		entity_id = str(item.get("id") or f"entity_{idx}")
		entity_name = _normalize_label(item.get("name"))
		if not entity_name:
			entity_name = _normalize_label(item.get("type"))
		if not entity_name:
			entity_name = f"Entity {idx}"
		entities.append({
			"id": entity_id,
			"name": entity_name[:64],
			"type": str(item.get("type", "rectangle"))
		})

	if not entities:
		return []

	elements: list[dict[str, Any]] = [
		{"type": "cameraUpdate", "width": 1200, "height": 700, "x": 0, "y": 0}
	]

	relationships = payload.get("relationships") or []
	adj: dict[str, list[str]] = {e["id"]: [] for e in entities}
	in_degree: dict[str, int] = {e["id"]: 0 for e in entities}

	for rel in relationships:
		if not isinstance(rel, dict):
			continue
		src = str(rel.get("source_id", ""))
		tgt = str(rel.get("target_id", ""))
		if src in adj and tgt in adj:
			adj[src].append(tgt)
			in_degree[tgt] += 1

	levels: dict[str, int] = {}
	queue = []
	for e_id, deg in in_degree.items():
		if deg == 0:
			queue.append((e_id, 0))

	if not queue and entities:
		queue.append((entities[0]["id"], 0))

	while queue:
		curr_id, level = queue.pop(0)
		levels[curr_id] = max(levels.get(curr_id, 0), level)
		for neighbor in adj[curr_id]:
			if levels.get(neighbor, -1) < level + 1:
				levels[neighbor] = level + 1
				queue.append((neighbor, level + 1))

	for e in entities:
		if e["id"] not in levels:
			levels[e["id"]] = 0

	level_nodes: dict[int, list[dict]] = {}
	for e in entities:
		lvl = levels[e["id"]]
		level_nodes.setdefault(lvl, []).append(e)

	layout_style = str(payload.get("layout_suggestion", "horizontal")).lower()
	is_vertical = layout_style in ["vertical", "top-to-bottom", "bottom-up"]

	base_box_w = 240
	base_box_h = 110
	gap_x = 120
	gap_y = 120
	start_x = 120
	start_y = 200

	id_to_bounds: dict[str, dict[str, Any]] = {}

	for lvl in sorted(level_nodes.keys()):
		nodes = level_nodes[lvl]
		node_sizes: list[tuple[float, float]] = []
		for entity in nodes:
			name_len = len(str(entity.get("name") or ""))
			node_w = float(min(360, max(base_box_w, 36 + name_len * 8)))
			node_h = float(base_box_h + (22 if name_len > 24 else 0))
			node_sizes.append((node_w, node_h))

		# Compute starting positions to roughly center the level
		if is_vertical:
			total_width = sum(width for width, _ in node_sizes) + max(0, len(nodes) - 1) * gap_x
			current_x = start_x + (1200 - total_width) / 2
			level_max_h = max((height for _, height in node_sizes), default=base_box_h)
			current_y = start_y + lvl * (level_max_h + gap_y)
		else:
			total_height = sum(height for _, height in node_sizes) + max(0, len(nodes) - 1) * gap_y
			current_x = start_x + lvl * (base_box_w + gap_x + 40)
			current_y = start_y + (700 - total_height) / 2

		cursor_x = current_x
		cursor_y = current_y

		for idx, entity in enumerate(nodes):
			node_w, node_h = node_sizes[idx]
			if is_vertical:
				x = cursor_x
				y = current_y
				cursor_x += node_w + gap_x
			else:
				x = current_x
				y = cursor_y
				cursor_y += node_h + gap_y

			box_id = f"box_{entity['id']}"
			label_id = f"label_{entity['id']}"

			raw_type = entity["type"].lower()
			name_lower = entity["name"].lower()
			shape_type = "rectangle"
			if raw_type in ["actor", "ellipse", "circle", "user"] or "user" in name_lower or "client" in name_lower:
				shape_type = "ellipse"
			elif raw_type in ["diamond", "decision", "condition"]:
				shape_type = "diamond"
			elif raw_type == "output":
				shape_type = "ellipse"

			bg_color = "#bbf7d0"
			if shape_type == "ellipse":
				bg_color = "#fdba74" if raw_type == "output" else "#fef08a"
			elif shape_type == "diamond":
				bg_color = "#fbcfe8"
			elif raw_type in ["database", "storage"] or "database" in name_lower or "sql" in name_lower or "storage" in name_lower or "firebase" in name_lower or "redis" in name_lower:
				bg_color = "#bae6fd"
				shape_type = "rectangle"
			elif raw_type in ("tool", "library", "package") or "oauth" in name_lower or "jwt" in name_lower:
				bg_color = "#e9d5ff"
			elif raw_type in ("apiserver", "api", "backend") or "django" in name_lower or "fastapi" in name_lower:
				bg_color = "#99f6e4"
			elif raw_type == "component":
				bg_color = "#bbf7d0"

			roundness = {"type": 3} if shape_type == "rectangle" else None

			elements.append(
				{
					"type": shape_type,
					"id": box_id,
					"x": x,
					"y": y,
					"width": node_w,
					"height": node_h,
					"roundness": roundness,
					"strokeColor": "#1f2937",
					"backgroundColor": bg_color,
					"text": entity["name"],
					"textAlign": "center",
					"verticalAlign": "middle",
					"fillStyle": "solid",
					"boundElements": [{"type": "text", "id": label_id}],
				}
			)
			elements.append(
				{
					"type": "text",
					"id": label_id,
					"containerId": box_id,
					"x": x + 14,
					"y": y + (node_h / 2) - 12,
					"width": node_w - 28,
					"height": node_h - 24,
					"text": entity["name"],
					"fontSize": 20,
					"fontFamily": 1,
					"textAlign": "center",
					"verticalAlign": "middle",
					"strokeColor": "#0f172a",
				}
			)
			id_to_bounds[entity["id"]] = {
				"x": x,
				"y": y,
				"w": node_w,
				"h": node_h,
				"shape": shape_type,
			}

	if isinstance(relationships, list) and relationships:
		edge_specs: list[dict[str, Any]] = []
		for rel in relationships:
			if not isinstance(rel, dict):
				continue
			source_id = str(rel.get("source_id", ""))
			target_id = str(rel.get("target_id", ""))
			if source_id not in id_to_bounds or target_id not in id_to_bounds:
				continue
			source_bounds = id_to_bounds[source_id]
			target_bounds = id_to_bounds[target_id]
			sx0 = float(source_bounds["x"])
			sy0 = float(source_bounds["y"])
			sw = float(source_bounds["w"])
			sh = float(source_bounds["h"])
			tx0 = float(target_bounds["x"])
			ty0 = float(target_bounds["y"])
			tw = float(target_bounds["w"])
			th = float(target_bounds["h"])
			tcx, tcy = tx0 + tw / 2, ty0 + th / 2
			scx, scy = sx0 + sw / 2, sy0 + sh / 2
			sx, sy, source_fp = _exc_shape_exit_toward(
				str(source_bounds.get("shape") or "rectangle"),
				sx0,
				sy0,
				sw,
				sh,
				tcx,
				tcy,
			)
			tx, ty, target_fp = _exc_shape_exit_toward(
				str(target_bounds.get("shape") or "rectangle"),
				tx0,
				ty0,
				tw,
				th,
				scx,
				scy,
			)
			dx = tx - sx
			dy = ty - sy
			edge_specs.append(
				{
					"source_id": source_id,
					"target_id": target_id,
					"sx": sx,
					"sy": sy,
					"dx": dx,
					"dy": dy,
					"source_fp": source_fp,
					"target_fp": target_fp,
					"rel": rel,
				}
			)

		arrow_idx = 1
		for spec in edge_specs:
			rel = spec["rel"]
			source_id = str(spec["source_id"])
			target_id = str(spec["target_id"])
			sx, sy = float(spec["sx"]), float(spec["sy"])
			dx, dy = float(spec["dx"]), float(spec["dy"])
			arrow_id = f"arrow_{arrow_idx}"
			arrow_element: dict[str, Any] = {
				"type": "arrow",
				"id": arrow_id,
				"x": sx,
				"y": sy,
				"width": dx,
				"height": dy,
				"points": [[0, 0], [dx, dy]],
				"strokeColor": "#0f766e",
				"strokeWidth": 2,
				"endArrowhead": "arrow",
				"startBinding": _exc_arrow_binding(
					f"box_{source_id}",
					tuple(spec.get("source_fp") or (0.5, 0.5)),
				),
				"endBinding": _exc_arrow_binding(
					f"box_{target_id}",
					tuple(spec.get("target_fp") or (0.5, 0.5)),
				),
			}

			rel_label = str(rel.get("label", "")).strip()
			if rel_label and rel_label != "-":
				text_id = f"arrow_label_{arrow_idx}"
				arrow_element["boundElements"] = [{"type": "text", "id": text_id}]
				elements.append(arrow_element)
				elements.append(
					{
						"type": "text",
						"id": text_id,
						"containerId": arrow_id,
						"x": sx + dx / 2,
						"y": sy + dy / 2 - 20,
						"text": rel_label[:30],
						"fontSize": 18,
						"fontFamily": 1,
						"textAlign": "center",
						"verticalAlign": "middle",
						"strokeColor": "#0f172a",
					}
				)
			else:
				elements.append(arrow_element)
			arrow_idx += 1
	else:
		chain_specs: list[dict[str, Any]] = []
		for i in range(1, len(entities)):
			source_id = str(entities[i - 1]["id"])
			target_id = str(entities[i]["id"])
			if source_id not in id_to_bounds or target_id not in id_to_bounds:
				continue
			source_bounds = id_to_bounds[source_id]
			target_bounds = id_to_bounds[target_id]
			sx0 = float(source_bounds["x"])
			sy0 = float(source_bounds["y"])
			sw = float(source_bounds["w"])
			sh = float(source_bounds["h"])
			tx0 = float(target_bounds["x"])
			ty0 = float(target_bounds["y"])
			tw = float(target_bounds["w"])
			th = float(target_bounds["h"])
			tcx, tcy = tx0 + tw / 2, ty0 + th / 2
			scx, scy = sx0 + sw / 2, sy0 + sh / 2
			sx, sy, source_fp = _exc_shape_exit_toward(
				str(source_bounds.get("shape") or "rectangle"),
				sx0,
				sy0,
				sw,
				sh,
				tcx,
				tcy,
			)
			tx, ty, target_fp = _exc_shape_exit_toward(
				str(target_bounds.get("shape") or "rectangle"),
				tx0,
				ty0,
				tw,
				th,
				scx,
				scy,
			)
			dx = tx - sx
			dy = ty - sy
			chain_specs.append(
				{
					"source_id": source_id,
					"target_id": target_id,
					"sx": sx,
					"sy": sy,
					"dx": dx,
					"dy": dy,
					"source_fp": source_fp,
					"target_fp": target_fp,
				}
			)
		for i, spec in enumerate(chain_specs, start=1):
			source_id = str(spec["source_id"])
			target_id = str(spec["target_id"])
			sx, sy = float(spec["sx"]), float(spec["sy"])
			dx, dy = float(spec["dx"]), float(spec["dy"])
			elements.append(
				{
					"type": "arrow",
					"id": f"arrow_{i}",
					"x": sx,
					"y": sy,
					"width": dx,
					"height": dy,
					"points": [[0, 0], [dx, dy]],
					"strokeColor": "#0f766e",
					"strokeWidth": 2,
					"endArrowhead": "arrow",
					"startBinding": _exc_arrow_binding(
						f"box_{source_id}",
						tuple(spec.get("source_fp") or (0.5, 0.5)),
					),
					"endBinding": _exc_arrow_binding(
						f"box_{target_id}",
						tuple(spec.get("target_fp") or (0.5, 0.5)),
					),
				}
			)

	return elements


def _ensure_renderable_result(result_payload: dict[str, Any], requirement: str) -> dict[str, Any]:
	result_payload = _ensure_rectangle_text_fields(result_payload, requirement)

	entities = result_payload.get("entities")
	entity_count = len(entities) if isinstance(entities, list) else 0
	rectangle_count = _rectangle_count(result_payload.get("elements"))

	# Prefer rebuilding from entities when model output under-represents the parsed system.
	should_rebuild_from_entities = entity_count > 0 and rectangle_count < entity_count

	if _has_renderable_elements(result_payload) and not should_rebuild_from_entities:
		return result_payload

	entity_elements = _build_elements_from_entities_and_relationships(result_payload)
	if entity_elements:
		existing_warning = str(result_payload.get("warning", "")).strip()
		entity_warning = "Rendered scene from entities/relationships because original elements were invalid or incomplete."
		result_payload["type"] = result_payload.get("type") or "excalidraw"
		result_payload["version"] = result_payload.get("version") or 2
		result_payload["source"] = result_payload.get("source") or "backend-entity-coerced"
		result_payload["elements"] = entity_elements
		result_payload["appState"] = result_payload.get("appState") or {"viewBackgroundColor": "#ffffff"}
		result_payload["warning"] = f"{existing_warning} | {entity_warning}" if existing_warning else entity_warning
		return result_payload

	fallback_elements = GenerateDiagramMCPTool._build_elements_from_prompt(requirement)
	existing_warning = str(result_payload.get("warning", "")).strip()
	coerce_warning = (
		"Result did not contain renderable Excalidraw elements; generated deterministic fallback scene."
	)
	warning = f"{existing_warning} | {coerce_warning}" if existing_warning else coerce_warning

	result_payload["type"] = result_payload.get("type") or "excalidraw"
	result_payload["version"] = result_payload.get("version") or 2
	result_payload["source"] = result_payload.get("source") or "backend-coerced-fallback"
	result_payload["elements"] = _add_text_overlays_for_rectangles(fallback_elements)
	result_payload["appState"] = result_payload.get("appState") or {"viewBackgroundColor": "#ffffff"}
	result_payload["diagram_type"] = result_payload.get("diagram_type") or "fallback"
	result_payload["entities"] = result_payload.get("entities") or []
	result_payload["relationships"] = result_payload.get("relationships") or []
	result_payload["layout_suggestion"] = result_payload.get("layout_suggestion") or "linear"
	result_payload["warning"] = warning
	return result_payload


def _as_float(value: Any, default: float) -> float:
	if _is_number(value):
		return float(value)
	return default


def _normalize_node_color(value: Any) -> str:
	color = str(value or "").strip()
	if not color:
		return "rgba(56, 189, 248, 0.25)"
	if color.startswith("#") or color.startswith("rgb"):
		return color
	return "rgba(56, 189, 248, 0.25)"


def _normalize_workspace_text(value: str) -> str:
	text = str(value or "")
	text = text.replace("\u2192", "->")
	text = re.sub(r"\bans\b", "and", text, flags=re.IGNORECASE)
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def _normalize_action_token(value: str) -> str:
	token = re.sub(r"[^a-z]", "", str(value or "").lower())
	if not token:
		return "flow"
	return _ACTION_BASE_BY_TOKEN.get(token, token)


def _clean_entity_phrase(value: str) -> str:
	candidate = str(value or "")
	candidate = re.sub(r"[\(\)\[\]\{\}:]", " ", candidate)
	candidate = re.sub(r"[^A-Za-z0-9/&\-\s]", " ", candidate)
	parts = [part for part in candidate.strip().split() if part]

	while parts and parts[0].lower() in _ENTITY_TRIM_WORDS:
		parts.pop(0)
	while parts and parts[-1].lower() in _ENTITY_TRIM_WORDS:
		parts.pop()

	if not parts:
		return ""

	if len(parts) > 6:
		parts = parts[:6]

	formatted: list[str] = []
	for part in parts:
		lower_part = part.lower()
		if lower_part in _ENTITY_TRIM_WORDS and len(parts) > 1:
			continue
		if part.isupper() or any(char.isdigit() for char in part):
			formatted.append(part)
		else:
			formatted.append(part.capitalize())

	label = " ".join(formatted).strip()
	if not label:
		return ""

	if label.lower() in _GENERIC_ENTITY_WORDS:
		return ""

	return label[:64]


def _parse_requirement_stage(requirement: str) -> dict[str, Any]:
	text = _normalize_workspace_text(requirement)
	entities_by_key: dict[str, dict[str, str]] = {}
	actions: list[str] = []
	relationships: list[dict[str, str]] = []
	seen_relationships: set[str] = set()
	seen_actions: set[str] = set()

	def register_entity(raw_label: str) -> str:
		clean_label = _clean_entity_phrase(raw_label)
		if not clean_label:
			return ""
		key = clean_label.lower()
		if key not in entities_by_key:
			if len(entities_by_key) >= _WORKSPACE_MAX_NODES:
				return ""
			entities_by_key[key] = {
				"id": f"entity_{len(entities_by_key) + 1}",
				"label": clean_label,
			}
		return entities_by_key[key]["id"]

	def register_action(raw_action: str) -> str:
		action = _normalize_action_token(raw_action)
		if not action:
			return ""
		if action not in seen_actions:
			seen_actions.add(action)
			actions.append(action)
		return action

	def register_relationship(source_id: str, target_id: str, raw_action: str) -> None:
		if not source_id or not target_id or source_id == target_id:
			return
		action = register_action(raw_action)
		if not action:
			action = "flow"
		relation_key = f"{source_id}->{target_id}:{action}"
		if relation_key in seen_relationships:
			return
		seen_relationships.add(relation_key)
		relationships.append(
			{
				"source_id": source_id,
				"target_id": target_id,
				"action": action,
			}
		)

	if "->" in text:
		for chain in re.findall(r"[A-Za-z0-9][^.;\n]{0,220}?->[A-Za-z0-9][^.;\n]{0,220}", text):
			segments = [_clean_entity_phrase(segment) for segment in chain.split("->")]
			segments = [segment for segment in segments if segment]
			if len(segments) < 2:
				continue
			for index in range(1, len(segments)):
				source_id = register_entity(segments[index - 1])
				target_id = register_entity(segments[index])
				register_relationship(source_id, target_id, "flow")

	role_entity_pattern = re.compile(
		r"(?P<name>[A-Za-z][A-Za-z0-9/&\-\s]{1,70}?)\s+(?:as|for)\s+(?P<role>frontend|backend|client|server|database|storage|auth|authentication|api|ui)\b",
		re.IGNORECASE,
	)
	for match in role_entity_pattern.finditer(text):
		name = _clean_entity_phrase(match.group("name"))
		role = _normalize_workspace_text(match.group("role")).lower()
		if role.startswith("auth"):
			role = "authentication"
		if name:
			register_entity(f"{name} {role}")

	verb_pattern = "|".join(sorted((re.escape(word) for word in _WORKSPACE_ACTION_WORDS), key=len, reverse=True))
	active_pattern = re.compile(
		rf"(?P<src>[A-Za-z][A-Za-z0-9/&\-\s]{{1,80}}?)\s+(?P<verb>{verb_pattern})\s+(?P<tgt>[A-Za-z][A-Za-z0-9/&\-\s]{{1,80}}?)(?=$|[,.;]|\bthen\b|\band\b|\bwhile\b)",
		re.IGNORECASE,
	)
	from_to_pattern = re.compile(
		rf"(?P<verb>{verb_pattern})\s+from\s+(?P<src>[A-Za-z][A-Za-z0-9/&\-\s]{{1,80}}?)\s+to\s+(?P<tgt>[A-Za-z][A-Za-z0-9/&\-\s]{{1,80}}?)(?=$|[,.;]|\bthen\b|\band\b)",
		re.IGNORECASE,
	)

	for sentence in re.split(r"[.;\n]", text):
		chunk = sentence.strip()
		if not chunk:
			continue

		for match in from_to_pattern.finditer(chunk):
			source_id = register_entity(match.group("src"))
			target_id = register_entity(match.group("tgt"))
			register_relationship(source_id, target_id, match.group("verb"))

		for match in active_pattern.finditer(chunk):
			source_id = register_entity(match.group("src"))
			target_id = register_entity(match.group("tgt"))
			register_relationship(source_id, target_id, match.group("verb"))

	for candidate in re.split(r"->|,|;|\band\b|\bthen\b|\bafter\b|\bbefore\b", text, flags=re.IGNORECASE):
		register_entity(candidate)

	def find_entity_ids(*keywords: str) -> list[str]:
		entity_ids: list[str] = []
		for entity in entities_by_key.values():
			label = str(entity.get("label") or "").lower()
			if any(keyword in label for keyword in keywords):
				entity_ids.append(str(entity.get("id") or ""))
		return [entity_id for entity_id in entity_ids if entity_id]

	if not relationships and len(entities_by_key) >= 2:
		frontend_ids = find_entity_ids("frontend", "client", "ui", "web", "browser", "react")
		backend_ids = find_entity_ids("backend", "server", "api", "fastapi")
		auth_ids = find_entity_ids("auth", "authentication", "identity", "login")
		storage_ids = find_entity_ids("database", "storage", "cache", "repository", "table")

		if frontend_ids and backend_ids:
			register_relationship(frontend_ids[0], backend_ids[0], "request")

		if backend_ids and auth_ids:
			register_relationship(backend_ids[0], auth_ids[0], "authenticate")

		if backend_ids and storage_ids:
			register_relationship(backend_ids[0], storage_ids[0], "store")

	if not relationships and len(entities_by_key) >= 2:
		connection_cue = re.search(
			r"\b(and|with|using|via|through|between|connect|integration|frontend|backend|client|server)\b",
			text,
			re.IGNORECASE,
		)
		if connection_cue:
			ordered_entities = list(entities_by_key.values())
			for index in range(1, len(ordered_entities)):
				source_id = str(ordered_entities[index - 1].get("id") or "")
				target_id = str(ordered_entities[index].get("id") or "")
				register_relationship(source_id, target_id, "flow")

	entities = list(entities_by_key.values())
	parsing_failed = len(entities) < 2 or len(relationships) == 0

	return {
		"entities": entities,
		"actions": actions,
		"relationships": relationships,
		"parse_failed": parsing_failed,
	}


def _contains_hint(value: str, hints: set[str]) -> bool:
	lower_value = value.lower()
	return any(hint in lower_value for hint in hints)


def _infer_workspace_layer(label: str, incoming: int, outgoing: int) -> int:
	if _contains_hint(label, _STORAGE_HINT_WORDS):
		return 2
	if _contains_hint(label, _OUTPUT_HINT_WORDS):
		return 3
	if _contains_hint(label, _INPUT_HINT_WORDS):
		return 0
	if incoming == 0 and outgoing > 0:
		return 0
	if outgoing == 0 and incoming > 0:
		return 3
	return 1


def _edge_priority(edge: dict[str, Any], layer_by_node: dict[str, int]) -> int:
	source_layer = layer_by_node.get(str(edge.get("from") or ""), 1)
	target_layer = layer_by_node.get(str(edge.get("to") or ""), 1)
	score = 0
	if source_layer < target_layer:
		score += 5
	elif source_layer == target_layer:
		score += 2
	if _normalize_label(edge.get("label")):
		score += 1
	score += max(0, 3 - abs(target_layer - source_layer))
	return score


def _prune_workspace_edges(
	edges: list[dict[str, Any]],
	layer_by_node: dict[str, int],
	max_edges: int,
) -> list[dict[str, Any]]:
	best_by_pair: dict[tuple[str, str], dict[str, Any]] = {}

	for edge in edges:
		source = str(edge.get("from") or "").strip()
		target = str(edge.get("to") or "").strip()
		if not source or not target or source == target:
			continue

		pair = tuple(sorted((source, target)))
		existing = best_by_pair.get(pair)
		if not existing:
			best_by_pair[pair] = edge
			continue

		if _edge_priority(edge, layer_by_node) > _edge_priority(existing, layer_by_node):
			best_by_pair[pair] = edge

	filtered = list(best_by_pair.values())
	filtered.sort(
		key=lambda edge: (
			-layer_by_node.get(str(edge.get("from") or ""), 1),
			-layer_by_node.get(str(edge.get("to") or ""), 1),
			str(edge.get("from") or ""),
			str(edge.get("to") or ""),
		)
	)

	forward_only: list[dict[str, Any]] = []
	seen_exact: set[str] = set()
	for edge in filtered:
		source = str(edge.get("from") or "").strip()
		target = str(edge.get("to") or "").strip()
		source_layer = layer_by_node.get(source, 1)
		target_layer = layer_by_node.get(target, 1)
		if source_layer > target_layer:
			continue

		normalized_label = _normalize_label(edge.get("label"))
		edge_key = f"{source}->{target}:{normalized_label.lower()}"
		if edge_key in seen_exact:
			continue
		seen_exact.add(edge_key)

		forward_only.append(
			{
				"id": edge.get("id") or f"edge_{len(forward_only) + 1}",
				"from": source,
				"to": target,
				"label": normalized_label,
			}
		)

	if not forward_only and filtered:
		first = filtered[0]
		forward_only = [
			{
				"id": first.get("id") or "edge_1",
				"from": str(first.get("from") or ""),
				"to": str(first.get("to") or ""),
				"label": _normalize_label(first.get("label")),
			}
		]

	if max_edges > 0 and len(forward_only) > max_edges:
		forward_only.sort(key=lambda edge: _edge_priority(edge, layer_by_node), reverse=True)
		forward_only = forward_only[:max_edges]

	for index, edge in enumerate(forward_only, start=1):
		edge["id"] = f"edge_{index}"

	return forward_only


def _ensure_workspace_connectivity(
	nodes: list[dict[str, Any]],
	edges: list[dict[str, Any]],
	layer_by_node: dict[str, int],
) -> list[dict[str, Any]]:
	node_ids = [str(node.get("id") or "").strip() for node in nodes]
	node_ids = [node_id for node_id in node_ids if node_id]
	if len(node_ids) <= 1:
		return edges

	node_order = {node_id: idx for idx, node_id in enumerate(node_ids)}
	normalized_edges: list[dict[str, Any]] = []
	existing_pairs: set[tuple[str, str]] = set()
	adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_ids}

	for edge in edges:
		source = str(edge.get("from") or "").strip()
		target = str(edge.get("to") or "").strip()
		if source not in adjacency or target not in adjacency or source == target:
			continue
		pair = (source, target)
		if pair in existing_pairs:
			continue

		normalized_edges.append(
			{
				"id": edge.get("id") or f"edge_{len(normalized_edges) + 1}",
				"from": source,
				"to": target,
				"label": _normalize_label(edge.get("label")) or "flow",
			}
		)
		existing_pairs.add(pair)
		adjacency[source].add(target)
		adjacency[target].add(source)

	visited: set[str] = set()
	components: list[list[str]] = []
	for node_id in node_ids:
		if node_id in visited:
			continue
		stack = [node_id]
		component: list[str] = []
		while stack:
			current = stack.pop()
			if current in visited:
				continue
			visited.add(current)
			component.append(current)
			for neighbor in adjacency.get(current, set()):
				if neighbor not in visited:
					stack.append(neighbor)
		components.append(component)

	if len(components) <= 1:
		for index, edge in enumerate(normalized_edges, start=1):
			edge["id"] = f"edge_{index}"
		return normalized_edges

	def component_sort_key(component: list[str]) -> tuple[int, int]:
		min_layer = min(layer_by_node.get(node_id, 1) for node_id in component)
		min_index = min(node_order.get(node_id, 10**6) for node_id in component)
		return min_layer, min_index

	components.sort(key=component_sort_key)

	for idx in range(1, len(components)):
		left_component = components[idx - 1]
		right_component = components[idx]

		source = max(
			left_component,
			key=lambda node_id: (layer_by_node.get(node_id, 1), -node_order.get(node_id, 0)),
		)
		target = min(
			right_component,
			key=lambda node_id: (layer_by_node.get(node_id, 1), node_order.get(node_id, 0)),
		)

		source_layer = layer_by_node.get(source, 1)
		target_layer = layer_by_node.get(target, 1)
		if source_layer > target_layer:
			source = min(
				left_component,
				key=lambda node_id: (layer_by_node.get(node_id, 1), node_order.get(node_id, 0)),
			)
			target = max(
				right_component,
				key=lambda node_id: (layer_by_node.get(node_id, 1), -node_order.get(node_id, 0)),
			)
			source_layer = layer_by_node.get(source, 1)
			target_layer = layer_by_node.get(target, 1)

		if source_layer > target_layer:
			source, target = target, source

		if source == target:
			continue

		pair = (source, target)
		if pair in existing_pairs:
			continue

		normalized_edges.append(
			{
				"id": f"edge_{len(normalized_edges) + 1}",
				"from": source,
				"to": target,
				"label": "flow",
			}
		)
		existing_pairs.add(pair)

	for index, edge in enumerate(normalized_edges, start=1):
		edge["id"] = f"edge_{index}"

	return normalized_edges


def _build_fail_safe_workspace_graph(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	seed_labels = [
		_clean_entity_phrase(entity.get("label"))
		for entity in parsed.get("entities", [])
		if isinstance(entity, dict)
	]
	seed_labels = [label for label in seed_labels if label]

	if len(seed_labels) >= 3:
		labels = [seed_labels[0], seed_labels[len(seed_labels) // 2], seed_labels[-1]]
	else:
		labels = ["Input", "Process", "Output"]

	layers = [0, 1, 3]
	nodes: list[dict[str, Any]] = []
	for index, label in enumerate(labels):
		layer = layers[index]
		nodes.append(
			{
				"id": f"node_{index + 1}",
				"title": label,
				"description": "Fail-safe linear flow node.",
				"x": layer * _WORKSPACE_X_STEP,
				"y": 0.0,
				"width": _WORKSPACE_NODE_WIDTH,
				"height": _WORKSPACE_NODE_HEIGHT,
				"color": _WORKSPACE_LAYER_COLORS.get(layer, "rgba(59, 130, 246, 0.24)"),
			}
		)

	edges = [
		{"id": "edge_1", "from": "node_1", "to": "node_2", "label": "flow"},
		{"id": "edge_2", "from": "node_2", "to": "node_3", "label": "flow"},
	]
	return nodes, edges


def _generate_workspace_graph_stage(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	entities_raw = parsed.get("entities")
	relationships_raw = parsed.get("relationships")
	if not isinstance(entities_raw, list) or not entities_raw:
		return _build_fail_safe_workspace_graph(parsed)

	if parsed.get("parse_failed"):
		return _build_fail_safe_workspace_graph(parsed)

	entities: list[dict[str, str]] = []
	for entity in entities_raw:
		if not isinstance(entity, dict):
			continue
		entity_id = str(entity.get("id") or "").strip()
		label = _clean_entity_phrase(entity.get("label"))
		if not entity_id or not label:
			continue
		entities.append(
			{
				"id": entity_id,
				"label": label,
				"kind": str(entity.get("kind") or "").strip().lower(),
				"shape": str(entity.get("shape") or "").strip().lower(),
			}
		)

	if len(entities) < 2:
		return _build_fail_safe_workspace_graph(parsed)

	incoming: dict[str, int] = {entity["id"]: 0 for entity in entities}
	outgoing: dict[str, int] = {entity["id"]: 0 for entity in entities}
	relationships: list[dict[str, str]] = []

	if isinstance(relationships_raw, list):
		for relation in relationships_raw:
			if not isinstance(relation, dict):
				continue
			source_id = str(relation.get("source_id") or "").strip()
			target_id = str(relation.get("target_id") or "").strip()
			action = _normalize_action_token(str(relation.get("action") or "flow"))
			if source_id not in outgoing or target_id not in incoming:
				continue
			if source_id == target_id:
				continue
			outgoing[source_id] += 1
			incoming[target_id] += 1
			relationships.append(
				{
					"source_id": source_id,
					"target_id": target_id,
					"action": action,
				}
			)

	if not relationships:
		return _build_fail_safe_workspace_graph(parsed)

	layer_by_entity: dict[str, int] = {}
	for entity in entities:
		layer_by_entity[entity["id"]] = _infer_workspace_layer(
			entity["label"],
			incoming.get(entity["id"], 0),
			outgoing.get(entity["id"], 0),
		)

	if not any(layer == 0 for layer in layer_by_entity.values()):
		layer_by_entity[entities[0]["id"]] = 0
	if not any(layer == 3 for layer in layer_by_entity.values()):
		layer_by_entity[entities[-1]["id"]] = 3

	order_by_entity = {entity["id"]: idx for idx, entity in enumerate(entities)}
	layer_buckets: dict[int, list[dict[str, str]]] = {0: [], 1: [], 2: [], 3: []}
	for entity in entities:
		layer_buckets.setdefault(layer_by_entity.get(entity["id"], 1), []).append(entity)

	for layer_idx in (1, 2, 3):
		bucket = layer_buckets.get(layer_idx, [])

		def relation_order(entity: dict[str, str]) -> tuple[float, int]:
			predecessor_indexes = [
				order_by_entity[relation["source_id"]]
				for relation in relationships
				if relation.get("target_id") == entity["id"] and relation.get("source_id") in order_by_entity
			]
			if predecessor_indexes:
				average_order = sum(predecessor_indexes) / len(predecessor_indexes)
			else:
				average_order = float(order_by_entity.get(entity["id"], 0))
			return average_order, order_by_entity.get(entity["id"], 0)

		bucket.sort(key=relation_order)

	nodes: list[dict[str, Any]] = []
	entity_to_node: dict[str, str] = {}
	for layer_idx in (0, 1, 2, 3):
		bucket = layer_buckets.get(layer_idx, [])
		for index_within_layer, entity in enumerate(bucket):
			node_id = f"node_{len(nodes) + 1}"
			entity_to_node[entity["id"]] = node_id
			nodes.append(
				{
					"id": node_id,
					"title": entity["label"],
					"description": "Input node." if layer_idx == 0 else "Output node." if layer_idx == 3 else "Storage node." if layer_idx == 2 else "Processing node.",
					"x": layer_idx * _WORKSPACE_X_STEP,
					"y": index_within_layer * _WORKSPACE_Y_STEP,
					"width": _WORKSPACE_NODE_WIDTH,
					"height": _WORKSPACE_NODE_HEIGHT,
					"color": _WORKSPACE_LAYER_COLORS.get(layer_idx, "rgba(59, 130, 246, 0.24)"),
					"kind": entity.get("kind") or "",
					"shape": entity.get("shape") or "",
				}
			)

	edges: list[dict[str, Any]] = []
	for relation in relationships:
		source_node = entity_to_node.get(relation["source_id"])
		target_node = entity_to_node.get(relation["target_id"])
		if not source_node or not target_node:
			continue
		edges.append(
			{
				"id": f"edge_{len(edges) + 1}",
				"from": source_node,
				"to": target_node,
				"label": _normalize_label(relation.get("action")) or "flow",
			}
		)

	layer_by_node = {
		node["id"]: int(round(_as_float(node.get("x"), 0.0) / _WORKSPACE_X_STEP))
		for node in nodes
	}
	max_edges = max(len(nodes) * 2, 2)
	cleaned_edges = _prune_workspace_edges(edges, layer_by_node, max_edges)
	return nodes, cleaned_edges


def _fix_node_overlaps(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
	occupied: set[tuple[float, float]] = set()
	for node in nodes:
		x = _as_float(node.get("x"), 0.0)
		y = _as_float(node.get("y"), 0.0)
		while (x, y) in occupied:
			y += _WORKSPACE_Y_STEP
		node["x"] = x
		node["y"] = y
		occupied.add((x, y))
	return nodes


def _looks_like_backend_template(requirement: str, nodes: list[dict[str, Any]]) -> bool:
	requirement_lower = str(requirement or "").lower()
	if any(term in requirement_lower for term in _DEFAULT_TEMPLATE_TERMS):
		return False

	template_hits = 0
	for node in nodes:
		title = str(node.get("title") or "").lower()
		if any(term in title for term in _DEFAULT_TEMPLATE_TERMS):
			template_hits += 1

	return template_hits >= 3


def _validate_workspace_graph(
	requirement: str,
	parsed: dict[str, Any],
	nodes: list[dict[str, Any]],
	edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	if not nodes:
		return _build_fail_safe_workspace_graph(parsed)

	nodes = _fix_node_overlaps(nodes)
	layer_by_node = {
		node["id"]: int(round(_as_float(node.get("x"), 0.0) / _WORKSPACE_X_STEP))
		for node in nodes
	}
	edges = _prune_workspace_edges(edges, layer_by_node, max(max(1, len(nodes)) * 2, 2))
	edges = _ensure_workspace_connectivity(nodes, edges, layer_by_node)

	if _looks_like_backend_template(requirement, nodes):
		return _build_fail_safe_workspace_graph(parsed)

	return nodes, edges


def _build_workspace_graph_from_requirement(requirement: str) -> dict[str, Any]:
	parsed = _parse_requirement_stage(requirement)
	nodes, edges = _generate_workspace_graph_stage(parsed)
	nodes, edges = _validate_workspace_graph(requirement, parsed, nodes, edges)

	if not nodes:
		nodes, edges = _build_fail_safe_workspace_graph(parsed)

	return {"nodes": nodes, "edges": edges}


def _build_parsed_stage_from_crew_payload(
	requirement: str,
	crew_payload: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
	prompt_parsed = _parse_requirement_stage(requirement)
	if not isinstance(crew_payload, dict):
		return prompt_parsed, False

	raw_nodes = crew_payload.get("nodes")
	raw_edges = crew_payload.get("edges")
	raw_entities = crew_payload.get("entities")
	raw_relationships = crew_payload.get("relationships")

	entities: list[dict[str, str]] = []
	relationships: list[dict[str, str]] = []
	actions: list[str] = []
	seen_actions: set[str] = set()
	seen_relationships: set[str] = set()
	id_lookup: dict[str, str] = {}
	label_lookup: dict[str, str] = {}

	def register_entity(
		raw_id: Any,
		raw_label: Any,
		kind: str = "",
		shape: str = "",
	) -> str:
		label = _clean_entity_phrase(str(raw_label or ""))
		if not label:
			return ""

		kind_n = str(kind or "").strip().lower()
		shape_n = str(shape or "").strip().lower()

		label_key = label.lower()
		if label_key in label_lookup:
			canonical_id = label_lookup[label_key]
			for ent in entities:
				if ent["id"] == canonical_id:
					if kind_n:
						ent["kind"] = kind_n
					if shape_n:
						ent["shape"] = shape_n
					break
		else:
			canonical_id = f"entity_{len(entities) + 1}"
			label_lookup[label_key] = canonical_id
			entities.append(
				{"id": canonical_id, "label": label, "kind": kind_n, "shape": shape_n}
			)

		normalized_raw_id = str(raw_id or "").strip()
		if normalized_raw_id:
			id_lookup[normalized_raw_id] = canonical_id
			if normalized_raw_id.startswith("box_"):
				id_lookup[normalized_raw_id[4:]] = canonical_id
			else:
				id_lookup[f"box_{normalized_raw_id}"] = canonical_id

		return canonical_id

	def resolve_entity(raw_ref: Any) -> str:
		normalized_ref = str(raw_ref or "").strip()
		if not normalized_ref:
			return ""
		if normalized_ref in id_lookup:
			return id_lookup[normalized_ref]
		if normalized_ref.startswith("box_") and normalized_ref[4:] in id_lookup:
			return id_lookup[normalized_ref[4:]]

		label_candidate = _clean_entity_phrase(normalized_ref)
		if not label_candidate:
			return ""
		return register_entity(normalized_ref, label_candidate)

	def register_action(raw_action: Any) -> str:
		action = _normalize_action_token(str(raw_action or "flow"))
		if action not in seen_actions:
			seen_actions.add(action)
			actions.append(action)
		return action

	def register_relationship(raw_source: Any, raw_target: Any, raw_action: Any) -> None:
		source_id = resolve_entity(raw_source)
		target_id = resolve_entity(raw_target)
		if not source_id or not target_id or source_id == target_id:
			return

		action = register_action(raw_action)
		relation_key = f"{source_id}->{target_id}:{action}"
		if relation_key in seen_relationships:
			return

		seen_relationships.add(relation_key)
		relationships.append(
			{
				"source_id": source_id,
				"target_id": target_id,
				"action": action,
			}
		)

	if isinstance(raw_nodes, list):
		for idx, node in enumerate(raw_nodes, start=1):
			if not isinstance(node, dict):
				continue
			node_id = str(node.get("id") or f"node_{idx}")
			node_label = (
				_normalize_label(node.get("title"))
				or _normalize_label(node.get("name"))
				or _normalize_label(node.get("label"))
				or _normalize_label(node.get("type"))
			)
			register_entity(
				node_id,
				node_label,
				str(node.get("kind") or node.get("role") or "").strip().lower(),
				str(node.get("shape") or "").strip().lower(),
			)

	if isinstance(raw_entities, list):
		for idx, entity in enumerate(raw_entities, start=1):
			if not isinstance(entity, dict):
				continue
			entity_id = str(entity.get("id") or f"entity_{idx}")
			entity_label = (
				_normalize_label(entity.get("name"))
				or _normalize_label(entity.get("label"))
				or _normalize_label(entity.get("type"))
			)
			register_entity(
				entity_id,
				entity_label,
				str(entity.get("kind") or "").strip().lower(),
				str(entity.get("shape") or "").strip().lower(),
			)

	if isinstance(raw_edges, list):
		for edge in raw_edges:
			if not isinstance(edge, dict):
				continue
			register_relationship(
				edge.get("from") or edge.get("source_id"),
				edge.get("to") or edge.get("target_id"),
				edge.get("label") or edge.get("action") or "flow",
			)

	if isinstance(raw_relationships, list):
		for relation in raw_relationships:
			if not isinstance(relation, dict):
				continue
			register_relationship(
				relation.get("source_id") or relation.get("from"),
				relation.get("target_id") or relation.get("to"),
				relation.get("action") or relation.get("label") or "flow",
			)

	if len(entities) >= 2 and len(relationships) >= 1:
		return {
			"entities": entities,
			"actions": actions,
			"relationships": relationships,
			"parse_failed": False,
		}, True

	return prompt_parsed, False


def _workspace_node_excalidraw_type(title: str, kind: str, shape: str) -> str:
	"""Map workspace metadata + label heuristics to entity types used for Excalidraw styling."""
	sh = (shape or "").strip().lower()
	k = (kind or "").strip().lower()
	if sh in ("diamond", "ellipse", "rectangle"):
		if sh == "diamond":
			return "decision"
		if sh == "ellipse":
			return "actor"
	if k in ("decision", "condition", "gateway"):
		return "decision"
	if k in ("input", "actor", "client", "ui"):
		return "actor"
	if k in ("storage", "database", "persistence"):
		return "database"
	if k in ("tool", "library", "package", "integration"):
		return "tool"
	if k in ("output", "response"):
		return "output"

	low = str(title or "").lower()
	if any(w in low for w in ("valid?", "authorized?", "success?", "decision")):
		return "decision"
	if any(w in low for w in ("react", "frontend", "client ui", "browser", "spa ", "next.js")):
		return "actor"
	if any(w in low for w in ("postgres", "mysql", "sqlite", "redis", "database", "orm", "migrations")):
		return "database"
	if any(w in low for w in ("jwt", "oauth", "openid", "celery", "npm", "pip", "sdk", "stripe")):
		return "tool"
	if any(w in low for w in ("django", "fastapi", "express", "graphql", "rest api", "backend", "gunicorn")):
		return "apiserver"
	return "component"


def _build_workspace_graph_from_crew_payload(
	requirement: str,
	crew_payload: dict[str, Any],
) -> dict[str, Any]:
	parsed, used_crew_data = _build_parsed_stage_from_crew_payload(requirement, crew_payload)
	nodes, edges = _generate_workspace_graph_stage(parsed)
	nodes, edges = _validate_workspace_graph(requirement, parsed, nodes, edges)

	if not nodes:
		nodes, edges = _build_fail_safe_workspace_graph(parsed)

	return {
		"nodes": nodes,
		"edges": edges,
		"used_crew_data": used_crew_data,
	}


def _build_elements_from_workspace_graph(
	nodes: list[dict[str, Any]],
	edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
	if not nodes:
		return []

	entities: list[dict[str, Any]] = []
	relationships: list[dict[str, Any]] = []
	for node in nodes:
		node_id = str(node.get("id") or "").strip()
		if not node_id:
			continue
		title = _normalize_label(node.get("title")) or _normalize_label(node.get("description")) or node_id
		exc_type = _workspace_node_excalidraw_type(
			title,
			str(node.get("kind") or ""),
			str(node.get("shape") or ""),
		)
		entities.append(
			{
				"id": node_id,
				"name": title,
				"type": exc_type,
			}
		)

	for edge in edges:
		source_id = str(edge.get("from") or "").strip()
		target_id = str(edge.get("to") or "").strip()
		if not source_id or not target_id or source_id == target_id:
			continue
		relationships.append(
			{
				"source_id": source_id,
				"target_id": target_id,
				"label": _normalize_label(edge.get("label")) or "flow",
			}
		)

	payload = {
		"entities": entities,
		"relationships": relationships,
		"layout_suggestion": "horizontal",
	}
	return _build_elements_from_entities_and_relationships(payload)


def _extract_workspace_graph(
	result_payload: dict[str, Any],
	requirement: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
	elements = result_payload.get("elements")
	relationships = result_payload.get("relationships")
	entities = result_payload.get("entities")

	text_by_container: dict[str, str] = {}
	if isinstance(elements, list):
		for element in elements:
			if not isinstance(element, dict) or element.get("type") != "text":
				continue
			container_id = str(element.get("containerId") or "").strip()
			text_value = _normalize_label(element.get("text"))
			if container_id and text_value:
				text_by_container[container_id] = text_value

	entity_map: dict[str, dict[str, Any]] = {}
	if isinstance(entities, list):
		for idx, entity in enumerate(entities, start=1):
			if not isinstance(entity, dict):
				continue
			entity_id = str(entity.get("id") or f"entity_{idx}").strip()
			if entity_id:
				entity_map[entity_id] = entity

	nodes: list[dict[str, Any]] = []
	id_lookup: dict[str, str] = {}
	seen_node_ids: set[str] = set()

	def register_node(
		raw_id: str,
		title: str,
		description: str,
		x: float,
		y: float,
		width: float,
		height: float,
		color: str,
	) -> None:
		base_id = raw_id.strip() or f"node_{len(nodes) + 1}"
		normalized_id = base_id[4:] if base_id.startswith("box_") else base_id
		candidate = normalized_id or f"node_{len(nodes) + 1}"
		if candidate in seen_node_ids:
			candidate = f"{candidate}_{len(nodes) + 1}"

		seen_node_ids.add(candidate)
		id_lookup[base_id] = candidate
		id_lookup[normalized_id] = candidate
		id_lookup[f"box_{normalized_id}"] = candidate

		nodes.append(
			{
				"id": candidate,
				"title": title,
				"description": description,
				"x": x,
				"y": y,
				"width": width,
				"height": height,
				"color": color,
			}
		)

	if isinstance(elements, list):
		for idx, element in enumerate(elements, start=1):
			if not isinstance(element, dict):
				continue

			shape_type = element.get("type")
			if shape_type not in {"rectangle", "ellipse", "diamond"}:
				continue

			raw_id = str(element.get("id") or f"node_{idx}")
			normalized_id = raw_id[4:] if raw_id.startswith("box_") else raw_id
			entity_ref = entity_map.get(raw_id) or entity_map.get(normalized_id)

			title = (
				_normalize_label(element.get("text"))
				or _normalize_label(text_by_container.get(raw_id))
				or _normalize_label(text_by_container.get(normalized_id))
				or _normalize_label(entity_ref.get("name") if entity_ref else "")
				or _normalize_label(entity_ref.get("type") if entity_ref else "")
				or f"Component {len(nodes) + 1}"
			)

			entity_type = _normalize_label(entity_ref.get("type") if entity_ref else "")
			description = (
				f"{entity_type} component" if entity_type else "AI-generated architecture component."
			)

			register_node(
				raw_id=raw_id,
				title=title,
				description=description,
				x=_as_float(element.get("x"), 260 + ((idx - 1) % 4) * 280),
				y=_as_float(element.get("y"), 220 + ((idx - 1) // 4) * 180),
				width=_as_float(element.get("width"), 240),
				height=_as_float(element.get("height"), 108),
				color=_normalize_node_color(element.get("backgroundColor")),
			)

	if not nodes and isinstance(entities, list):
		for idx, entity in enumerate(entities, start=1):
			if not isinstance(entity, dict):
				continue

			entity_id = str(entity.get("id") or f"entity_{idx}")
			title = _normalize_label(entity.get("name")) or _normalize_label(entity.get("type")) or f"Component {idx}"
			entity_type = _normalize_label(entity.get("type"))
			description = f"{entity_type} component" if entity_type else "AI-generated architecture component."

			register_node(
				raw_id=entity_id,
				title=title,
				description=description,
				x=260 + ((idx - 1) % 4) * 280,
				y=220 + ((idx - 1) // 4) * 180,
				width=240,
				height=108,
				color="rgba(56, 189, 248, 0.25)",
			)

	if not nodes:
		fallback_labels = _labels_from_requirement(requirement)
		for idx, label in enumerate(fallback_labels[:8], start=1):
			register_node(
				raw_id=f"fallback_{idx}",
				title=label,
				description="Fallback component derived from prompt.",
				x=260 + ((idx - 1) % 4) * 280,
				y=220 + ((idx - 1) // 4) * 180,
				width=240,
				height=108,
				color="rgba(56, 189, 248, 0.25)",
			)

	edges: list[dict[str, Any]] = []
	seen_edges: set[str] = set()

	def add_edge(source: str, target: str, label: str = "") -> None:
		if not source or not target or source == target:
			return
		if source not in seen_node_ids or target not in seen_node_ids:
			return

		normalized_label = _normalize_label(label)
		edge_key = f"{source}->{target}:{normalized_label.lower()}"
		if edge_key in seen_edges:
			return

		seen_edges.add(edge_key)
		edges.append(
			{
				"id": f"edge_{len(edges) + 1}",
				"from": source,
				"to": target,
				"label": normalized_label,
			}
		)

	def resolve_node_id(raw_id: str) -> str:
		value = raw_id.strip()
		if not value:
			return ""
		if value in id_lookup:
			return id_lookup[value]
		if value.startswith("box_"):
			return id_lookup.get(value[4:], "")
		return id_lookup.get(f"box_{value}", "")

	if isinstance(relationships, list):
		for relation in relationships:
			if not isinstance(relation, dict):
				continue

			source_raw = str(relation.get("source_id") or "").strip()
			target_raw = str(relation.get("target_id") or "").strip()
			source = resolve_node_id(source_raw)
			target = resolve_node_id(target_raw)

			add_edge(source, target, str(relation.get("label") or ""))

	if not edges and isinstance(elements, list):
		for element in elements:
			if not isinstance(element, dict) or element.get("type") != "arrow":
				continue

			start_binding = element.get("startBinding")
			end_binding = element.get("endBinding")
			if not isinstance(start_binding, dict) or not isinstance(end_binding, dict):
				continue

			source_raw = str(start_binding.get("elementId") or "").strip()
			target_raw = str(end_binding.get("elementId") or "").strip()
			source = resolve_node_id(source_raw)
			target = resolve_node_id(target_raw)

			arrow_id = str(element.get("id") or "")
			label = _normalize_label(text_by_container.get(arrow_id))
			add_edge(source, target, label)

	if not edges and len(nodes) > 1:
		for idx in range(1, len(nodes)):
			add_edge(nodes[idx - 1]["id"], nodes[idx]["id"], "flow")

	return nodes, edges


def _build_workspace_steps(
	requirement: str,
	nodes: list[dict[str, Any]],
	edges: list[dict[str, Any]],
	base_response: dict[str, Any],
	result_payload: dict[str, Any],
) -> list[dict[str, Any]]:
	timestamp = datetime.now(timezone.utc).isoformat()
	prompt_preview = requirement.strip().replace("\n", " ")[:90]

	steps: list[dict[str, Any]] = [
		{
			"id": "backend-received",
			"title": "Prompt accepted",
			"message": f"Queued prompt: {prompt_preview}",
			"status": "completed",
			"timestamp": timestamp,
		},
		{
			"id": "backend-plan",
			"title": "Generating architecture",
			"message": "Crew agents analyzed the requirement and drafted system structure.",
			"status": "completed",
			"timestamp": timestamp,
		},
		{
			"id": "backend-nodes",
			"title": "Creating nodes",
			"message": f"Produced {len(nodes)} architecture node(s) for the canvas.",
			"status": "completed",
			"timestamp": timestamp,
		},
		{
			"id": "backend-edges",
			"title": "Connecting elements",
			"message": f"Generated {len(edges)} connection edge(s).",
			"status": "completed",
			"timestamp": timestamp,
		},
	]

	if base_response.get("fallback"):
		steps.append(
			{
				"id": "backend-fallback",
				"title": "Fallback applied",
				"message": "Primary generation failed, deterministic fallback graph used.",
				"status": "completed",
				"timestamp": timestamp,
			}
		)

	if base_response.get("mcp_synced"):
		steps.append(
			{
				"id": "backend-sync",
				"title": "Canvas sync",
				"message": "Diagram synced to MCP drawing service.",
				"status": "completed",
				"timestamp": timestamp,
			}
		)
	else:
		steps.append(
			{
				"id": "backend-sync",
				"title": "Canvas sync",
				"message": "MCP sync unavailable; returning local graph payload.",
				"status": "running",
				"timestamp": timestamp,
			}
		)

	warning = str(result_payload.get("warning") or "").strip()
	if warning:
		steps.append(
			{
				"id": "backend-warning",
				"title": "Generation warning",
				"message": warning,
				"status": "error",
				"timestamp": timestamp,
			}
		)

	return steps


def _sync_elements_to_mcp(elements: Any) -> dict[str, Any]:
	"""Push elements to Excalidraw MCP and return sync metadata."""
	if not isinstance(elements, list) or not elements:
		return {"mcp_synced": False, "mcp_checkpoint_id": "", "mcp_warning": "No elements to sync."}

	mcp_url = os.getenv("MCP_HTTP_URL", "http://localhost:3001/mcp").strip() or "http://localhost:3001/mcp"
	try:
		GenerateDiagramMCPTool._post_jsonrpc(
			mcp_url,
			{
				"jsonrpc": "2.0",
				"id": 1,
				"method": "initialize",
				"params": {
					"protocolVersion": "2024-11-05",
					"capabilities": {},
					"clientInfo": {"name": "agent-backend-api", "version": "0.1.0"},
				},
			},
		)

		rpc = GenerateDiagramMCPTool._post_jsonrpc(
			mcp_url,
			{
				"jsonrpc": "2.0",
				"id": 2,
				"method": "tools/call",
				"params": {
					"name": "create_view",
					"arguments": {
						"elements": json.dumps(elements),
					},
				},
			},
		)

		checkpoint_id = (
			rpc.get("result", {})
			.get("structuredContent", {})
			.get("checkpointId", "")
		)
		return {
			"mcp_synced": True,
			"mcp_checkpoint_id": checkpoint_id,
			"mcp_warning": "",
		}
	except Exception as exc:
		return {
			"mcp_synced": False,
			"mcp_checkpoint_id": "",
			"mcp_warning": f"MCP sync failed: {exc}",
		}


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/generate")
def generate(payload: GenerateRequest) -> dict[str, Any]:
	requirement = payload.requirement.strip()
	if not requirement:
		raise HTTPException(status_code=400, detail="requirement is required")

	inputs = {
		"user_input": requirement,
		"input_source": payload.input_source or "text",
	}
	_save_inputs_snapshot(inputs, entrypoint="fastapi_generate")

	try:
		result = AgentBackend().crew().kickoff(inputs=inputs)
	except Exception as exc:
		fallback = _build_fallback_result(requirement, str(exc))
		sync_meta = _sync_elements_to_mcp(fallback.get("elements", []))
		if sync_meta.get("mcp_checkpoint_id"):
			fallback["checkpoint_id"] = sync_meta["mcp_checkpoint_id"]
		if sync_meta.get("mcp_warning"):
			existing_warning = str(fallback.get("warning", "")).strip()
			fallback["warning"] = (
				f"{existing_warning} | {sync_meta['mcp_warning']}" if existing_warning else sync_meta["mcp_warning"]
			)
		return {
			"ok": True,
			"requirement": requirement,
			"result": fallback,
			"fallback": True,
			"mcp_synced": sync_meta["mcp_synced"],
			"mcp_checkpoint_id": sync_meta["mcp_checkpoint_id"],
		}

	normalized = _normalize_crew_output(result)
	renderable = _ensure_renderable_result(normalized, requirement)
	sync_meta = _sync_elements_to_mcp(renderable.get("elements", []))
	if sync_meta.get("mcp_checkpoint_id"):
		renderable["checkpoint_id"] = sync_meta["mcp_checkpoint_id"]
	if sync_meta.get("mcp_warning"):
		existing_warning = str(renderable.get("warning", "")).strip()
		renderable["warning"] = (
			f"{existing_warning} | {sync_meta['mcp_warning']}" if existing_warning else sync_meta["mcp_warning"]
		)
	return {
		"ok": True,
		"requirement": requirement,
		"result": renderable,
		"mcp_synced": sync_meta["mcp_synced"],
		"mcp_checkpoint_id": sync_meta["mcp_checkpoint_id"],
	}


@app.post("/workspace/generate")
def generate_workspace(payload: WorkspaceGenerateRequest) -> dict[str, Any]:
	prompt = payload.prompt.strip()
	if not prompt:
		raise HTTPException(status_code=400, detail="prompt is required")

	inputs = {
		"user_input": prompt,
		"input_source": payload.input_source or "text",
	}
	_save_inputs_snapshot(inputs, entrypoint="fastapi_workspace_generate")

	warnings: list[str] = []
	crew_used = False
	crew_output_used = False
	nodes: list[dict[str, Any]] = []
	edges: list[dict[str, Any]] = []
	elements: list[dict[str, Any]] = []

	try:
		crew_result = AgentBackend().crew().kickoff(inputs=inputs)
		crew_payload = _normalize_crew_output(crew_result)
		crew_renderable = _ensure_renderable_result(dict(crew_payload), prompt)
		crew_warning = str(crew_renderable.get("warning") or "").strip()
		if crew_warning:
			warnings.append(crew_warning)

		workspace_graph = _build_workspace_graph_from_crew_payload(prompt, crew_payload)
		nodes = workspace_graph.get("nodes", [])
		edges = workspace_graph.get("edges", [])
		if not nodes:
			fallback_graph = _build_workspace_graph_from_requirement(prompt)
			nodes = fallback_graph.get("nodes", [])
			edges = fallback_graph.get("edges", [])
		elements = _build_elements_from_workspace_graph(nodes, edges)
		crew_output_used = bool(workspace_graph.get("used_crew_data")) and bool(nodes)
		crew_used = True
	except Exception as exc:
		warnings.append(f"CrewAI workspace generation failed: {exc}")
		workspace_graph = _build_workspace_graph_from_requirement(prompt)
		nodes = workspace_graph.get("nodes", [])
		edges = workspace_graph.get("edges", [])
		elements = _build_elements_from_workspace_graph(nodes, edges)
		crew_output_used = False

	if not elements:
		elements = _build_elements_from_workspace_graph(nodes, edges)

	sync_meta = _sync_elements_to_mcp(elements)
	if sync_meta.get("mcp_warning"):
		warnings.append(str(sync_meta.get("mcp_warning") or ""))

	fallback_used = not crew_output_used
	result_payload = {"warning": " | ".join([w for w in warnings if w.strip()])}
	base_response = {
		"fallback": fallback_used,
		"mcp_synced": bool(sync_meta.get("mcp_synced")),
	}
	steps = _build_workspace_steps(prompt, nodes, edges, base_response, result_payload)

	return {
		"nodes": nodes,
		"edges": edges,
		"elements": elements,
		"execution_steps": steps,
		"meta": {
			"input_source": payload.input_source or "text",
			"crew_used": crew_used,
			"crew_output_used": crew_output_used,
			"render_source": "graph-rebuild",
			"fallback": fallback_used,
			"mcp_synced": bool(sync_meta.get("mcp_synced")),
			"checkpoint_id": str(sync_meta.get("mcp_checkpoint_id") or ""),
			"warning": result_payload["warning"],
		},
	}


def _build_agent_input_payload(inputs: dict[str, Any], entrypoint: str) -> dict[str, Any]:
	"""Build a structured payload that can be handed to the next MCP agent."""
	raw_text = str(inputs.get("user_input", "")).strip()
	input_source = str(inputs.get("input_source", "text")).strip() or "text"

	payload: dict[str, Any] = {
		"input": {
			"text": raw_text,
			"source": input_source,
		},
		"elaborated": {
			"intent_summary": (
				f"User requested: {raw_text}" if raw_text else "No user_input provided."
			),
			"handoff_target": "next_agent_for_mcp",
			"handoff_action": "Pass normalized JSON to the downstream agent that calls MCP.",
			"expected_next_payload_fields": [
				"cleaned_input",
				"original_input",
				"language_detected",
				"input_source",
				"noise_removed",
				"corrections_made",
				"ambiguities_flagged",
				"confidence_score",
				"is_multi_command",
				"segmented_commands",
			],
		},
		"meta": {
			"entrypoint": entrypoint,
			"saved_at_utc": datetime.now(timezone.utc).isoformat(),
		},
	}

	if "crewai_trigger_payload" in inputs:
		payload["trigger_payload"] = inputs["crewai_trigger_payload"]

	return payload


def _save_inputs_snapshot(inputs: dict[str, Any], entrypoint: str) -> None:
	"""Save the latest run input payload for downstream handoff."""
	payload = _build_agent_input_payload(inputs, entrypoint)
	AGENT_INPUT_PATH.write_text(
		json.dumps(payload, ensure_ascii=True, indent=2),
		encoding="utf-8",
	)


def _build_default_inputs() -> dict:
	return {
		"user_input": "um please make a flowchart for user login and password reset",
		"input_source": "text",
	}


def _build_run_inputs_from_cli() -> dict:
	inputs = _build_default_inputs()
	if len(sys.argv) > 1:
		cli_prompt = " ".join(sys.argv[1:]).strip()
		if cli_prompt:
			inputs["user_input"] = cli_prompt
	return inputs


def run():
	"""Run Agent 1 and Agent 2."""
	inputs = _build_run_inputs_from_cli()
	_save_inputs_snapshot(inputs, entrypoint="run_crew")
	try:
		return AgentBackend().crew().kickoff(inputs=inputs)
	except Exception as e:
		raise Exception(f"An error occurred while running the agents: {e}")


def train():
	"""Train the crew for a given number of iterations."""
	inputs = _build_default_inputs()
	try:
		AgentBackend().crew().train(
			n_iterations=int(sys.argv[1]),
			filename=sys.argv[2],
			inputs=inputs,
		)
	except Exception as e:
		raise Exception(f"An error occurred while training the crew: {e}")


def replay():
	"""Replay the crew execution from a specific task."""
	try:
		AgentBackend().crew().replay(task_id=sys.argv[1])
	except Exception as e:
		raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
	"""Test the crew execution and return results."""
	inputs = _build_default_inputs()
	try:
		AgentBackend().crew().test(
			n_iterations=int(sys.argv[1]),
			eval_llm=sys.argv[2],
			inputs=inputs,
		)
	except Exception as e:
		raise Exception(f"An error occurred while testing the crew: {e}")


def run_with_trigger():
	"""Run agents with trigger payload."""
	if len(sys.argv) < 2:
		raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

	try:
		trigger_payload = json.loads(sys.argv[1])
	except json.JSONDecodeError:
		raise Exception("Invalid JSON payload provided as argument")

	trigger_text = (
		trigger_payload.get("user_input")
		or trigger_payload.get("prompt")
		or trigger_payload.get("text")
		or ""
	)

	inputs = {
		"crewai_trigger_payload": trigger_payload,
		"user_input": trigger_text,
		"input_source": trigger_payload.get("input_source", "text"),
	}
	_save_inputs_snapshot(inputs, entrypoint="run_with_trigger")

	try:
		return AgentBackend().crew().kickoff(inputs=inputs)
	except Exception as e:
		raise Exception(f"An error occurred while running with trigger: {e}")


def serve_api():
	"""Run FastAPI server for real-time requirement-to-diagram requests."""
	try:
		import uvicorn
	except ImportError as exc:
		raise Exception("uvicorn is required. Install it with `uv add uvicorn`") from exc

	host = os.getenv("API_HOST", "0.0.0.0")
	port = int(os.getenv("API_PORT", "8000"))
	uvicorn.run("agent_backend.main:app", host=host, port=port, reload=False)

