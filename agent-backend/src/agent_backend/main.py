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
		if not isinstance(rel, dict): continue
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

	box_w = 240
	box_h = 110
	gap_x = 120
	gap_y = 120
	start_x = 120
	start_y = 200

	id_to_anchor: dict[str, tuple[float, float]] = {}

	for lvl in sorted(level_nodes.keys()):
		nodes = level_nodes[lvl]
		# Compute starting positions to roughly center the level
		if is_vertical:
			total_width = len(nodes) * box_w + max(0, len(nodes) - 1) * gap_x
			current_x = start_x + (1200 - total_width) / 2
			current_y = start_y + lvl * (box_h + gap_y)
		else:
			total_height = len(nodes) * box_h + max(0, len(nodes) - 1) * gap_y
			current_x = start_x + lvl * (box_w + gap_x)
			current_y = start_y + (700 - total_height) / 2

		for idx, entity in enumerate(nodes):
			x = current_x + idx * (box_w + gap_x) if is_vertical else current_x
			y = current_y if is_vertical else current_y + idx * (box_h + gap_y)

			box_id = f"box_{entity['id']}"
			label_id = f"label_{entity['id']}"

			raw_type = entity["type"].lower()
			name_lower = entity["name"].lower()
			shape_type = "rectangle"
			if raw_type in ["actor", "ellipse", "circle", "user"] or "user" in name_lower or "client" in name_lower:
				shape_type = "ellipse"
			elif raw_type in ["diamond", "decision", "condition"]:
				shape_type = "diamond"

			bg_color = "#c3fae8" # default green-ish
			if shape_type == "ellipse":
				bg_color = "#fef08a" # yellow
			elif shape_type == "diamond":
				bg_color = "#fbcfe8" # pink
			elif raw_type in ["database", "storage"] or "database" in name_lower or "sql" in name_lower or "storage" in name_lower or "firebase" in name_lower:
				bg_color = "#bae6fd" # blue
				shape_type = "rectangle" # databases are usually just colored rectangles if cylinders aren't allowed

			roundness = {"type": 3} if shape_type == "rectangle" else None

			elements.append(
				{
					"type": shape_type,
					"id": box_id,
					"x": x,
					"y": y,
					"width": box_w,
					"height": box_h,
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
					"y": y + (box_h / 2) - 12,
					"width": box_w - 28,
					"height": box_h - 24,
					"text": entity["name"],
					"fontSize": 20,
					"fontFamily": 1,
					"textAlign": "center",
					"verticalAlign": "middle",
					"strokeColor": "#0f172a",
				}
			)
			id_to_anchor[entity["id"]] = (x + box_w, y + box_h / 2)

	if isinstance(relationships, list) and relationships:
		arrow_idx = 1
		for rel in relationships:
			if not isinstance(rel, dict):
				continue
			source_id = str(rel.get("source_id", ""))
			target_id = str(rel.get("target_id", ""))
			if source_id not in id_to_anchor or target_id not in id_to_anchor:
				continue
			sx, sy = id_to_anchor[source_id]
			tx, ty = id_to_anchor[target_id]
			dx = tx - sx
			dy = ty - sy
			arrow_id = f"arrow_{arrow_idx}"
			arrow_element = {
				"type": "arrow",
				"id": arrow_id,
				"x": sx,
				"y": sy,
				"width": abs(dx),
				"height": abs(dy),
				"points": [[0, 0], [dx, dy]],
				"strokeColor": "#0f766e",
				"strokeWidth": 2,
				"endArrowhead": "arrow",
				"startBinding": {"elementId": f"box_{source_id}", "focus": 0.5, "gap": 3},
				"endBinding": {"elementId": f"box_{target_id}", "focus": 0.5, "gap": 3},
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
		for i in range(1, len(entities)):
			sx, sy = id_to_anchor[entities[i - 1]["id"]]
			tx, ty = id_to_anchor[entities[i]["id"]]
			dx = tx - sx
			dy = ty - sy
			elements.append(
				{
					"type": "arrow",
					"id": f"arrow_{i}",
					"x": sx,
					"y": sy,
					"width": abs(dx),
					"height": abs(dy),
					"points": [[0, 0], [dx, dy]],
					"strokeColor": "#0f766e",
					"strokeWidth": 2,
					"endArrowhead": "arrow",
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

	base_response = generate(
		GenerateRequest(
			requirement=prompt,
			input_source=payload.input_source or "text",
		)
	)

	result_payload = base_response.get("result")
	if not isinstance(result_payload, dict):
		raise HTTPException(status_code=500, detail="invalid generation payload")

	nodes, edges = _extract_workspace_graph(result_payload, prompt)
	steps = _build_workspace_steps(prompt, nodes, edges, base_response, result_payload)

	return {
		"ok": True,
		"prompt": prompt,
		"nodes": nodes,
		"edges": edges,
		"execution_steps": steps,
		"meta": {
			"fallback": bool(base_response.get("fallback")),
			"mcp_synced": bool(base_response.get("mcp_synced")),
			"checkpoint_id": str(base_response.get("mcp_checkpoint_id") or ""),
			"warning": str(result_payload.get("warning") or ""),
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

