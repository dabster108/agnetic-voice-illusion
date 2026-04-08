#!/usr/bin/env python
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent_backend.crew import AgentBackend


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_INPUT_PATH = PROJECT_ROOT / "agent_input.json"


def _configure_litellm_runtime() -> None:
	"""Reduce LiteLLM shutdown logging noise without changing model behavior."""
	# Keep user-provided setting if present; otherwise avoid very verbose internal logs.
	os.environ.setdefault("LITELLM_LOG", "ERROR")

	try:
		import litellm

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
	"""Run Agent 1 only."""
	inputs = _build_run_inputs_from_cli()
	_save_inputs_snapshot(inputs, entrypoint="run_crew")
	try:
		return AgentBackend().crew().kickoff(inputs=inputs)
	except Exception as e:
		raise Exception(f"An error occurred while running Agent 1: {e}")


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
	"""Run Agent 1 with trigger payload."""
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

