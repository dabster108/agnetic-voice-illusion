#!/usr/bin/env python
import json
import sys

from agent_backend.crew import AgentBackend


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

	try:
		return AgentBackend().crew().kickoff(inputs=inputs)
	except Exception as e:
		raise Exception(f"An error occurred while running with trigger: {e}")

