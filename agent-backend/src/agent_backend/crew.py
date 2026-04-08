import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


def _resolve_model_name() -> str:
	"""Resolve model name from env with a safe default for LiteLLM + Mistral."""
	return (
		os.getenv("MODEL", "").strip()
		or os.getenv("LITELLM_MODEL", "").strip()
		or "mistral/mistral-small-latest"
	)


def _validate_provider_credentials(model_name: str) -> None:
	"""Fail fast with actionable messages when provider keys are missing."""
	if model_name.startswith("mistral/") and not os.getenv("MISTRAL_API_KEY", "").strip():
		raise ValueError(
			"MISTRAL_API_KEY is required for mistral/* models. "
			"Set it in .env before running the crew."
		)


@CrewBase
class AgentBackend:
	"""Single-agent backend for Agent 1 preprocessing."""

	agents: List[BaseAgent]
	tasks: List[Task]

	@agent
	def input_preprocessor_agent(self) -> Agent:
		model_name = _resolve_model_name()
		_validate_provider_credentials(model_name)
		return Agent(
			config=self.agents_config["input_preprocessor_agent"],  # type: ignore[index]
			llm=model_name,
			verbose=True,
		)

	@task
	def preprocess_user_input_task(self) -> Task:
		return Task(
			config=self.tasks_config["preprocess_user_input_task"],  # type: ignore[index]
		)

	@crew
	def crew(self) -> Crew:
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)

