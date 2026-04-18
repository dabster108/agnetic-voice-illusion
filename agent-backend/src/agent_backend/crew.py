import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from agent_backend.tools.custom_tool import GenerateDiagramMCPTool

def _resolve_model_name() -> str:
	"""Resolve model name from env with a safe default for LiteLLM + Groq."""
	return (
		os.getenv("MODEL", "").strip()
		or os.getenv("LITELLM_MODEL", "").strip()
		or "groq/llama-3.1-8b-instant"
	)


def _validate_provider_credentials(model_name: str) -> None:
	"""Fail fast with actionable messages when provider keys are missing."""
	if model_name.startswith("groq/") and not os.getenv("GROQ_API_KEY", "").strip():
		raise ValueError(
			"GROQ_API_KEY is required for groq/* models. "
			"Set it in .env before running the crew."
		)


@CrewBase
class AgentBackend:
	"""Multi-agent backend for Agent 1 preprocessing and Agent 2 generation."""

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
		
	@agent
	def diagram_builder_agent(self) -> Agent:
		model_name = _resolve_model_name()
		_validate_provider_credentials(model_name)
		use_mcp_tool = os.getenv("USE_MCP_TOOL", "false").strip().lower() in {"1", "true", "yes", "on"}
		agent_kwargs = {
			"config": self.agents_config["diagram_builder_agent"],  # type: ignore[index]
			"llm": model_name,
			"verbose": True,
		}
		if use_mcp_tool:
			agent_kwargs["tools"] = [GenerateDiagramMCPTool()]
		return Agent(
			**agent_kwargs,
		)

	@agent
	def schema_builder_agent(self) -> Agent:
		model_name = _resolve_model_name()
		_validate_provider_credentials(model_name)
		return Agent(
			config=self.agents_config["schema_builder_agent"],  # type: ignore[index]
			llm=model_name,
			verbose=True,
		)

	@task
	def preprocess_user_input_task(self) -> Task:
		return Task(
			config=self.tasks_config["preprocess_user_input_task"],  # type: ignore[index]
		)

	@task
	def schema_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config["schema_generation_task"],  # type: ignore[index]
		)

	@task
	def diagram_generation_task(self) -> Task:
		return Task(
			config=self.tasks_config["diagram_generation_task"],  # type: ignore[index]
		)

	@crew
	def crew(self) -> Crew:
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)

