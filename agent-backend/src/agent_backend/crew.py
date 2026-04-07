from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class AgentBackend:
	"""Single-agent backend for Agent 1 preprocessing."""

	agents: List[BaseAgent]
	tasks: List[Task]

	@agent
	def input_preprocessor_agent(self) -> Agent:
		return Agent(
			config=self.agents_config["input_preprocessor_agent"],  # type: ignore[index]
			verbose=True,
		)

	@task
	def preprocess_user_input_task(self) -> Task:
		return Task(
			config=self.tasks_config["preprocess_user_input_task"],  # type: ignore[index]
			output_file="agent1_output.json",
		)

	@crew
	def crew(self) -> Crew:
		return Crew(
			agents=self.agents,
			tasks=self.tasks,
			process=Process.sequential,
			verbose=True,
		)

