from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class Smartdebate():
    """Smartdebate crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def debater(self) -> Agent:
        return Agent(
            config=self.agents_config['debater'], # type: ignore[index]
            verbose=True
        )

    @agent
    def judge(self) -> Agent:
        return Agent(
            config=self.agents_config['judge'], # type: ignore[index]
            verbose=True
        )

    @task
    def propose_task(self) -> Task:
        return Task(config=self.tasks_config['propose_task'])

    @task
    def oppose_task(self) -> Task:
        return Task(config=self.tasks_config['oppose_task'])

    @task
    def decide_task(self) -> Task:
        return Task(config=self.tasks_config['decide_task'])

    @crew
    def crew(self) -> Crew:
        """Creates the Smartdebate crew"""
        # To learn how to add knowledge sources to our crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True
            # process=Process.hierarchical, # In case we wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
