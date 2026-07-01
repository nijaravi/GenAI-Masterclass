"""
planner_agent.py — First agent in the pipeline.

Responsibility: Receive the raw user message and produce a structured
intent plan that the Orchestrator can act on.
"""
from backend.config import AgentStep, UserRole
from backend.utils.llm_client import run_planner


class PlannerAgent:
    """
    PlannerAgent analyses the user's raw query and produces a brief
    intent summary used by the Orchestrator for routing decisions.
    """

    name = "PlannerAgent"

    def run(self, user_message: str, user_role: UserRole) -> AgentStep:
        persona = user_role.value
        plan_output = run_planner(user_message, persona)

        return AgentStep(
            agent=self.name,
            input=user_message,
            output=plan_output,
            metadata={
                "persona": persona,
                "message_length": len(user_message),
            },
        )
