"""
coder_agent.py — Code generation specialist agent.

Responsibility: Generate, explain, or debug code for developer queries.
Aware of Walmart API conventions (OAuth 2.0, required headers, pagination).
"""
from backend.config import AgentStep, UserRole
from backend.utils.llm_client import run_coder_agent


class CoderAgent:
    """
    CoderAgent handles requests that require code generation or
    technical implementation guidance.
    """

    name = "CoderAgent"

    def run(self, user_message: str, user_role: UserRole) -> AgentStep:
        persona = user_role.value
        code_response = run_coder_agent(user_message, persona)

        return AgentStep(
            agent=self.name,
            input=user_message,
            output=code_response,
            metadata={
                "persona": persona,
                "output_type": "code_generation",
            },
        )
