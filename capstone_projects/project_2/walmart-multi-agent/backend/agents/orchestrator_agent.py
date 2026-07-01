"""
orchestrator_agent.py — Core routing brain of the multi-agent system.

Responsibility: Read the Planner's intent output and decide which
specialist agent or tool should handle the request.

Routing destinations:
  1. RAG Agent        — factual Q&A, document retrieval
  2. Tool-Call Agent  — calculations, structured lookups
  3. Coder Agent      — code generation / explanation
  4. MCP Agent        — external system integration via MCP servers
"""
from backend.config import AgentStep, RouteDecision, UserRole
from backend.utils.llm_client import run_orchestrator_routing


ROUTE_RATIONALE = {
    RouteDecision.RAG: (
        "Query requires retrieval from the vector knowledge base. "
        "Routing to RAG Agent for semantic search + synthesis."
    ),
    RouteDecision.TOOL_CALL: (
        "Query requires computation or structured data lookup. "
        "Routing to Tool-Call Agent for deterministic execution."
    ),
    RouteDecision.CODER: (
        "Query requires code generation or technical explanation. "
        "Routing to Coder Agent."
    ),
    RouteDecision.MCP: (
        "Query requires integration with an external system. "
        "Routing to MCP Agent for protocol-based tool invocation."
    ),
}


class OrchestratorAgent:
    """
    OrchestratorAgent takes the Planner's analysis and makes a
    deterministic routing decision about which specialist to invoke.
    """

    name = "OrchestratorAgent"

    def run(
        self,
        planner_step: AgentStep,
        original_message: str,
        user_role: UserRole,
    ) -> tuple[AgentStep, RouteDecision]:
        persona = user_role.value
        route = run_orchestrator_routing(
            planner_output=planner_step.output,
            user_message=original_message,
            persona=persona,
        )
        rationale = ROUTE_RATIONALE[route]

        step = AgentStep(
            agent=self.name,
            input=planner_step.output,
            output=f"Route decision: {route.value.upper()}. {rationale}",
            metadata={
                "route": route.value,
                "persona": persona,
            },
        )
        return step, route
