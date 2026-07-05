"""
Section 4.7: 'The Coder Agent is not a single LLM call - internally it is
a small CrewAI crew of role-based sub-agents ... LangGraph calls this crew
as if it were any other node - it sends the crew a task and gets back one
structured result.'

Three sub-agents, exactly as specified:
  1. code-retrieval agent   - pulls the relevant diff/PR
  2. requirements-reading   - reads the ticket, extracts what's required
  3. review agent           - compares the two, separately flags vulns

CrewAI is deliberately not used anywhere else in the graph (Section 4.7) -
the Planner and Orchestrator stay plain LangGraph nodes. This module is
the one and only place CrewAI appears.
"""
from __future__ import annotations

from crewai import LLM, Agent, Crew, Process, Task

from app import config

# Toy fixtures standing in for a real PR diff + Jira ticket.
_PR_DIFF = (
    "PR #482: adds retry-with-backoff to the payment webhook handler. "
    "Changes: payment_webhook.py (+42/-6). Adds MAX_RETRIES=3, jittered "
    "backoff, and a dead-letter queue publish on final failure. No new "
    "input validation added on the webhook payload."
)
_TICKET = (
    "JIRA-1190: Payment webhook must retry transient failures up to 3 "
    "times with backoff, and failed events must be recoverable (not "
    "silently dropped). Acceptance criteria: (1) retry logic present, "
    "(2) failed events land in a durable queue, (3) no regression to "
    "existing payload validation."
)


def build_crew() -> Crew:
    llm = LLM(model=f"openai/{config.OPENAI_MODEL}", api_key=config.OPENAI_API_KEY)

    retrieval_agent = Agent(
        role="Code Retrieval Specialist",
        goal="Pull the relevant diff/PR for the request under review",
        backstory="You fetch exactly the code change under discussion, nothing more.",
        llm=llm,
    )
    requirements_agent = Agent(
        role="Requirements Reader",
        goal="Read the ticket and extract precisely what is required",
        backstory="You turn a ticket into a short, unambiguous checklist.",
        llm=llm,
    )
    review_agent = Agent(
        role="Coverage & Vulnerability Reviewer",
        goal="Compare the diff against requirements and separately flag security concerns",
        backstory="You never conflate 'meets requirements' with 'is secure' - you report both independently.",
        llm=llm,
    )

    retrieval_task = Task(
        description=f"Summarize the code change under review.\n\n{_PR_DIFF}",
        expected_output="A short factual summary of what the diff changes.",
        agent=retrieval_agent,
    )
    requirements_task = Task(
        description=f"Extract the acceptance criteria from this ticket.\n\n{_TICKET}",
        expected_output="A numbered list of acceptance criteria.",
        agent=requirements_agent,
    )
    review_task = Task(
        description="Compare the diff summary against the acceptance criteria, state "
                    "which are met, and separately list any security concerns.",
        expected_output="Coverage verdict per criterion, plus a 'Security concerns' section.",
        agent=review_agent,
        context=[retrieval_task, requirements_task],
    )

    return Crew(
        agents=[retrieval_agent, requirements_agent, review_agent],
        tasks=[retrieval_task, requirements_task, review_task],
        process=Process.sequential,
        verbose=False,
    )


def run_crew() -> str:
    """Builds and runs the crew for real, against config.OPENAI_MODEL.
    Sync/blocking - callers should run this in a thread (see
    app/nodes/specialists.py's coder_agent_node)."""
    return str(build_crew().kickoff())
