"""
Section 4.7: 'The Coder Agent is not a single LLM call - internally it is a
small CrewAI crew of role-based sub-agents ... LangGraph calls this crew as
if it were any other node - it sends the crew a task and gets back one
structured result.'

Three sub-agents, exactly as specified:
  1. code-retrieval agent   - pulls the relevant diff/PR
  2. requirements-reading   - reads the ticket, extracts what's required
  3. review agent           - compares the two, separately flags vulns

CrewAI is deliberately not used anywhere else in the graph (Section 4.7) -
the Planner and Orchestrator stay plain LangGraph nodes. This module is the
one and only place CrewAI appears.

Two entry points:
  build_and_run_crew()  - real CrewAI kickoff() against a live LLM. Used
                           when config.LIVE_LLM is true.
  run_demo_crew()       - exercises the exact same three-role structure
                           (same toy PR/ticket fixtures, same sequential
                           hand-off) but drives each step with the
                           deterministic DemoChatModel instead of a real
                           LLM call, since CrewAI's kickoff() needs a real
                           model backend to talk to. This keeps the demo
                           network-free while still being a faithful,
                           inspectable stand-in for the same three-role
                           division of labor.
"""
from __future__ import annotations

from app import config

# ── toy fixtures standing in for a real PR diff + Jira ticket ─────────────
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


def _build_agents_and_tasks():
    from crewai import Agent, Task

    retrieval_agent = Agent(
        role="Code Retrieval Specialist",
        goal="Pull the relevant diff/PR for the request under review",
        backstory="You fetch exactly the code change under discussion, "
                   "nothing more, so downstream agents work from ground truth.",
        verbose=False,
    )
    requirements_agent = Agent(
        role="Requirements Reader",
        goal="Read the ticket and extract precisely what is required",
        backstory="You turn a ticket into a short, unambiguous checklist "
                   "of acceptance criteria.",
        verbose=False,
    )
    review_agent = Agent(
        role="Coverage & Vulnerability Reviewer",
        goal="Compare the diff against requirements and separately flag security concerns",
        backstory="You are a careful reviewer who never conflates "
                   "'meets requirements' with 'is secure' - you report both independently.",
        verbose=False,
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
        description="Compare the diff summary against the acceptance criteria. "
                    "State which criteria are met, which are not, and separately "
                    "list any security/vulnerability concerns you notice in the diff "
                    "(these are not the same thing as unmet requirements).",
        expected_output="Coverage verdict per criterion, plus a separate "
                        "'Security concerns' section.",
        agent=review_agent,
        context=[retrieval_task, requirements_task],
    )

    return [retrieval_agent, requirements_agent, review_agent], [
        retrieval_task,
        requirements_task,
        review_task,
    ]


def build_and_run_crew() -> str:
    """Live path: real CrewAI kickoff() against config.OPENAI_MODEL."""
    from crewai import Crew, Process, LLM

    agents, tasks = _build_agents_and_tasks()
    llm = LLM(model=f"openai/{config.OPENAI_MODEL}", api_key=config.OPENAI_API_KEY)
    for agent in agents:
        agent.llm = llm

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def run_demo_crew() -> str:
    """Offline path: same three-role structure, driven by DemoChatModel so
    the demo needs no network access or API key."""
    from app.llm import get_chat_model

    model = get_chat_model()

    diff_summary = model.invoke(
        f"[Code Retrieval Specialist] Summarize this diff:\n{_PR_DIFF}"
    )
    if diff_summary == "(demo model) acknowledged.":
        diff_summary = (
            "PR #482 adds retry-with-backoff (3 retries, jitter) to the payment "
            "webhook handler and publishes exhausted failures to a dead-letter "
            "queue. No changes were made to payload validation."
        )

    criteria = model.invoke(
        f"[Requirements Reader] Extract acceptance criteria from:\n{_TICKET}"
    )
    if criteria == "(demo model) acknowledged.":
        criteria = (
            "1. Retry logic with backoff, up to 3 attempts.\n"
            "2. Failed events must be durably recoverable, not dropped.\n"
            "3. No regression to existing payload validation."
        )

    review = (
        "Coverage vs. JIRA-1190:\n"
        "  1. Retry with backoff (3 attempts) - MET (MAX_RETRIES=3, jittered backoff).\n"
        "  2. Failed events durably recoverable - MET (dead-letter queue publish on final failure).\n"
        "  3. No regression to payload validation - MET (diff makes no changes to validation code).\n\n"
        "Security concerns (tracked separately from coverage, per review policy):\n"
        "  - The diff adds no new input validation on the webhook payload. This isn't "
        "a regression, but if the payload schema hasn't been hardened elsewhere, "
        "repeated automatic retries could amplify an attempted payload-injection attack "
        "up to 3x before landing in the dead-letter queue. Recommend confirming payload "
        "signature verification happens before the retry loop, not after."
    )

    return (
        f"[1/3 Code Retrieval]\n{diff_summary}\n\n"
        f"[2/3 Requirements]\n{criteria}\n\n"
        f"[3/3 Review]\n{review}"
    )
