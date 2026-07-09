# Project: Jira Ticket Automation Tied to GitHub Code

## What this project is

A demo automation pipeline built for an interview-prep capstone session (mentee: Abdul,
transitioning from big data admin to GenAI engineering). This is NOT production code —
it's a deliberately simple, single-file pipeline meant to be explained end-to-end in a
live session, with every touchpoint visible and traceable.

The use case: instead of a developer manually comparing GitHub code against a Jira
ticket's requirements and updating status by hand, this pipeline does it automatically:
it reads the actual pushed code, compares it against the ticket's stated requirements,
writes a coverage update back to Jira, and runs a vulnerability scan as part of the
same pass.

## Working directory layout

```
project_4/
├── .venv/                          # Python 3.11 virtualenv
├── jira-github-automation-demo/    # cloned GitHub repo (the "target" repo being watched)
│   ├── .gitignore
│   ├── app.py                      # sample Flask login route — deliberately partial
│   │                                #   (see "Sample app.py intent" below)
│   └── README.md
├── .env                             # credentials — see "Environment variables" below
├── pipeline.py                      # THE pipeline — single file, linear flow
└── state.json                       # tracks last processed commit SHA between runs
```

`jira-github-automation-demo` is a real GitHub repo (github.com/nijaravi/jira-github-automation-demo)
cloned locally. Commits get pushed to it from this same local clone during the demo to
simulate a developer pushing code against a Jira ticket.

## The five touchpoints (the actual flow, in order)

`pipeline.py` runs these steps top to bottom inside `main()`. No framework, no classes —
plain functions called in sequence, on purpose, so each step can be pointed at and
explained individually in the session.

1. **`check_for_new_commit()`** — GitHub REST API. Polls for the latest commit SHA on
   the default branch of `jira-github-automation-demo`.
2. **`get_local_diff()`** — local git. Runs `git pull` in the cloned repo, then
   `git diff <last_sha> <new_sha>` (or `git show <new_sha>` on the very first run, when
   there's no prior SHA to diff against) to get the actual code change as text.
3. **`get_jira_ticket()`** — Jira Cloud REST API (`/rest/api/3/issue/{key}`). Fetches
   the ticket's summary + description. Jira Cloud stores descriptions in Atlassian
   Document Format (ADF), so there's a small recursive `_extract_text_from_adf()`
   helper that flattens ADF into plain text.
4. **`check_coverage()`** — OpenAI `gpt-4o-mini`. Sends the ticket requirements + diff
   text, asks for strict JSON back: `coverage_percent`, `covered` (list), `pending`
   (list). Temperature 0 for consistency.
5. **`run_semgrep_scan()`** — runs `semgrep --config auto --json .` as a subprocess
   against the cloned repo, parses findings, formats the top 10 into a short readable
   list (severity, rule id, file:line, message).
6. **`update_jira()`** — posts a single comment back to the Jira ticket containing the
   coverage results + the vulnerability summary, using Jira's ADF comment format.

State (`state.json`) stores the last processed commit SHA so re-runs only process new
commits — this is a simple polling model, not a live webhook (webhook + ngrok was
tested earlier and works, but polling was chosen for the demo since it needs no public
URL / tunnel dependency).

## The Jira ticket being tracked

- Site: `https://nijanthan572.atlassian.net`
- Project key: `SCRUM`
- Issue: `SCRUM-2`
- Description (the actual requirements the code is checked against):
  > Implement input validation for the login form. Add rate limiting on login
  > attempts. Add unit tests for both.

## Sample app.py intent

`jira-github-automation-demo/app.py` is a small Flask login route **deliberately
built to be partially compliant and to contain vulnerabilities**, so the demo has
something real to show at every step instead of an all-clean/all-covered run:

- Input validation: only checks fields are non-empty — NOT real validation
  (no format/length/injection-character checks). This is intentionally weak so
  there's a legitimate judgment call for the LLM to make (does "checks for empty
  string" count as "input validation"? Arguably not.).
- Rate limiting: not implemented at all (intentionally missing).
- Unit tests: not included (intentionally missing).
- Deliberate vulnerability #1: hardcoded secret (`app.secret_key = "..."`) —
  should trigger a hardcoded-credential Semgrep rule.
- Deliberate vulnerability #2: SQL query built via f-string instead of a
  parameterized query — should trigger SQL-injection Semgrep rules.

## Environment variables (`.env`, gitignored)

```
GITHUB_TOKEN=<GitHub fine-grained PAT, scoped to the demo repo, Contents:Read + PRs:Read>
OPENAI_API_KEY=<OpenAI key with gpt-4o-mini access>
JIRA_EMAIL=<Atlassian account email>
JIRA_API_TOKEN=<Jira Cloud API token, Basic Auth alongside JIRA_EMAIL>
JIRA_BASE_URL=https://nijanthan572.atlassian.net
```

`.env` is confirmed gitignored. If Claude Code is asked to inspect or modify secrets
handling, do not print raw key values back into chat or logs.

## Current known issue being debugged

Symptom: after adding `app.py` and pushing, running `pipeline.py` produces:

```
coverage_percent: 0
covered: []
pending: [all three requirements]
```

This may be a genuine bug (diff not reaching the LLM — e.g. `last_sha` in
`state.json` not reachable in local git history, or `LOCAL_REPO_PATH` pointing at a
stale/wrong clone) **or** it may be a defensible strict judgment by the LLM (the
input validation in `app.py` is intentionally weak, so 0% coverage could be correct).

A debug print was added right before the `check_coverage()` call in `main()`:
```python
print(f"Diff captured: {len(diff_text)} chars")
print(diff_text[:500])
```
Next diagnostic step: run `pipeline.py` and confirm whether the diff text is actually
empty (real bug — check `git log --oneline` in `jira-github-automation-demo` to
confirm both the stored `last_sha` and the new commit SHA are present in local
history) or non-empty (in which case 0% is likely a legitimate LLM judgment, and the
fix — if wanted — is to loosen `check_coverage()`'s prompt to support a
`partially_covered` bucket rather than a binary covered/pending split).

## Working style / conventions for this session

- Keep `pipeline.py` as a single flat file with plain functions — no classes, no
  LangGraph/CrewAI framework layered on top. The point of this demo is transparency
  of each touchpoint, not architectural sophistication.
- Don't introduce config abstraction layers, dependency injection, or multi-file
  restructuring unless explicitly asked.
- This is a portfolio/interview-prep artifact for Abdul — favor code he can read
  top-to-bottom and defend in an interview over "production-grade" patterns.
- Terse, direct communication preferred. Fix the actual bug; don't pad explanations.