"""
Jira Ticket Automation Tied to GitHub Code — demo pipeline

Flow (5 touchpoints, called in order from main()):
  1. check_for_new_commit()   -> GitHub API: poll for latest commit SHA
  2. get_local_diff()         -> local git: pull + diff since last processed commit
  3. get_jira_ticket()        -> Jira API: fetch ticket description
  4. check_coverage()         -> OpenAI (gpt-4o-mini): compare ticket vs diff, return JSON
  5. run_semgrep_scan()       -> Semgrep: scan the repo for vulnerabilities
  6. update_jira()            -> Jira API: post coverage + vuln findings as a comment

State: last processed commit SHA is stored in state.json so re-runs only
process new commits.

Run:
    python pipeline.py
"""

import os
import json
import subprocess
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---- config from .env ------------------------------------------------
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
GITHUB_REPO = "nijaravi/jira-github-automation-demo"          # owner/repo
LOCAL_REPO_PATH = "./jira-github-automation-demo"              # local clone path

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
JIRA_EMAIL = os.environ["JIRA_EMAIL"]
JIRA_API_TOKEN = os.environ["JIRA_API_TOKEN"]
JIRA_BASE_URL = os.environ["JIRA_BASE_URL"]
JIRA_ISSUE_KEY = "SCRUM-2"

STATE_FILE = "state.json"

client = OpenAI(api_key=OPENAI_API_KEY)


# ---- 1. GitHub touchpoint ---------------------------------------------
def check_for_new_commit():
    """Poll GitHub API for the latest commit SHA on the default branch."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/commits"
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    resp = requests.get(url, headers=headers, params={"per_page": 1})
    resp.raise_for_status()
    latest_sha = resp.json()[0]["sha"]
    return latest_sha


def load_last_processed_sha():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f).get("last_sha")
    return None


def save_last_processed_sha(sha):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_sha": sha}, f)


# ---- 2. Local git touchpoint ------------------------------------------
def get_local_diff(last_sha, new_sha):
    """Pull latest code locally and return the diff text since last_sha."""
    subprocess.run(["git", "pull"], cwd=LOCAL_REPO_PATH, check=True)

    if last_sha is None:
        # first run — no baseline yet, just summarize current files
        diff_cmd = ["git", "show", new_sha]
    else:
        diff_cmd = ["git", "diff", last_sha, new_sha]

    result = subprocess.run(
        diff_cmd, cwd=LOCAL_REPO_PATH, capture_output=True, text=True, check=True
    )
    return result.stdout


# ---- 3. Jira fetch touchpoint ------------------------------------------
def get_jira_ticket():
    """Fetch the Jira ticket description as plain text."""
    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{JIRA_ISSUE_KEY}"
    resp = requests.get(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    resp.raise_for_status()
    fields = resp.json()["fields"]

    # Jira Cloud stores description in Atlassian Document Format (ADF).
    # For this demo we just pull the plain text nodes out of it.
    desc_adf = fields.get("description")
    text = _extract_text_from_adf(desc_adf) if desc_adf else ""
    return {"summary": fields["summary"], "description": text}


def _extract_text_from_adf(node):
    if isinstance(node, dict):
        if node.get("type") == "text":
            return node.get("text", "")
        return "".join(_extract_text_from_adf(c) for c in node.get("content", []))
    if isinstance(node, list):
        return "".join(_extract_text_from_adf(n) for n in node)
    return ""


# ---- 4. LLM coverage-check touchpoint -----------------------------------
def check_coverage(ticket, diff_text):
    """Ask gpt-4o-mini to compare ticket requirements against the code diff."""
    prompt = f"""You are reviewing whether a code change satisfies a Jira ticket's requirements.

Ticket summary: {ticket['summary']}
Ticket requirements:
{ticket['description']}

Code diff:
{diff_text[:6000]}

Respond ONLY with valid JSON, no markdown fences, in this exact shape:
{{
  "coverage_percent": <integer 0-100>,
  "covered": ["requirement addressed by the diff", ...],
  "pending": ["requirement NOT yet addressed", ...]
}}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


# ---- 5. Semgrep touchpoint ----------------------------------------------
def run_semgrep_scan():
    """Run Semgrep against the local repo and return a short findings summary."""
    result = subprocess.run(
        ["semgrep", "--config", "auto", "--json", "."],
        cwd=LOCAL_REPO_PATH,
        capture_output=True,
        text=True,
    )
    try:
        findings = json.loads(result.stdout).get("results", [])
    except json.JSONDecodeError:
        return "Semgrep scan failed to parse output."

    if not findings:
        return "No vulnerabilities found."

    lines = []
    for f in findings[:10]:  # cap for a readable comment
        lines.append(
            f"- [{f['extra']['severity']}] {f['check_id']} "
            f"({f['path']}:{f['start']['line']}) — {f['extra']['message']}"
        )
    return "\n".join(lines)


# ---- 6. Jira update touchpoint -------------------------------------------
def update_jira(coverage_result, vuln_summary, new_sha):
    """Post the coverage check + vulnerability findings as a Jira comment."""
    comment_text = (
        f"Automated update for commit {new_sha[:7]}\n\n"
        f"Coverage: {coverage_result['coverage_percent']}%\n"
        f"Covered:\n" + "\n".join(f"  - {c}" for c in coverage_result["covered"]) + "\n"
        f"Pending:\n" + "\n".join(f"  - {p}" for p in coverage_result["pending"]) + "\n\n"
        f"Vulnerability scan:\n{vuln_summary}"
    )

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{JIRA_ISSUE_KEY}/comment"
    body = {
        "body": {
            "type": "doc",
            "version": 1,
            "content": [
                {
                    "type": "paragraph",
                    "content": [{"type": "text", "text": comment_text}],
                }
            ],
        }
    }
    resp = requests.post(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN), json=body)
    resp.raise_for_status()
    print(f"Jira comment posted on {JIRA_ISSUE_KEY}")


# ---- main flow -------------------------------------------------------
def main():
    print("Checking GitHub for new commits...")
    latest_sha = check_for_new_commit()
    last_sha = load_last_processed_sha()

    if latest_sha == last_sha:
        print("No new commits since last run. Nothing to do.")
        return

    print(f"New commit found: {latest_sha[:7]} (previous: {last_sha[:7] if last_sha else 'none'})")

    print("Pulling code and computing diff...")
    diff_text = get_local_diff(last_sha, latest_sha)

    print(f"Diff captured: {len(diff_text)} chars")
    print(diff_text[:500])

    print("Fetching Jira ticket...")
    ticket = get_jira_ticket()

    print("Running coverage check with gpt-4o-mini...")
    coverage_result = check_coverage(ticket, diff_text)
    print(json.dumps(coverage_result, indent=2))

    print("Running Semgrep vulnerability scan...")
    vuln_summary = run_semgrep_scan()
    print(vuln_summary)

    print("Updating Jira ticket...")
    update_jira(coverage_result, vuln_summary, latest_sha)

    save_last_processed_sha(latest_sha)
    print("Done.")


if __name__ == "__main__":
    main()