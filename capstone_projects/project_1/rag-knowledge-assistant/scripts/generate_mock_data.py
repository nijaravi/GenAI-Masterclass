"""Generate a mock internal knowledge base for Meridian Corp (fictional).

Produces markdown docs across four categories (hr, it, security, engineering)
with front-matter, plus a gold eval set. The content is intentionally specific —
real policy numbers, error codes, thresholds — so retrieval and faithfulness
scoring have concrete facts to hit or miss.

Run:  python scripts/generate_mock_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

RAW = Path("data/raw")
EVAL = Path("data/eval")

DOCS: list[dict] = [
    {
        "path": "hr/leave-policy.md",
        "title": "Annual Leave & Time-Off Policy",
        "category": "hr",
        "body": """# Annual Leave & Time-Off Policy

## Annual leave entitlement
Full-time employees accrue **25 days** of paid annual leave per calendar year,
accruing at 2.08 days per month. Employees in their first year accrue pro-rata
from their start date. Up to **5 days** may be carried over into the next year
and must be used by 31 March or they are forfeited.

## Requesting leave
Leave requests are submitted in Workday at least **10 business days** in advance
for any block longer than 3 days. Your line manager approves or declines within
3 business days. Policy reference: HR-LV-2024-03.

## Sick leave
Employees receive 10 paid sick days per year. A doctor's note is required for any
absence longer than **3 consecutive days**. Sick leave does not carry over.

## Parental leave
Primary caregivers are entitled to 16 weeks of paid parental leave; secondary
caregivers to 6 weeks. Requests must be filed at least 30 days before the
expected start date.
""",
    },
    {
        "path": "hr/expenses.md",
        "title": "Expense Reimbursement Policy",
        "category": "hr",
        "body": """# Expense Reimbursement Policy

## Submission window
Expenses must be submitted within **60 days** of being incurred. Claims older
than 60 days require VP approval. Policy reference: HR-EX-2023-11.

## Daily meal allowance
The per-diem meal allowance is **$75** for domestic travel and **$110** for
international travel. Alcohol is not reimbursable.

## Approval thresholds
Expenses under $500 are auto-approved by your manager. Expenses between $500 and
$5,000 require director approval. Anything above $5,000 requires finance review
and a purchase order raised in advance.
""",
    },
    {
        "path": "it/vpn-setup.md",
        "title": "VPN Access and Setup Guide",
        "category": "it",
        "body": """# VPN Access and Setup Guide

## Overview
Meridian uses the GlobalProtect VPN client for remote access. All access to
internal systems from outside the office network requires the VPN plus an active
MFA session.

## Setup steps
1. Install GlobalProtect from the Self-Service portal.
2. Set the portal address to `vpn.meridian.internal`.
3. Sign in with your SSO credentials and approve the MFA push.

## Common error: GP-1107
Error code **GP-1107** means your device certificate has expired. Resolve it by
opening Self-Service, running "Renew Device Certificate", then reconnecting. If
it persists, the certificate authority sync may be delayed up to 4 hours.

## Split tunneling
Split tunneling is disabled by policy: all traffic routes through the VPN while
connected. Contact the Service Desk for exceptions (ticket queue: NET-ACCESS).
""",
    },
    {
        "path": "it/password-reset.md",
        "title": "Password Reset and Account Lockout",
        "category": "it",
        "body": """# Password Reset and Account Lockout

## Self-service reset
Reset your password at `mypassword.meridian.internal`. Passwords must be at least
**14 characters** with one uppercase, one number, and one symbol, and cannot
match any of your previous 10 passwords.

## Lockout policy
Accounts lock after **5 failed sign-in attempts** within 15 minutes. A locked
account auto-unlocks after 30 minutes, or immediately via the Service Desk after
identity verification.

## Password rotation
Standard accounts rotate every 365 days. Privileged/admin accounts rotate every
90 days and require a hardware security key.
""",
    },
    {
        "path": "it/laptop-refresh.md",
        "title": "Laptop Refresh and Hardware Requests",
        "category": "it",
        "body": """# Laptop Refresh and Hardware Requests

## Refresh cycle
Standard laptops are refreshed every **3 years**. Engineering workstations are
refreshed every 2 years due to heavier workloads. Eligibility is visible in the
Asset portal.

## Requesting hardware
Raise a request in the IT Service Catalog under "Hardware". Standard configs ship
in 3-5 business days. Non-standard configs (e.g. 64GB RAM, dedicated GPU) require
manager approval and a business justification.

## Damaged or lost devices
Report lost or stolen devices immediately to the Service Desk so the device can
be remotely wiped. A police report is required for stolen-device insurance claims
over $1,000.
""",
    },
    {
        "path": "security/data-classification.md",
        "title": "Data Classification Standard",
        "category": "security",
        "body": """# Data Classification Standard

## Tiers
Meridian classifies data into four tiers: **Public**, **Internal**,
**Confidential**, and **Restricted**. Reference: SEC-DC-2024-01.

- Public: approved for external release.
- Internal: default for day-to-day business data.
- Confidential: customer PII, contracts, financials. Encryption at rest required.
- Restricted: secrets, credentials, regulated health/payment data. Access is
  logged and reviewed quarterly.

## Handling Restricted data
Restricted data must never be copied to personal devices, pasted into external
tools, or sent over unencrypted channels. Storage must use the approved KMS with
AES-256 encryption.

## Reporting a data incident
Suspected data leaks must be reported to security@meridian.internal within
**1 hour** of discovery. The incident response team triages within 30 minutes.
""",
    },
    {
        "path": "security/acceptable-use.md",
        "title": "Acceptable Use Policy",
        "category": "security",
        "body": """# Acceptable Use Policy

## Approved tools
Only software from the approved catalog may be installed on company devices.
Generative-AI tools are permitted **only** through the company-sanctioned
gateway; pasting Confidential or Restricted data into public AI tools is
prohibited. Reference: SEC-AUP-2024-02.

## Email and phishing
Report suspected phishing using the "Report Phish" button in Outlook. Never
enter SSO credentials on a page reached via an email link.

## Personal use
Incidental personal use of company devices is allowed within reason. Storing
personal media libraries or running personal servers is not permitted.
""",
    },
    {
        "path": "engineering/deployment-runbook.md",
        "title": "Service Deployment Runbook",
        "category": "engineering",
        "body": """# Service Deployment Runbook

## Pipeline
Deployments run through GitHub Actions. A merge to `main` triggers build, test,
and a push to the staging environment. Production deploys require a manual
approval gate and at least **two** reviewer approvals on the release PR.

## Rollback
To roll back, re-run the previous successful release workflow or run
`make rollback ENV=prod TAG=<previous-tag>`. Rollbacks should complete within
**5 minutes**; if a rollback exceeds the SLA, page the on-call via PagerDuty.

## Health checks
Every service must expose `/healthz`. The orchestrator marks a pod unhealthy
after 3 consecutive failed checks (10s interval) and restarts it.

## Canary
Production releases roll out as a 10% canary for 15 minutes. If error rate
exceeds **2%** during canary, the deploy auto-aborts.
""",
    },
    {
        "path": "engineering/oncall.md",
        "title": "On-Call and Incident Severity",
        "category": "engineering",
        "body": """# On-Call and Incident Severity

## Rotation
On-call rotates weekly, Monday 10:00 to the following Monday 10:00. Primary
on-call must acknowledge pages within **5 minutes**; secondary is paged if the
primary does not ack within 10 minutes.

## Severity levels
- **SEV1**: full outage or data loss. Page immediately, open a war room, exec
  notification within 15 minutes.
- **SEV2**: major feature degraded, no workaround. Respond within 30 minutes.
- **SEV3**: minor or cosmetic. Handle during business hours.

## Postmortems
Every SEV1 and SEV2 requires a blameless postmortem within **5 business days**,
documented in the incident tracker with action items and owners.
""",
    },
    {
        "path": "engineering/code-review.md",
        "title": "Code Review Standards",
        "category": "engineering",
        "body": """# Code Review Standards

## Requirements
Every change to `main` requires review. PRs should be under **400 lines** of
diff where practical; larger changes should be split. At least one reviewer must
be a code owner of the affected area.

## What reviewers check
Correctness, test coverage for new logic, security (no secrets in code, input
validation), and observability (are new code paths logged/metered?).

## SLAs
Reviewers should respond within **one business day**. If a PR blocks a release,
tag it `priority-review` and notify the team channel.
""",
    },
]

EVAL_SET = [
    {"question": "How many days of annual leave do full-time employees get?",
     "relevant_sources": ["hr/leave-policy.md"],
     "expected_fact": "25 days"},
    {"question": "How much annual leave can I carry over to next year?",
     "relevant_sources": ["hr/leave-policy.md"],
     "expected_fact": "5 days, must be used by 31 March"},
    {"question": "What does VPN error GP-1107 mean and how do I fix it?",
     "relevant_sources": ["it/vpn-setup.md"],
     "expected_fact": "expired device certificate; renew via Self-Service"},
    {"question": "How many failed logins before my account locks?",
     "relevant_sources": ["it/password-reset.md"],
     "expected_fact": "5 failed attempts in 15 minutes"},
    {"question": "What is the per-diem meal allowance for international travel?",
     "relevant_sources": ["hr/expenses.md"],
     "expected_fact": "$110"},
    {"question": "How quickly must I report a suspected data leak?",
     "relevant_sources": ["security/data-classification.md"],
     "expected_fact": "within 1 hour"},
    {"question": "Can I paste confidential data into public AI tools?",
     "relevant_sources": ["security/acceptable-use.md"],
     "expected_fact": "no; only via sanctioned gateway"},
    {"question": "How many approvals are needed for a production deploy?",
     "relevant_sources": ["engineering/deployment-runbook.md"],
     "expected_fact": "two reviewer approvals plus manual gate"},
    {"question": "What error rate aborts a canary deploy?",
     "relevant_sources": ["engineering/deployment-runbook.md"],
     "expected_fact": "2%"},
    {"question": "How fast must primary on-call acknowledge a page?",
     "relevant_sources": ["engineering/oncall.md"],
     "expected_fact": "5 minutes"},
    {"question": "What's the password minimum length?",
     "relevant_sources": ["it/password-reset.md"],
     "expected_fact": "14 characters"},
    {"question": "When does a SEV1 require exec notification?",
     "relevant_sources": ["engineering/oncall.md"],
     "expected_fact": "within 15 minutes"},
]


def main() -> None:
    for doc in DOCS:
        p = RAW / doc["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        fm = f"---\ntitle: {doc['title']}\ncategory: {doc['category']}\n---\n\n"
        p.write_text(fm + doc["body"], encoding="utf-8")
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "eval_set.jsonl").write_text(
        "\n".join(json.dumps(x) for x in EVAL_SET), encoding="utf-8"
    )
    print(f"Wrote {len(DOCS)} documents to {RAW} and {len(EVAL_SET)} eval items.")


if __name__ == "__main__":
    main()
