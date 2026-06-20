---
title: Service Deployment Runbook
category: engineering
---

# Service Deployment Runbook

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
