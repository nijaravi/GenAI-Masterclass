---
title: Code Review Standards
category: engineering
---

# Code Review Standards

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
