---
title: On-Call and Incident Severity
category: engineering
---

# On-Call and Incident Severity

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
