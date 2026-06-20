---
title: Data Classification Standard
category: security
---

# Data Classification Standard

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
