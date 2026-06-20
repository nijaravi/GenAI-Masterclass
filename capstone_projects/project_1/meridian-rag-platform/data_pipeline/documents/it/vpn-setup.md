---
title: VPN Access and Setup Guide
category: it
---

# VPN Access and Setup Guide

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
