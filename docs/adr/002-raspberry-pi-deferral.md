# ADR-002: Deferring Raspberry Pi deployment until after Month 8

## Status: ✅ Accepted
## Context
Pi hardware is not available; the final goal is edge deployment.

## Decision
- Software-first development on `profile: desktop.yaml`
- `deploy/pi/` is documentation only and (later) a Dockerfile — no executable code
- `config/profiles/pi.yaml` is kept as a placeholder
- rust-core from Week 12 onward must be cross-compile friendly
  (no allocation in the hot loop, no x86 dependency)

## Consequences
- ✅ Full focus on simulation during the main 8 months
- ✅ Zero risk of dead code
- ⚠️ Engineering constraint: every PR to rust-core must
  not break `cargo build --target aarch64-unknown-linux-gnu`
- ℹ️ Revisit: after hardware purchase (→ ADR-003)
