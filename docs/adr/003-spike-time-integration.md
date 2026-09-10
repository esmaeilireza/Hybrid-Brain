# ADR-003: Integration method and spike timing

## Status: ✅ Accepted | Date: Week 2

## Decision
- Time-driven with explicit Euler (dt=1ms, stable because dt << τ_m=20ms)
- Exact spike time via **linear interpolation** at threshold-crossing tick:
  t* = t + dt · (v_th − v_prev)/(v_new − v_prev)
- Voltage reset from point t* (not end of tick) — supports fast spiking
- t* stored as spike_frac → direct input to STDP (Week 4)

## Why not Event-driven
Breaks NumPy vectorization + incompatible with Izhikevich (nonlinear) +
event-queue complexity — its advantage (low-activity networks) doesn't fit our profile.

## Consequences
- ✅ Spike time accuracy: sub-ms (instead of 1ms error) → clean STDP
- ✅ Full compatibility with Rust path (Week 12)
- ⚠️ ~10% computational overhead per tick — acceptable

## Amendment (Week 2 validation)
Sub-tick "reset from t*" was replaced by standard reset (v = v_reset at
end of tick). Reason: the t*-reset charged v above v_reset during the
tick remainder, biasing f(I) +4..12% above the analytic model, growing
with current. Spike time interpolation is retained (last_spike_frac)
for STDP. Reference model and implementation now describe the same
dynamics; f-I validation error drops below ~3%.

## Amendment 2 (Week 2, post-validation)
First amendment implementation accidentally dropped the mandatory
voltage reset (v = v_reset) after a spike; consequence: firing rate
pinned at the refractory ceiling (~500 Hz) for all currents. Caught
immediately by test_refractory_respected. Final semantics confirmed:
- Spike TIME: sub-tick precision via linear interpolation (kept for STDP)
- Spike VOLTAGE: reset to v_reset at end of tick
- f-I validation vs analytic theory: error < 5% for I in [15, 40]
  (residual error is the explicit-Euler discretization error)
