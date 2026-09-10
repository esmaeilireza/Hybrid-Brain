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
