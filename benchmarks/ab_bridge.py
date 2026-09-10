"""A/B bit-exact bridge: Python LIF vs Rust neurons.rs.

Generates a seeded input stream, runs it through BOTH cores, and
diffs. Acceptance: spike masks identical on every tick; fracs equal
within 1e-12 (f64 round-trip through text). Any mismatch = Rust
semantics drift, fixed in Rust first (never Python)."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.neuron import LifPopulation

PROBE = Path(__file__).resolve().parent.parent / "rust-core" / "target" / "release" / "ab_probe.exe"

PARAMS = {
    "dt": 1.0, "tau_m": 20.0, "v_rest": -65.0,
    "v_threshold": -55.0, "v_reset": -70.0, "refractory_ms": 2.0,
}
N_NEURONS = 100
N_TICKS = 1000


def make_stream() -> np.ndarray:
    rng = np.random.default_rng(42)
    # mixed regime: quiet, moderate, strong drive across neurons
    base = rng.uniform(0.0, 30.0, N_NEURONS)
    wobble = rng.normal(0, 3.0, (N_TICKS, N_NEURONS))
    return np.clip(base[None, :] + wobble, 0.0, 60.0)


def run_python(stream: np.ndarray) -> tuple[list, list]:
    p = PARAMS
    pop = LifPopulation(N_NEURONS, {
        "simulation": {"dt_ms": p["dt"]},
        "neuron": {
            "tau_m_ms": p["tau_m"], "v_rest_mV": p["v_rest"],
            "v_threshold_mV": p["v_threshold"], "v_reset_mV": p["v_reset"],
            "refractory_ms": p["refractory_ms"],
        },
    })
    masks, fracs = [], []
    for t in range(N_TICKS):
        mask = pop.step(stream[t])
        masks.append(mask.copy())
        fracs.append(pop.last_spike_frac.copy())
    return masks, fracs


def run_rust(stream: np.ndarray) -> tuple[list, list]:
    params_line = " ".join(str(PARAMS[k]) for k in
                           ("dt", "tau_m", "v_rest", "v_threshold",
                            "v_reset", "refractory_ms")) + f" {N_NEURONS}"
    stdin_lines = ["AB", params_line]
    for t in range(N_TICKS):
        stdin_lines.append(" ".join(f"{x:.17e}" for x in stream[t]))

    proc = subprocess.run(
        [str(PROBE)],
        input="\n".join(stdin_lines) + "\n",
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed: {proc.stderr}")
    out_lines = proc.stdout.strip().splitlines()
    if not out_lines or not out_lines[0].startswith("OK"):
        raise RuntimeError(f"probe bad header: {out_lines[:3]}")

    masks, fracs = [], []
    for line in out_lines[1:]:
        parts = line.split(" ", 1)
        masks.append([c == "1" for c in parts[0]])
        fracs.append([float("nan") if v == "NaN" else float(v)
                      for v in parts[1].split(",")])
    return masks, fracs


def main() -> None:
    stream = make_stream()
    print(f"Generating {N_TICKS} ticks x {N_NEURONS} neurons...")

    py_masks, py_fracs = run_python(stream)
    print("Python reference done.")

    rs_masks, rs_fracs = run_rust(stream)
    print("Rust probe done.")

    mask_mismatches = 0
    frac_max_err = 0.0
    for t in range(N_TICKS):
        for i in range(N_NEURONS):
            if py_masks[t][i] != rs_masks[t][i]:
                mask_mismatches += 1
                if mask_mismatches <= 5:
                    print(f"  MASK MISMATCH tick={t} neuron={i} "
                          f"py={py_masks[t][i]} rs={rs_masks[t][i]}")
            a, b = py_fracs[t][i], rs_fracs[t][i]
            if not (np.isnan(a) and np.isnan(b)):
                err = abs(a - b)
                if err > frac_max_err:
                    frac_max_err = err

    print(f"\nmask mismatches: {mask_mismatches} / {N_TICKS * N_NEURONS}")
    print(f"max frac abs error: {frac_max_err:.3e}")

    if mask_mismatches == 0 and frac_max_err < 1e-12:
        print("A/B ACCEPTED: Rust core is bit-exact vs Python")
    else:
        print("A/B FAILED: semantics drift detected - fix in Rust first")
        sys.exit(1)


if __name__ == "__main__":
    main()
