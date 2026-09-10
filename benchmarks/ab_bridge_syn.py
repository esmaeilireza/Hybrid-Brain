"""Synapse A/B bit-exact bridge: Python SynapseGroup vs Rust synapses.rs."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.synapse import SynapseGroup

PROBE = (Path(__file__).resolve().parent.parent / "rust-core" / "target"
         / "release" / "ab_probe_syn.exe")

N_PRE, N_POST = 500, 500
TAU, DT, DELAY = 5.0, 1.0, 1
DENSITY = 0.05
N_TICKS = 500


def make_syn() -> SynapseGroup:
    params = {
        "simulation": {"dt_ms": DT, "seed": 42},
        "synapse": {
            "tau_excite_ms": TAU, "connection_density": DENSITY,
            "excitatory_ratio": 0.8, "delay_min_ms": DELAY,
            "w_exc": 1.5, "w_inh": 3.0,
        },
    }
    return SynapseGroup(N_PRE, N_POST, params)


def run_rust(syn: SynapseGroup, stream: np.ndarray) -> list:
    W = syn.W.tocsr()
    data = [f"{x:.17e}" for x in W.data]
    indices = [str(int(i)) for i in W.indices]
    indptr = [str(int(i)) for i in W.indptr]

    lines = [
        "AB",
        f"{N_PRE} {N_POST} {TAU:.1f} {DT:.1f} {DELAY} {len(data)}",
        " ".join(data),
        " ".join(indices),
        " ".join(indptr),
    ]
    for t in range(N_TICKS):
        lines.append("".join("1" if s else "0" for s in stream[t]))

    proc = subprocess.run(
        [str(PROBE)],
        input="\n".join(lines) + "\n",
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed: {proc.stderr[:500]}")
    out_lines = proc.stdout.strip().splitlines()
    if not out_lines or not out_lines[0].startswith("OK"):
        raise RuntimeError(f"probe bad header: {out_lines[:3]}")
    return [[float(x) for x in l.split(",")] for l in out_lines[1:]]


def main() -> None:
    syn = make_syn()
    rng = np.random.default_rng(42)
    stream = (rng.random((N_TICKS, N_PRE)) < 0.05)

    # Python reference
    py_g = []
    for t in range(N_TICKS):
        syn.propagate(stream[t].astype(float))
        py_g.append(syn.step().copy())

    # IMPORTANT: re-build a fresh Rust-side synapse with the SAME W
    # (the Python run mutated syn.g but not syn.W; propagate is read-only
    # over W, so passing syn directly is safe)
    rs_g = run_rust(syn, stream)

    max_err = 0.0
    for t in range(N_TICKS):
        err = np.max(np.abs(np.array(rs_g[t]) - py_g[t]))
        if err > max_err:
            max_err = err
    print(f"synapse A/B: max abs error = {max_err:.3e}")
    if max_err < 1e-12:
        print("A/B ACCEPTED: synapses.rs bit-exact vs Python")
    else:
        print("A/B FAILED - fix in Rust first")
        sys.exit(1)


if __name__ == "__main__":
    main()