# EEG simulator (W27, ADR-020) - simulated EEG proxy, not a scalp
# forward model (stated verbatim in the paper). Region activity from
# live Cortex6 + Amygdala runs; FFT band power on standard bands.
# Scenario THREAT = intermittent high-drive bursts + dopamine gate
# open (amygdala conditioning, Week-18 mechanism).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from regions.cortex6 import Cortex6
from regions.amygdala import Amygdala

PARAMS = {
    "simulation": {"dt_ms": 1.0},
    "neuron": {"tau_m_ms": 20.0, "v_rest_mV": -65.0,
               "v_threshold_mV": -55.0, "v_reset_mV": -70.0,
               "refractory_ms": 2.0},
    "synapse": {"tau_excite_ms": 5.0, "connection_density": 0.05,
                "excitatory_ratio": 0.8, "delay_min_ms": 1.0,
                "w_exc": 1.5, "w_inh": 3.0},
}

FS = 1000.0   # dt = 1 ms
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0),
         "alpha": (8.0, 13.0), "beta": (13.0, 30.0),
         "gamma": (30.0, 100.0)}
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4",
            "P3", "P4", "O1", "O2"]   # 10-20 montage


class EEGSimulator:
    def __init__(self, seed: int = 0) -> None:
        self.cortex = Cortex6(240, PARAMS, seed=seed)
        self.amgd = Amygdala(50, 400, PARAMS, seed=seed + 1)
        self.rng = np.random.default_rng(seed)

    def step(self, threat: bool):
        drive = np.full(self.cortex.n_in, 30.0)
        if threat and self.rng.random() < 0.10:
            drive = np.full(self.cortex.n_in, 60.0)   # threat burst
        counts = self.cortex.step(drive)
        place = np.zeros(400)
        place[self.rng.integers(0, 400)] = 1.0
        dop = 0.0 if threat else 0.15   # gate open under threat
        asp = self.amgd.step(place, dopamine_level=dop)
        return int(sum(counts)), int(asp.sum())


def run_scenario(sim, threat: bool, n_ticks: int = 3000):
    sig = np.zeros(n_ticks)
    for i in range(n_ticks):
        cx, am = sim.step(threat)
        sig[i] = cx + am * 3.0
    return sig


def band_power(signal, fs: float = FS) -> dict:
    sig = np.asarray(signal) - np.mean(signal)
    f = np.fft.rfftfreq(len(sig), 1.0 / fs)
    power = np.abs(np.fft.rfft(sig)) ** 2
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[name] = float(power[m].sum())
    return out

def eeg_channels(sig, seed: int = 0) -> dict:
    # 10-20 channels: weighted region mix + 1/f-ish electrode noise
    rng = np.random.default_rng(seed)
    weights = np.linspace(0.5, 1.5, len(CHANNELS))
    chans = {}
    for i, name in enumerate(CHANNELS):
        noise = np.cumsum(rng.normal(0.0, 0.02, len(sig)))
        chans[name] = weights[i] * sig + noise
    return chans


def main():
    n = 3000
    print("running REST scenario...")
    sig_rest = run_scenario(EEGSimulator(seed=1), False, n)
    print("running THREAT scenario...")
    sig_threat = run_scenario(EEGSimulator(seed=1), True, n)
    bp_r = band_power(sig_rest)
    bp_t = band_power(sig_threat)
    print(f"{'band':>7} {'rest':>12} {'threat':>12} {'ratio':>7}")
    ratios = {}
    for b in BANDS:
        r = bp_t[b] / max(bp_r[b], 1e-9)
        ratios[b] = r
        print(f"{b:>7} {bp_r[b]:12.1f} {bp_t[b]:12.1f} {r:7.2f}")
    hg = (ratios["beta"] + ratios["gamma"]) / 2.0
    print(f"beta+gamma threat/rest ratio: {hg:.2f} (criterion > 1.2)")
    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(__file__).resolve().parent.parent / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    Pr = np.abs(np.fft.rfft(sig_rest - sig_rest.mean())) ** 2
    Pt = np.abs(np.fft.rfft(sig_threat - sig_threat.mean())) ** 2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(f[1:], Pr[1:], label="rest", alpha=0.8)
    ax.semilogy(f[1:], Pt[1:], label="threat", alpha=0.8)
    ax.set_xlim(0, 100); ax.set_xlabel("Hz"); ax.set_ylabel("power")
    ax.set_title("Simulated EEG power spectrum: rest vs threat")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig6_eeg_spectrum.png",
                                    dpi=300)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(BANDS)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [bp_r[b] for b in names], 0.4, label="rest")
    ax.bar(x + 0.2, [bp_t[b] for b in names], 0.4, label="threat")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("band power"); ax.legend()
    ax.set_title("Band power: rest vs threat")
    fig.tight_layout(); fig.savefig(out / "fig7_band_power.png",
                                    dpi=300)
    plt.close(fig)
    print("figures saved: fig6_eeg_spectrum.png, fig7_band_power.png")
    ok = hg > 1.2
    print("ACCEPTED: threat shifts spectrum to beta/gamma" if ok
          else "FAILED: no spectral shift - tune threat burst profile")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
