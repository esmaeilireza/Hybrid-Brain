//! LIF population - port of core/neuron.py (Week 2).
//!
//! Contract: bit-exact A/B vs Python. Same update order, same float64
//! ops, same sub-tick interpolation (ADR-003). The A/B harness feeds
//! identical spike streams to both and asserts identical spike masks
//! + last_spike_frac.

/// Leaky Integrate-and-Fire population (vectorized over neurons).
pub struct LifPopulation {
    pub n: usize,
    pub dt: f64,
    pub tau_m: f64,
    pub v_rest: f64,
    pub v_threshold: f64,
    pub v_reset: f64,
    refractory_steps: i32,
    leak: f64,
    pub v: Vec<f64>,
    refractory: Vec<i32>,
    /// Sub-tick fraction of last spike (NaN when no spike this tick).
    pub last_spike_frac: Vec<f64>,
    /// Spike mask of the last tick.
    pub last_spikes: Vec<bool>,
}

impl LifPopulation {
    pub fn new(
        n: usize,
        dt: f64,
        tau_m: f64,
        v_rest: f64,
        v_threshold: f64,
        v_reset: f64,
        refractory_ms: f64,
    ) -> Self {
        Self {
            n,
            dt,
            tau_m,
            v_rest,
            v_threshold,
            v_reset,
            refractory_steps: (refractory_ms / dt).round() as i32,
            leak: dt / tau_m,
            v: vec![v_rest; n],
            refractory: vec![0; n],
            last_spike_frac: vec![f64::NAN; n],
            last_spikes: vec![false; n],
        }
    }

    /// One tick. Returns the spike mask (mirror of Python `step`).
    pub fn step(&mut self, input_current: &[f64]) -> Vec<bool> {
        let n = self.n;
        let v_prev = self.v.clone();

        // Explicit Euler, leak toward v_rest (Week-2 fix), only active cells
        for i in 0..n {
            if self.refractory[i] <= 0 {
                self.v[i] = v_prev[i]
                    + self.leak * (self.v_rest - v_prev[i] + input_current[i]);
            }
        }

        // Threshold + sub-tick linear interpolation (ADR-003)
        let mut spiked = vec![false; n];
        for i in 0..n {
            let active = self.refractory[i] <= 0;
            if active && self.v[i] >= self.v_threshold {
                spiked[i] = true;
                let before = v_prev[i];
                let after = self.v[i];
                let denom = after - before;
                self.last_spike_frac[i] = if before < self.v_threshold && denom != 0.0 {
                    ((self.v_threshold - before) / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                // Standard reset at end of tick (ADR-003 amendment 2)
                self.v[i] = self.v_reset;
                self.refractory[i] = self.refractory_steps;
            } else {
                self.last_spike_frac[i] = f64::NAN;
            }
        }

        // Full refractory for spikers; countdown for the rest
        // (the Week-4 off-by-one fix, mirrored exactly)
        for i in 0..n {
            self.refractory[i] = if spiked[i] {
                self.refractory_steps
            } else {
                (self.refractory[i] - 1).max(0)
            };
        }

        self.last_spikes = spiked.clone();
        spiked
    }

    pub fn reset(&mut self) {
        self.v.iter_mut().for_each(|v| *v = self.v_rest);
        self.refractory.iter_mut().for_each(|r| *r = 0);
        self.last_spike_frac.iter_mut().for_each(|f| *f = f64::NAN);
        self.last_spikes.iter_mut().for_each(|s| *s = false);
    }
}
