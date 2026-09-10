//! Synapse propagation - port of core/synapse.py hot loop (ADR-008
//! Option A: propagate/step only; STDP stays in Python for now).
//!
//! Contract: bit-exact A/B vs Python SynapseGroup.propagate/step.
//! Same CSR sparse representation, same decay, same delay ring buffer.

pub struct SynapseGroup {
    pub n_pre: usize,
    pub n_post: usize,
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    pub tau_ms: f64,
    pub dt: f64,
    decay: f64,
    pub g: Vec<f64>,
    delay_ticks: usize,
    pending: Vec<Vec<f64>>,
}

impl SynapseGroup {
    pub fn from_csr(
        data: Vec<f64>, indices: Vec<usize>, indptr: Vec<usize>,
        n_pre: usize, n_post: usize,
        tau_ms: f64, dt: f64, delay_ticks: usize,
    ) -> Self {
        Self {
            n_pre, n_post, data, indices, indptr,
            tau_ms, dt,
            decay: 1.0 - dt / tau_ms,
            g: vec![0.0; n_post],
            delay_ticks: delay_ticks.max(1),
            pending: vec![vec![0.0; n_post]; delay_ticks.max(1)],
        }
    }

    pub fn propagate(&mut self, spikes_pre: &[bool]) {
        let buf = &mut self.pending[self.delay_ticks - 1];
        for post in 0..self.n_post {
            let start = self.indptr[post];
            let end = self.indptr[post + 1];
            let mut acc = 0.0;
            for k in start..end {
                let pre = self.indices[k];
                if spikes_pre[pre] {
                    acc += self.data[k];
                }
            }
            buf[post] += acc;
        }
    }

    pub fn step(&mut self) -> &[f64] {
        let released = self.pending.remove(0);
        self.pending.push(vec![0.0; self.n_post]);
        for i in 0..self.n_post {
            self.g[i] = self.g[i] * self.decay + released[i];
        }
        &self.g
    }

    pub fn reset(&mut self) {
        self.g.iter_mut().for_each(|x| *x = 0.0);
        for buf in self.pending.iter_mut() {
            buf.iter_mut().for_each(|x| *x = 0.0);
        }
    }
}