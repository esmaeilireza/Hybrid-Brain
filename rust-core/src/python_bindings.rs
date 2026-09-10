//! PyO3 bindings - gated behind the "python-bindings" feature.
//! Built via: maturin develop --release --features python-bindings
use crate::neurons::LifPopulation;
use crate::synapses::SynapseGroup;
use pyo3::prelude::*;

#[pyclass]
pub struct PyLif {
    inner: LifPopulation,
}

#[pymethods]
impl PyLif {
    #[new]
    fn new(
        n: usize, dt: f64, tau_m: f64, v_rest: f64,
        v_threshold: f64, v_reset: f64, refractory_ms: f64,
    ) -> Self {
        Self {
            inner: LifPopulation::new(
                n, dt, tau_m, v_rest, v_threshold, v_reset, refractory_ms,
            ),
        }
    }

    fn step(&mut self, input: Vec<f64>) -> PyResult<Vec<bool>> {
        if input.len() != self.inner.n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "input length mismatch",
            ));
        }
        Ok(self.inner.step(&input))
    }

    #[getter]
    fn last_spike_frac(&self) -> Vec<f64> {
        self.inner.last_spike_frac.clone()
    }

    #[getter]
    fn v(&self) -> Vec<f64> {
        self.inner.v.clone()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyclass]
pub struct PySyn {
    inner: SynapseGroup,
}

#[pymethods]
impl PySyn {
    #[new]
    #[pyo3(signature = (data, indices, indptr, n_pre, n_post,
                        tau_ms, dt, delay_ticks))]
    fn new(
        data: Vec<f64>, indices: Vec<usize>, indptr: Vec<usize>,
        n_pre: usize, n_post: usize,
        tau_ms: f64, dt: f64, delay_ticks: usize,
    ) -> Self {
        Self {
            inner: SynapseGroup::from_csr(
                data, indices, indptr, n_pre, n_post,
                tau_ms, dt, delay_ticks,
            ),
        }
    }

    fn propagate(&mut self, spikes: Vec<bool>) {
        self.inner.propagate(&spikes);
    }

    fn step(&mut self) -> Vec<f64> {
        self.inner.step().to_vec()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    #[getter]
    fn g(&self) -> Vec<f64> {
        self.inner.g.clone()
    }
}

#[pymodule]
fn hybrid_brain_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLif>()?;
    m.add_class::<PySyn>()?;
    Ok(())
}
