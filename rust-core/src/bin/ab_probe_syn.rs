//! A/B probe for SynapseGroup - mirrors ab_probe.rs line protocol.
use std::io::{self, BufRead, Write};

use hybrid_brain_core::synapses::SynapseGroup;

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout();
    let mut lines = stdin.lock().lines();

    let _mode = lines.next().unwrap().unwrap();

    let hdr: Vec<f64> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let n_pre = hdr[0] as usize;
    let n_post = hdr[1] as usize;
    let tau_ms = hdr[2];
    let dt = hdr[3];
    let delay_ticks = hdr[4] as usize;
    let _nnz = hdr[5] as usize;

    let data: Vec<f64> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let indices: Vec<usize> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let indptr: Vec<usize> = lines.next().unwrap().unwrap()
        .split_whitespace().filter_map(|x| x.parse().ok()).collect();

    let mut syn = SynapseGroup::from_csr(
        data, indices, indptr, n_pre, n_post, tau_ms, dt, delay_ticks);

    writeln!(out, "OK AB").unwrap();
    out.flush().unwrap();

    for line in lines.flatten() {
        let spikes: Vec<bool> = line.chars()
            .map(|c| c == '1').collect();
        syn.propagate(&spikes);
        let g = syn.step();
        let g_str: Vec<String> =
            g.iter().map(|x| format!("{:.17e}", x)).collect();
        writeln!(out, "{}", g_str.join(",")).unwrap();
    }
    out.flush().unwrap();
}