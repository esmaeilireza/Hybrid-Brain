//! A/B probe: reads a line-based stream from stdin, runs it through
//! LifPopulation, writes results to stdout. Two modes:
//!   AB    - echo per-tick results (for the bit-exact A/B bridge)
//!   BENCH - internal timed loop (for the speed benchmark)
//! Protocol avoids JSON/serde: zero external crates, toolchain-agnostic.
use std::io::{self, BufRead, Write};

use hybrid_brain_core::neurons::LifPopulation;

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout();
    let mut lines = stdin.lock().lines();

    // Line 1: mode ("AB" or "BENCH")
    let mode = match lines.next() {
        Some(Ok(m)) => m,
        _ => {
            writeln!(out, "ERR no mode line").unwrap();
            return;
        }
    };

    // Line 2: params: dt tau_m v_rest v_threshold v_reset refractory_ms n
    let params: Vec<f64> = match lines.next() {
        Some(Ok(l)) => l.split_whitespace().filter_map(|x| x.parse().ok()).collect(),
        _ => {
            writeln!(out, "ERR no params").unwrap();
            return;
        }
    };
    if params.len() < 7 {
        writeln!(out, "ERR params need 7 values").unwrap();
        return;
    }
    let (dt, tau_m, v_rest, v_th, v_reset, refr_ms) =
        (params[0], params[1], params[2], params[3], params[4], params[5]);
    let n = params[6] as usize;

    let mut pop = LifPopulation::new(n, dt, tau_m, v_rest, v_th, v_reset, refr_ms);

    match mode.as_str() {
        "AB" => {
            writeln!(out, "OK AB").unwrap();
            out.flush().unwrap();
            for line in lines.flatten() {
                let input: Vec<f64> =
                    line.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if input.len() != n {
                    continue;
                }
                let mask = pop.step(&input);
                let mask_str: String =
                    mask.iter().map(|&b| if b { '1' } else { '0' }).collect();
                let fracs: Vec<String> = pop
                    .last_spike_frac
                    .iter()
                    .map(|f| {
                        if f.is_nan() {
                            "NaN".to_string()
                        } else {
                            format!("{:.17e}", f)
                        }
                    })
                    .collect();
                writeln!(out, "{} {}", mask_str, fracs.join(",")).unwrap();
            }
            out.flush().unwrap();
        }
        "BENCH" => {
            // Line 3: n_ticks
            let n_ticks: usize = match lines.next() {
                Some(Ok(l)) => l.trim().parse().unwrap_or(0),
                _ => 0,
            };
            let input: Vec<f64> = match lines.next() {
                Some(Ok(l)) => l.split_whitespace().filter_map(|x| x.parse().ok()).collect(),
                _ => vec![0.0; n],
            };
            let start = std::time::Instant::now();
            let mut total_spikes: u64 = 0;
            for _ in 0..n_ticks {
                let mask = pop.step(&input);
                total_spikes += mask.iter().filter(|&&b| b).count() as u64;
            }
            let elapsed = start.elapsed().as_secs_f64();
            writeln!(out, "ELAPSED {:.6} SPIKES {}", elapsed, total_spikes).unwrap();
            out.flush().unwrap();
        }
        _ => {
            writeln!(out, "ERR unknown mode").unwrap();
        }
    }
}
