//! Ad-hoc profiling harness for `solve()` at realistic movie-frame size
//! (59x150, matching `012_20250717_beta_gal_6nM.mp`). Not part of the test
//! suite — run manually with `cargo run --release --example bench_solve`.
//! Prints wall time for BaselineFrame::new (prefilter) vs the DE+polish
//! search, plus an objective-call counter, over many (seeded) frames to
//! approximate the per-frame cost the real pipeline pays.

use std::cell::Cell;
use std::time::Instant;

use drift_correction::optimize::{differential_evolution, Bound, DeConfig};
use drift_correction::solve::BaselineFrame;
use drift_correction::spline::shift2d;

fn textured_frame(rows: usize, cols: usize) -> Vec<f64> {
    let mut base = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let v = ((r as f64) * 0.31 + (c as f64) * 0.17).sin()
                + ((r as f64) * 0.11 - (c as f64) * 0.23).cos() * 0.5;
            base[r * cols + c] = v;
        }
    }
    let bumps: [(f64, f64, f64); 6] = [
        (12.0, 14.0, 8.0),
        (28.0, 22.0, -6.0),
        (20.0, 30.0, 5.0),
        (40.0, 10.0, 4.0),
        (5.0, 40.0, -3.0),
        (45.0, 45.0, 6.0),
    ];
    for r in 0..rows {
        for c in 0..cols {
            let mut extra = 0.0;
            for &(cx, cy, amp) in &bumps {
                let dx = c as f64 - cx;
                let dy = r as f64 - cy;
                extra += amp * (-(dx * dx + dy * dy) / 4.0).exp();
            }
            base[r * cols + c] += extra;
        }
    }
    base
}

fn main() {
    let rows = 59;
    let cols = 150;
    let base = textured_frame(rows, cols);
    let shifted = shift2d(&base, rows, cols, -1.4, 2.3);

    let n_frames = 30usize;

    // --- Phase 1: BaselineFrame::new (prefilter) cost in isolation ---
    let t0 = Instant::now();
    for _ in 0..n_frames {
        let _bf = BaselineFrame::new(std::hint::black_box(&base), rows, cols);
        std::hint::black_box(&_bf);
    }
    let prefilter_total = t0.elapsed();

    // --- Phase 2: full DE + Nelder-Mead search cost, with a call counter ---
    let call_count = Cell::new(0u64);
    let bf = BaselineFrame::new(&base, rows, cols);
    let objective = |v: &[f64]| {
        call_count.set(call_count.get() + 1);
        bf.shift_compare(v[0], v[1], &shifted)
    };
    let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
    let config = DeConfig::default();

    let t1 = Instant::now();
    let mut last = None;
    for seed in 0..n_frames as u64 {
        call_count.set(0);
        let result = differential_evolution(objective, &bounds, &config, seed);
        last = Some((result.x, call_count.get()));
    }
    let search_total = t1.elapsed();

    // --- Phase 3: raw shift_compare cost in isolation (no DE/NM overhead) ---
    let n_raw = 20_000u32;
    let t2 = Instant::now();
    let mut acc = 0.0;
    for i in 0..n_raw {
        let jitter = (i as f64) * 1e-6;
        acc += bf.shift_compare(0.3 + jitter, -0.2, &shifted);
    }
    std::hint::black_box(acc);
    let raw_total = t2.elapsed();
    println!(
        "raw shift_compare (interior candidate): total={:?} avg={:?} over {n_raw} calls",
        raw_total,
        raw_total / n_raw
    );

    println!("frame size: {rows}x{cols} ({} px)", rows * cols);
    println!("n_frames (repeats): {n_frames}");
    println!(
        "BaselineFrame::new: total={:?} avg={:?}",
        prefilter_total,
        prefilter_total / n_frames as u32
    );
    println!(
        "DE+polish search:   total={:?} avg={:?}",
        search_total,
        search_total / n_frames as u32
    );
    let last_calls = last.as_ref().map(|(_, c)| *c).unwrap_or(1);
    if let Some((x, calls)) = &last {
        println!("last result: x={x:?} objective_calls={calls}");
    }
    println!(
        "avg objective-call cost: {:?}",
        search_total / n_frames as u32 / last_calls as u32
    );
}
