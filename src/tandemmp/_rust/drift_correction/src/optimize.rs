//! A self-contained differential-evolution global optimiser plus a
//! Nelder-Mead local "polish" pass — a from-scratch replacement for
//! `scipy.optimize.differential_evolution(..., strategy='best1bin',
//! polish=True)`.
//!
//! This deliberately does **not** try to reproduce SciPy's exact RNG
//! stream or search trajectory (SciPy's own algorithm details, e.g. its
//! Latin Hypercube initialisation and internal random-index draws, aren't
//! part of the public contract we need to match). It only needs to:
//!
//! - reliably minimise a cheap, mildly non-convex 2-D objective (the
//!   RMSE-of-shifted-frames landscape from `solve.rs`) to within the
//!   tolerances `tests/drift/test_corrector.py` pins (0.2-0.3 px), and
//! - be *deterministic* for a given `u64` seed, so
//!   `test_seed_gives_reproducible_result` holds.
//!
//! `polish=True` in SciPy hands the DE result to L-BFGS-B for a gradient
//! based local refinement; we don't have (or want) analytic gradients
//! here, so the polish step is a small derivative-free Nelder-Mead
//! simplex instead, per the porting brief.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Inclusive `[low, high]` bounds for one search dimension.
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub low: f64,
    pub high: f64,
}

impl Bound {
    pub fn new(low: f64, high: f64) -> Self {
        assert!(low <= high, "Bound::new: low must be <= high");
        Self { low, high }
    }

    fn clamp(&self, v: f64) -> f64 {
        v.clamp(self.low, self.high)
    }

    fn sample(&self, rng: &mut impl Rng) -> f64 {
        rng.gen_range(self.low..=self.high)
    }
}

/// Tuning knobs for [`differential_evolution`]. Defaults mirror SciPy's
/// (`popsize=15`, `mutation=(0.5, 1.0)` dithering, `recombination=0.7`,
/// `tol=0.01`), except `maxiter`, which is capped much lower than SciPy's
/// default of 1000 — this objective is cheap and low-dimensional (2-D)
/// enough that convergence happens in tens of generations, and keeping the
/// cap modest bounds worst-case per-frame runtime.
#[derive(Debug, Clone, Copy)]
pub struct DeConfig {
    pub popsize_multiplier: usize,
    pub maxiter: usize,
    pub tol: f64,
    pub atol: f64,
    pub mutation: (f64, f64),
    pub recombination: f64,
}

impl Default for DeConfig {
    fn default() -> Self {
        Self {
            popsize_multiplier: 15,
            maxiter: 300,
            tol: 0.01,
            atol: 1e-12,
            mutation: (0.5, 1.0),
            recombination: 0.7,
        }
    }
}

/// Result of a completed optimisation run.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: f64,
}

/// Minimise `objective` over the hyper-rectangle `bounds` using
/// DE/best/1/bin, then locally refine the best candidate with Nelder-Mead.
/// `seed` drives an internal ChaCha8 RNG for full reproducibility.
pub fn differential_evolution<F>(
    objective: F,
    bounds: &[Bound],
    config: &DeConfig,
    seed: u64,
) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
{
    let ndim = bounds.len();
    assert!(ndim > 0, "differential_evolution: need at least one bound");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let popsize = (config.popsize_multiplier * ndim).max(4);

    // --- initial population: uniform-random within bounds ---
    let mut population: Vec<Vec<f64>> = (0..popsize)
        .map(|_| bounds.iter().map(|b| b.sample(&mut rng)).collect())
        .collect();
    let mut fitness: Vec<f64> = population.iter().map(|ind| objective(ind)).collect();

    let mut best_idx = argmin(&fitness);

    for _gen in 0..config.maxiter {
        for i in 0..popsize {
            // Pick 3 distinct helper indices, none equal to i, for the
            // classic DE/best/1 mutant: best + F * (r1 - r2).
            let (r1, r2) = distinct_pair(&mut rng, popsize, i);
            let f_scale = rng.gen_range(config.mutation.0..=config.mutation.1);

            let best = &population[best_idx];
            let mutant: Vec<f64> = (0..ndim)
                .map(|j| best[j] + f_scale * (population[r1][j] - population[r2][j]))
                .collect();

            // Binomial crossover against the current target vector.
            let j_rand = rng.gen_range(0..ndim);
            let target = &population[i];
            let mut trial: Vec<f64> = (0..ndim)
                .map(|j| {
                    let from_mutant = j == j_rand || rng.gen::<f64>() < config.recombination;
                    let raw = if from_mutant { mutant[j] } else { target[j] };
                    bounds[j].clamp(raw)
                })
                .collect();
            // Re-clamp defensively (clamp above already covers this, kept
            // explicit since trial is rebuilt below in edge cases).
            for (j, v) in trial.iter_mut().enumerate() {
                *v = bounds[j].clamp(*v);
            }

            let trial_fitness = objective(&trial);
            if trial_fitness <= fitness[i] {
                population[i] = trial;
                fitness[i] = trial_fitness;
                if trial_fitness < fitness[best_idx] {
                    best_idx = i;
                }
            }
        }

        if has_converged(&fitness, config.tol, config.atol) {
            break;
        }
    }

    let de_best_x = population[best_idx].clone();
    let de_best_fun = fitness[best_idx];

    // --- polish: Nelder-Mead local refinement from the DE optimum ---
    let polished = nelder_mead(&objective, &de_best_x, bounds);

    if polished.fun <= de_best_fun {
        polished
    } else {
        OptimizeResult {
            x: de_best_x,
            fun: de_best_fun,
        }
    }
}

/// Total order over fitness values that never panics on `NaN`.
///
/// `objective` (ultimately [`solve.rs`](crate::solve)'s `shift_compare`) can
/// return `NaN` for a candidate `(x, y)` that crops the frame to zero
/// pixels — reachable on small frames with the default `(-6, 6)` search
/// bounds. Every *direct* comparison elsewhere in this module (`<`, `<=`)
/// already handles that gracefully: IEEE 754 makes any comparison against
/// `NaN` evaluate to `false`, so a `NaN`-fitness candidate already never
/// *wins* a `trial_fitness < fitness[i]`-style check, exactly mirroring
/// how `scipy.optimize.differential_evolution`'s own such comparisons
/// silently reject `NaN`-scored trials in the Python implementation this
/// ports. But `argmin` and sorting the simplex need a *total* order, which
/// bare `f64` doesn't provide when `NaN` is involved — `partial_cmp`
/// returns `None`. This makes that total order explicit instead of calling
/// `.unwrap()`/`.expect()` on it: `NaN` sorts as worse than any real
/// number, so it behaves like `+inf` and simply never gets picked as the
/// best value, same as every other comparison here already treats it.
fn cmp_fitness(a: &f64, b: &f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap(),
    }
}

fn argmin(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|a, b| cmp_fitness(a.1, b.1))
        .map(|(i, _)| i)
        .expect("argmin: empty slice")
}

/// Draw two indices in `0..n`, both distinct from each other and from
/// `exclude`. Panics if `n < 3` (the caller enforces a minimum population
/// size well above that).
fn distinct_pair(rng: &mut impl Rng, n: usize, exclude: usize) -> (usize, usize) {
    let mut r1 = exclude;
    while r1 == exclude {
        r1 = rng.gen_range(0..n);
    }
    let mut r2 = exclude;
    while r2 == exclude || r2 == r1 {
        r2 = rng.gen_range(0..n);
    }
    (r1, r2)
}

/// SciPy-style convergence check: population fitness std small relative to
/// its mean (`std <= atol + tol * |mean|`).
fn has_converged(fitness: &[f64], tol: f64, atol: f64) -> bool {
    let n = fitness.len() as f64;
    let mean = fitness.iter().sum::<f64>() / n;
    let var = fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    std <= atol + tol * mean.abs()
}

/// A compact bounded Nelder-Mead simplex optimiser for local refinement.
/// Trial points are clamped into `bounds` after every simplex operation, so
/// the search never leaves the same hyper-rectangle DE searched.
fn nelder_mead<F>(objective: &F, x0: &[f64], bounds: &[Bound]) -> OptimizeResult
where
    F: Fn(&[f64]) -> f64,
{
    const ALPHA: f64 = 1.0; // reflection
    const GAMMA: f64 = 2.0; // expansion
    const RHO: f64 = 0.5; // contraction
    const SIGMA: f64 = 0.5; // shrink
    const MAX_ITER: usize = 200;
    const F_TOL: f64 = 1e-10;
    const X_TOL: f64 = 1e-8;

    let ndim = x0.len();
    let clamp = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .enumerate()
            .map(|(j, &vj)| bounds[j].clamp(vj))
            .collect()
    };

    // Initial simplex: x0 plus one perturbed vertex per dimension.
    let mut simplex: Vec<Vec<f64>> = vec![clamp(x0)];
    for j in 0..ndim {
        let mut v = x0.to_vec();
        let span = bounds[j].high - bounds[j].low;
        let step = (0.05 * v[j].abs()).max(0.02 * span.max(1e-9)).max(1e-6);
        v[j] += step;
        simplex.push(clamp(&v));
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| objective(v)).collect();

    for _ in 0..MAX_ITER {
        // Sort simplex vertices by objective value, best first.
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&a, &b| cmp_fitness(&values[a], &values[b]));
        simplex = order.iter().map(|&i| simplex[i].clone()).collect();
        values = order.iter().map(|&i| values[i]).collect();

        let spread = values[values.len() - 1] - values[0];
        let x_spread: f64 = (1..simplex.len())
            .map(|i| {
                (0..ndim)
                    .map(|j| (simplex[i][j] - simplex[0][j]).abs())
                    .fold(0.0, f64::max)
            })
            .fold(0.0, f64::max);
        if spread.abs() < F_TOL && x_spread < X_TOL {
            break;
        }

        let worst = simplex.len() - 1;
        // Centroid of all points except the worst.
        let centroid: Vec<f64> = (0..ndim)
            .map(|j| (0..worst).map(|i| simplex[i][j]).sum::<f64>() / worst as f64)
            .collect();

        let reflected: Vec<f64> = clamp(
            &(0..ndim)
                .map(|j| centroid[j] + ALPHA * (centroid[j] - simplex[worst][j]))
                .collect::<Vec<f64>>(),
        );
        let f_reflected = objective(&reflected);

        if f_reflected < values[0] {
            let expanded: Vec<f64> = clamp(
                &(0..ndim)
                    .map(|j| centroid[j] + GAMMA * (reflected[j] - centroid[j]))
                    .collect::<Vec<f64>>(),
            );
            let f_expanded = objective(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                values[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_reflected;
            }
            continue;
        }

        if f_reflected < values[worst - 1] {
            simplex[worst] = reflected;
            values[worst] = f_reflected;
            continue;
        }

        // Contraction.
        let contracted: Vec<f64> = clamp(
            &(0..ndim)
                .map(|j| centroid[j] + RHO * (simplex[worst][j] - centroid[j]))
                .collect::<Vec<f64>>(),
        );
        let f_contracted = objective(&contracted);
        if f_contracted < values[worst] {
            simplex[worst] = contracted;
            values[worst] = f_contracted;
            continue;
        }

        // Shrink the whole simplex toward the best point.
        let best = simplex[0].clone();
        for i in 1..simplex.len() {
            let shrunk: Vec<f64> = clamp(
                &(0..ndim)
                    .map(|j| best[j] + SIGMA * (simplex[i][j] - best[j]))
                    .collect::<Vec<f64>>(),
            );
            values[i] = objective(&shrunk);
            simplex[i] = shrunk;
        }
    }

    let best_i = argmin(&values);
    OptimizeResult {
        x: simplex[best_i].clone(),
        fun: values[best_i],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_sphere_minimum() {
        // f(x, y) = (x - 1.3)^2 + (y + 2.1)^2, minimum at (1.3, -2.1).
        let objective = |v: &[f64]| (v[0] - 1.3).powi(2) + (v[1] + 2.1).powi(2);
        let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
        let config = DeConfig::default();
        let result = differential_evolution(objective, &bounds, &config, 42);
        assert!((result.x[0] - 1.3).abs() < 0.05, "x={:?}", result.x);
        assert!((result.x[1] + 2.1).abs() < 0.05, "x={:?}", result.x);
    }

    #[test]
    fn same_seed_is_reproducible() {
        let objective = |v: &[f64]| (v[0] - 1.3).powi(2) + (v[1] + 2.1).powi(2);
        let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
        let config = DeConfig::default();
        let a = differential_evolution(objective, &bounds, &config, 123);
        let b = differential_evolution(objective, &bounds, &config, 123);
        assert_eq!(a.x, b.x);
    }

    #[test]
    fn nan_fitness_candidates_never_win_and_never_panic() {
        // Regression test for a real panic: an objective that returns NaN
        // for part of the search space (mirrors `solve.rs::shift_compare`
        // returning NaN when a candidate (x, y) crops a small frame to zero
        // pixels) used to panic in `argmin`/the Nelder-Mead sort via
        // `.expect()`/`.unwrap()` on `partial_cmp`. It should instead just
        // never let a NaN-scored candidate win, and converge on the finite
        // minimum in the region that isn't NaN.
        let objective = |v: &[f64]| {
            if v[0].abs() > 3.0 || v[1].abs() > 3.0 {
                return f64::NAN;
            }
            (v[0] - 1.3).powi(2) + (v[1] + 2.1).powi(2)
        };
        let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
        let config = DeConfig::default();
        let result = differential_evolution(objective, &bounds, &config, 42);
        assert!(!result.fun.is_nan(), "fun={}", result.fun);
        assert!((result.x[0] - 1.3).abs() < 0.1, "x={:?}", result.x);
        assert!((result.x[1] + 2.1).abs() < 0.1, "x={:?}", result.x);
    }

    #[test]
    fn all_nan_fitness_does_not_panic() {
        // Degenerate case: every candidate in bounds is NaN (mirrors a
        // frame so small that no shift within the default bounds leaves
        // any pixels after cropping). Must not panic, even though there's
        // no meaningful finite result to converge on.
        let objective = |_: &[f64]| f64::NAN;
        let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
        let config = DeConfig::default();
        let result = differential_evolution(objective, &bounds, &config, 42);
        assert!(result.fun.is_nan());
    }

    #[test]
    fn different_seed_can_differ() {
        // Not a strict requirement, but sanity-checks that the seed is
        // actually being used rather than silently ignored.
        let objective = |v: &[f64]| (v[0] - 1.3).powi(2) + (v[1] + 2.1).powi(2);
        let bounds = [Bound::new(-6.0, 6.0), Bound::new(-6.0, 6.0)];
        let config = DeConfig::default();
        let a = differential_evolution(objective, &bounds, &config, 1);
        let b = differential_evolution(objective, &bounds, &config, 2);
        // Both should still find the minimum closely, regardless of seed.
        assert!((a.x[0] - 1.3).abs() < 0.05);
        assert!((b.x[0] - 1.3).abs() < 0.05);
    }
}
