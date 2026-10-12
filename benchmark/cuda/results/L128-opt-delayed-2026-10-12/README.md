# L128 Opt: delay inverse updates and share projection work

Large Opt sampling caches keep four accepted rank-two updates before writing
the full inverse. The pending representation is
`B += u*v' - v*u' + e_a*v' - v*e_a'`, with the original inverse-column product
`u`, scaled inverse row `v`, and a separate moved-electron index `a`.
Keeping the unit correction separate avoids losing it when forming `u`.
Proposal ratios and inverse-vector products include every pending term.
A full rebuild discards the queue, and every sampling return flushes it;
observables and derivatives always consume a materialized skew inverse.
The Opt cache owns all buffers, with no process-wide registry.

The common cached sampler also computes integer projection-count deltas in
parallel and uses a fixed reduction for their parameter dot product. Sparse
orbital derivatives reuse the overlap reciprocal within each output warp;
extreme reciprocals retain division. All arithmetic remains FP64. FMA and
fixed reduction trees reassociate floating-point operations; RNG generation,
proposal scheduling and integer counts are unchanged. Small Opt states and
states exceeding the device shared-memory budget retain immediate updates.
PhysCal retains immediate updates; this report makes no PhysCal speed claim.

## Complete public Opt comparison

One RTX 3060 (device 1), Julia 1.13.1, one Julia and BLAS thread, FP64,
seed 79, 320 walkers, 20 SR steps, warmup/interval 128 proposals. Each clock
wraps the entire public `run_gpu_para_opt_from_namelist`, including parsing,
initialization, sampling, observables, derivatives, SR and returned results.
Imports, compilation priming, driver GC and comparison receipts are outside
the warm clocks. No competing local builds or numerical jobs ran during timing.

The baseline is PR #86 head `1f7946a`, whose tree equals upstream merge
`933100a`. Frozen baseline sampling/control and contraction bodies run in the
same process as the candidate. Only the baseline disables the Opt delay-buffer
allocation. One priming run per variant precedes five paired warm repetitions,
alternating AB/BA. The candidate source is `7f9d2fd`. It is the production implementation, without the
experimental dictionary-based buffer registry used in earlier prototypes.

| Complete Opt | Warm median seconds | Relative speed |
| --- | ---: | ---: |
| PR #86 | 8.367510787 | 1.0000x |
| Delayed updates and shared projection work | 6.488112183 | 1.2897x |

All five pairs improve. Baseline range: 8.34519–8.38583 s; candidate:
6.47161–6.50903 s. Against the original frozen 10.462263521 s, the cumulative
speedup is **1.6125x**. The 2x objective still requires at most 5.2311317605 s;
this patch does not meet it.

All **63 complete-run checks** pass: exact final configurations, occupancy,
projection counts, accepted counters and subsequent UInt32 RNG words;
per-element energies and parameters at atol/rtol 1e-9. Max absolute total
energy-history difference is 4.36e-10; parameter difference is 3.66e-11.
Final energy/site is 0.7795418080431082 versus 0.7795418080427028, with 14,742
accepted moves in the final interval for both implementations.

## Numerical validation

Focused L128 cancellation/control tests pass 490 assertions. Extended cache
and derivative tests pass 3,345 assertions, including long packets,
130-electron tail tiles, exchanges, rejection, threshold rebuilds, zero
sectors, queue flushing and extreme overlap reciprocals. Independent host
matrix multiplication checks the inverse using the existing absolute 1e-9
residual threshold. Existing numerical tolerances are unchanged.

A separate 320-walker/128-proposal diagnostic exposes differences that a
normwise cache comparison misses. Four sector Pfaffians differ from the old
incremental cache by more than per-entry atol/rtol 1e-9; each of those four
candidate entries is closer to independently recomputed CPU PfaPack values.
Both caches contain accumulated sector roundoff (maximum CPU-relative error
1.71e-8 baseline, 1.63e-8 candidate), so the old incremental cache is not an
exact oracle. Its initial failed comparison is retained in
[the diagnostic record](old-cache-oracle-rejected.log), rather than silently
reported as a passing per-entry sector gate.

Independent CPU factors for every final configuration give maximum relative
projected-overlap errors of **3.08e-10 baseline and 2.98e-10 candidate**. Both
pass the unchanged per-element atol/rtol 1e-9 gate. Exact discrete controls,
projection counts, and empty update queues also pass: nine assertions in
[the independent diagnostic](inverse-diagnostic.log). The maximum normalized
inverse residual `||A*X-I||∞/(||A||∞*||X||∞)` improves from 3.97e-11 to 1.55e-11.
The direct old-inverse per-entry gate also fails (maximum difference 1.36e-6).
These measured residuals and physical results do not establish a universal
forward-error bound for arbitrary inputs.

The full production GPU suite passes **10,391 assertions**, including the
existing independent CPU observables, Pfaffians, inverses and repeated
20-step Opt checks. All existing testsets ran with unchanged tolerances.
See [full production GPU output](full-gpu-suite.log).

## Reproduction

Use the pinned `test/gpu` environment and built native reference libraries:

```sh
mvmc_delayed_report=benchmark/cuda/results/L128-opt-delayed-2026-10-12
mvmc_delayed_inputs=benchmark/cuda/results/L128-opt-prefetch-2026-10-12
mvmc_delayed_run="$(mktemp -d)"
cp -R benchmark/cuda/inputs/hubbard_chain_L128 "$mvmc_delayed_run/input"
cp "$mvmc_delayed_inputs/modpara.def" "$mvmc_delayed_run/input/modpara.def"
gzip -dc "$mvmc_delayed_inputs/fixed-checkpoint.dat.gz" > "$mvmc_delayed_run/fixed-checkpoint.dat"
JULIA_NUM_THREADS=1 julia +1.13.1 --startup-file=no --project=test/gpu \
  "$mvmc_delayed_report/whole-opt-ab-driver.jl" \
  "$mvmc_delayed_run/input/namelist.def" "$mvmc_delayed_run/fixed-checkpoint.dat"
JULIA_NUM_THREADS=1 JULIA_MVMC_GPU_DEVICE=0 \
  julia +1.13.1 --startup-file=no --project=test/gpu test/gpu/runtests.jl
JULIA_NUM_THREADS=1 julia +1.13.1 --startup-file=no --project=test/gpu \
  "$mvmc_delayed_report/inverse-diagnostic.jl" "$mvmc_delayed_run/input/namelist.def"
```

The checkpoint is hash metadata linking the fixed cohort; Opt does not load it.
The committed driver replaces only the executed temporary source path with a
sibling relative path. See [raw A/B output](whole-opt-ab.log),
[extracted timings](verified-timings.json), [L128 checks](focused-l128.log), and
[cache/derivative checks](focused-cache-derivatives.log).
