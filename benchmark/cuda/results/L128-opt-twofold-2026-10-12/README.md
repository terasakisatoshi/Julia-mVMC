# L128 Opt: complete public runner, original versus optimized production

This report compares the original GPU source `358e59038a64f344ec15ea48557245e8ab32a9e8`
with production source `77eae13a801e53e4582d23c52a5d00da102ca696`.
Both execute the unchanged public `run_gpu_para_opt_from_namelist` entry point
in separate Julia processes, each loading its own checkout. No benchmark
method replacements, dispatch switches or experimental kernels are installed.

The fixed workload is L128, 320 walkers, 20 Opt steps, seed 79,
warmup/interval 128, FP64, Julia 1.13.1, one Julia thread and one BLAS thread,
on one RTX 3060 (device 1). All production GPU mode environment overrides
are unset. The checkpoint hash identifies the original input cohort; Opt
does not load that checkpoint. Its contents and the modified ModPara are
archived in the earlier prefetch report.

Each process gets one first-call priming run, excluded from warm statistics.
Five complete warm runs per checkout follow in AB/BA order. Only one process
executes at a time; the other waits for its next input line. Each clock includes
parsing, initialization, all sampling/observables/derivatives/SR steps and
returned results, with CUDA synchronization. Imports, driver GC, priming and
receipt serialization are outside warm clocks. Each process retains its own
warmed allocation pool. No competing local build or numerical job runs
during timing.

| Complete public Opt | Warm median seconds | Relative speed |
| --- | ---: | ---: |
| Original GPU implementation | 10.481069638 | 1.0000x |
| Optimized production | 5.131510695 | **2.0425x** |

Baseline range is 10.43438–10.50829 s; candidate range is 5.12596–5.14164 s.
All five candidate runs are below the frozen 2x target, 5.2311317605 s.
Against the original frozen median 10.462263521 s, the speedup is **2.0388x**.
First public calls (execution plus JIT, excluding imports) took 57.1222 s
and 54.3821 s respectively; they are not warm measurements.

All **70 complete-run checks** pass: configurations, occupancy, projection
counts, accepted counters and subsequent UInt32 RNG words agree exactly;
every energy and parameter agrees at the unchanged atol/rtol 1e-9.
Maximum absolute energy-history difference is **5.6742e-10**, and maximum
parameter difference is **4.8417e-11**. Each implementation repeats its own
numerical results exactly across all five runs. Final energy/site is
0.7795418080426397 versus 0.7795418080426915; final accepted moves are 14,742
for both. See [all raw trials](whole-opt-ab.log), [numerical validation](whole-opt-validation.log)
and [machine-readable timings](verified-timings.json).

## Implementation and numerical checks

The complete change includes the Slater-column prefetch from PR #85,
paired Pfaffian/inverse construction from PR #86, and the following work:

- Delay four accepted inverse updates in the owned Opt sampling cache;
  include all pending terms in proposals and flush before consumers.
- Compute integer projection deltas in parallel and reuse overlap reciprocals
  in orbital derivatives, retaining division for extreme reciprocal scales.
- Batch up to four paired elimination updates in shared scratch, with capacity
  determined by the device memory budget. Pivot permutations also permute
  pending vectors. Build the inverse by extending its leading block by two
  columns at a time, reusing the native LU buffer and removing three full
  auxiliary matrices and their pointer tables.
- Evaluate one-body/diagonal-density Hamiltonians with one warp per term and
  walker, reading contiguous packed inverse columns. Preserve ordered final
  Hamiltonian sums. Operator geometry and the Green-function request select
  this route; general two-pair operators keep the existing route.

All arithmetic remains FP64. FMA, blocked updates and fixed reduction trees
change rounding. Tests retain their existing numerical tolerances: independent
CPU Pfaffians, inverse backward residuals and conditioning-aware forward checks;
all factor panel capacities 0 through 4; singular sectors, insufficient scratch,
strong projection cancellation; and native/cached CPU observable comparisons.
The native boundary route remains available for insufficient scratch; singular
and strongly cancelling paired boundaries keep the existing rebuild fallback.

The complete production GPU suite passes all 66 reported testsets, containing
10,842 assertions, with exit status 0. Focused new production checks pass 786
assertions. The earlier delayed-update report records why an old incremental
cache is not an exact floating-point oracle, including its rejected per-entry
comparisons and independent CPU diagnostics. Those historical values are not
presented as measurements of this final source. The complete-run comparison
here tests final physical outputs directly against the original source.

PhysCal remains outside this speedup claim, following the requested Opt focus.

## Reproduction

Prepare independent checkouts at the two source commits above, instantiate
their pinned `test/gpu` environments and build the SFMT/PfaPack native libraries.
The root and GPU Julia manifests are unchanged. Use this report's scripts
from any checkout containing the report; they do not modify either source.

```sh
mvmc_report=benchmark/cuda/results/L128-opt-twofold-2026-10-12
mvmc_inputs=benchmark/cuda/results/L128-opt-prefetch-2026-10-12
mvmc_run="$(mktemp -d)"
cp -R benchmark/cuda/inputs/hubbard_chain_L128 "$mvmc_run/input"
cp "$mvmc_inputs/modpara.def" "$mvmc_run/input/modpara.def"
gzip -dc "$mvmc_inputs/fixed-checkpoint.dat.gz" > "$mvmc_run/fixed-checkpoint.dat"
uv run --no-project python "$mvmc_report/mvmc-twofold-ab.py" \
  /absolute/path/to/baseline-checkout /absolute/path/to/candidate-checkout \
  "$mvmc_run/input/namelist.def" "$mvmc_run/fixed-checkpoint.dat" "$mvmc_run/receipts"
julia +1.13.1 --startup-file=no "$mvmc_report/mvmc-twofold-validate.jl" "$mvmc_run/receipts"
JULIA_NUM_THREADS=1 JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --startup-file=no \
  --project=/absolute/path/to/candidate-checkout/test/gpu \
  /absolute/path/to/candidate-checkout/test/gpu/runtests.jl
```

The orchestration, worker and validation sources are byte-identical to those
executed. Raw logs retain first-call timings, all warm repetitions, numerical
differences and source/input identifiers. Binary receipts are not needed by
normal tests and are generated by the reproduction command.
