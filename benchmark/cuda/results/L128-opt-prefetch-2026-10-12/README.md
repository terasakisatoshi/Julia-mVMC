# L128 Opt: common Slater-column staging

The resident inverse update stages its common Slater column in the existing
second shared vector for matrices of dimension 128 and larger. Ordered FP64
multiply/add operations, RNG consumption and update scheduling are preserved.
The first barrier publishes the staged column; the existing post-product
barrier prevents its overwrite until all readers finish. Smaller matrices
retain the original path.

## Complete public Opt comparison

Linux host, Julia 1.13.1, one RTX 3060 (device 1), one Julia and BLAS thread,
FP64, seed 79, 320 walkers, 20 SR steps, warmup/interval 128 proposals.
Each clock wraps `run_gpu_para_opt_from_namelist`, including parsing, setup,
sampling, observables, derivatives, SR and final returned results. Imports,
first-use compilation, benchmark preparation and the driver's GC are outside
warmed clocks. There were no competing local builds or numerical jobs during
these measurements. This host cohort is separate from historical container
measurements and the paired-factor experiments.

The production source is commit `20dd152`. The frozen comparison kernel is
from `358e590` (also contained in upstream main `1642d30`). The comparator is
injected only by the benchmark driver; ordinary public calls use the new
production kernel. Both variants execute in the same process. One unprimed
call per variant precedes five paired warm repetitions, alternating AB/BA
order. All repetitions are retained.

| Complete Opt | Warm median seconds | Relative elapsed speed |
| --- | ---: | ---: |
| Frozen sampler | 10.511960454 | 1.0000× |
| Staged column | 10.450711013 | 1.0059× |

The median is 61.25 ms (0.583%) lower. All five paired comparisons favor the
staged version, by 41.79, 111.59, 27.62, 30.20 and 92.70 ms. Timing ranges are
10.48333–10.54341 s baseline and 10.40038–10.48237 s candidate. This is a small
incremental improvement; it does **not** meet the requested 2× Opt target.
The earlier isolated sampler probe was about 4.2% faster, but that percentage
must not be presented as a complete-Opt speedup.

The A/B run passed 63 repeated-control checks: exact configurations, occupancy,
projection counts, accepted counters and subsequent UInt32 RNG words, plus
per-element energy-history and parameter comparisons at atol/rtol 1e-9.
Each call reported final energy/site 0.7795418080426397 and 14,742 accepted
moves in the last interval. No floating-point equality is asserted bitwise.

The complete production GPU suite passed **8,326 assertions in 59 testsets**
on device 0. It covers independently computed CPU observables, strongly
cancelling configurations, singular-QP/cofactor fallbacks, multiple update
routes and public 20-step repeatability. Native PfaPack and SFMT were rebuilt
in the isolated worktree using unchanged canonical build commands. The first
attempt lacked SFMT's native library; after building it, the complete suite
passed without test exclusions or tolerance changes.

## Reproduction

Run this historical comparison at the PR #85 merge `cb12901`, which includes
the measured production source and this report. Its frozen kernel still uses
the surrounding production runner; later changes alter the comparison even
when the driver is unchanged.

From the repository root, with the pinned `test/gpu` environment and native
reference libraries built:

```sh
mvmc_prefetch_report=benchmark/cuda/results/L128-opt-prefetch-2026-10-12
mvmc_prefetch_run="$(mktemp -d)"
cp -R benchmark/cuda/inputs/hubbard_chain_L128 "$mvmc_prefetch_run/input"
cp "$mvmc_prefetch_report/modpara.def" "$mvmc_prefetch_run/input/modpara.def"
gzip -dc "$mvmc_prefetch_report/fixed-checkpoint.dat.gz" > "$mvmc_prefetch_run/fixed-checkpoint.dat"
JULIA_NUM_THREADS=1 julia +1.13.1 --project=test/gpu \
  "$mvmc_prefetch_report/whole-opt-ab-driver.jl" \
  "$mvmc_prefetch_run/input/namelist.def" \
  "$mvmc_prefetch_run/fixed-checkpoint.dat"
```

The driver uses device 1 and compares complete Opt only. The checkpoint hash
is preserved for correspondence with the frozen end-to-end cohort; Opt does
not load this fixed PhysCal checkpoint. `modpara.def` and all remaining input
files reproduce the frozen input settings. Kernel source hashes and all timing
repetitions are emitted in the raw log. The committed driver changes only its
comparator path from the measured `/tmp` path to a sibling relative path.

Evidence: [public A/B log](whole-opt-ab.log), [verified timing extraction](verified-timings.json),
[complete GPU suite](full-gpu-suite.log), [frozen comparison kernels](baseline-ab-kernels.jl),
[driver](whole-opt-ab-driver.jl) and [source/input hashes](sha256.txt).
