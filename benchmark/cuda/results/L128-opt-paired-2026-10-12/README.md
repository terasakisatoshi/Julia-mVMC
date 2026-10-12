# L128 Opt: reuse paired skew factors

Large Opt sampling boundaries compute the Pfaffian and inverse from one
paired skew elimination, replacing separate normal Pfaffian and LU inverse
factorizations. The decomposition is P*A*P' = U*D*U', where U is unit upper
triangular and D has independent 2-by-2 skew blocks. The inverse is formed
from U^-1 and D^-1 and then unpermuted. Buffers belong to the Opt workspace;
there is no process-wide workspace registry. Startup reuses its existing
normal Pfaffians and native LU inverse. PhysCal keeps its existing route.

The elimination retains ordered FP64 arithmetic. The inverse product uses
FP64 FMA. A zero/nonfinite sector Pfaffian, any nonfinite inverse entry, or
projected cancellation above 1024 triggers the independent normal Pfaffian
and cooperative inverse rebuild for that walker. The cancellation threshold
is a conservative fallback policy, not a universal forward-error bound.

## Complete public Opt comparison

Julia 1.13.1, one RTX 3060 (device 1), one Julia and BLAS thread, FP64,
seed 79, 320 walkers, 20 SR steps, warmup/interval 128 proposals. Each clock
wraps the complete public `run_gpu_para_opt_from_namelist`, including parsing,
initialization, sampling, observables, derivatives, SR and returned results.
Imports, compilation priming, benchmark setup and driver GC are outside the
warm clocks. No competing local builds or numerical jobs ran during timing.

One unprimed call per variant precedes five paired warm repetitions,
alternating AB/BA. The benchmark freezes both runner bodies and changes only
which boundary workspace the runner constructs. The baseline is upstream
`cb12901` (including PR #85); the candidate source is `cb66b07`. A subsequent
commit removes only an extra blank line at the factor file's end.

| Complete Opt | Warm median seconds | Relative speed |
| --- | ---: | ---: |
| Native boundary | 10.453751579 | 1.0000x |
| Paired boundary | 8.363484979 | 1.2499x |

The median drops by 2.0903 s (20.00%). Every paired comparison improves.
Baseline range: 10.41868–10.50380 s; candidate: 8.32192–8.38845 s.
The original frozen 10.462263521 s target still requires at most
5.2311317605 s for 2x; this patch does **not** meet that overall target.
PhysCal speed is not claimed by this experiment.

All 63 complete-run comparisons passed: exact configurations, occupancy,
projection counts, accepted counters and subsequent UInt32 RNG words;
per-element energies and parameters at atol/rtol 1e-9. Energy/site is
0.7795418080426397 baseline versus 0.7795418080431082 candidate, a difference
of about 4.7e-13. Each final interval has 14,742 accepted moves.

The complete production GPU suite passed **8,628 assertions in 61 testsets**.
This includes 302 new checks against independent CPU inverses and Pfaffians,
closed-form Pfaffians, controlled condition numbers up to 1e6, dimensions
2/6/32/64/128/130, singular QPs, poisoned reusable buffers, strong projected
cancellation and startup reuse. Residual bounds scale with n*eps; forward
bounds additionally include the independently computed matrix condition.
The existing suite also checks shared sampling controls, independent CPU
observables and repeated complete 20-step Opt runs. No existing tolerance was
changed and no test was excluded. The new test initially had an invalid
length-one range in its data generator; the fixed generator produced the
archived passing focused and full-suite runs.

## Reproduction

Use the pinned `test/gpu` environment and built native reference libraries.
From the repository root:

```sh
mvmc_paired_report=benchmark/cuda/results/L128-opt-paired-2026-10-12
mvmc_paired_inputs=benchmark/cuda/results/L128-opt-prefetch-2026-10-12
mvmc_paired_run="$(mktemp -d)"
cp -R benchmark/cuda/inputs/hubbard_chain_L128 "$mvmc_paired_run/input"
cp "$mvmc_paired_inputs/modpara.def" "$mvmc_paired_run/input/modpara.def"
gzip -dc "$mvmc_paired_inputs/fixed-checkpoint.dat.gz" > "$mvmc_paired_run/fixed-checkpoint.dat"
JULIA_NUM_THREADS=1 julia +1.13.1 --startup-file=no --project=test/gpu \
  "$mvmc_paired_report/whole-opt-ab-driver.jl" \
  "$mvmc_paired_run/input/namelist.def" "$mvmc_paired_run/fixed-checkpoint.dat"
JULIA_NUM_THREADS=1 JULIA_MVMC_GPU_DEVICE=0 \
  julia +1.13.1 --startup-file=no --project=test/gpu test/gpu/runtests.jl
```

The checkpoint is hash metadata linking this run to the frozen end-to-end
cohort; Opt does not load it. The committed driver only changes its temporary
runner-source path to a sibling relative path. Raw source/input hashes and
all repetitions are retained in [the A/B log](whole-opt-ab.log), with
[extracted timings](verified-timings.json), [full GPU output](full-gpu-suite.log)
and [focused checks](focused-tests.log). [PR #85's validation](../L128-opt-prefetch-2026-10-12/README.md)
is archived separately and is not part of this patch's speedup claim.
