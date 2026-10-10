# Inner conversion-copy dispatch

Run the standalone copy timing probe through the workspace project, with native
PfaPack/SFMT libraries already built. It uses existing production copy methods,
preallocated arrays, one warmup per method, and the median of five batches.
Each batch repeats the copy enough times to process at least ten million items
(at least twenty calls for the largest case). The CSV time is nanoseconds per
call; setup and compilation are excluded.

```sh
JULIA_NUM_THREADS=4,0 JULIA_NUM_GC_THREADS=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no \
julia +1.13.1 --project=. --startup-file=no benchmark/inner_copy_dispatch.jl
```

Use `2,0` for the two-worker run. The zero interactive pool keeps the driver
and the statically scheduled work in the default compute pool and keeps thread
IDs compatible with the reference's workspace arrays. UCX's signal setting is
needed at process startup with the benchmark machine's MPICH `ch4:ucx` build;
Julia uses SIGSEGV for threaded GC safepoints.

The results in `results/` include the unchanged `8d815db` baseline. The baseline
uses the generic 64-item numerical-kernel gate even for SIMD copies; the patch
uses a copy-specific 65,536-item gate. The generic kernel gate is unchanged.
The source/destination bounds, empty-copy handling, conversion values and
untouched destination tails remain the same. Neither copy consumes randomness.

The source workspace used for these measurements has an isolated, pinned Julia
1.13 lock and system-MPICH preferences supplied by the Rust comparison harness.
Use the same project and providers when reproducing the recorded data. See the
dated report for hardware, BLAS, MPI, full-workload timing scope and validation.

## Remaining runtime gap

`results/runtime_gap_20261010.md` profiles the remaining gap after the copy patch.
It records the exact executed source hashes, prepared inputs, lock, MPI/BLAS
providers, raw repetitions and correctness audit. The benchmark uses an isolated
Julia 1.13.1 project supplied by the Rust comparison harness, not a newly resolved
workspace environment. Keep that lock and system-MPI preferences to reproduce
the numbers. Its prepared namelist contains `NVMCSample=300` for the following
one-process diagnostic; `STEPS=300` controls the SR loop.

```sh
JULIA_NUM_THREADS=1,0 JULIA_NUM_GC_THREADS=1 \
JULIA_MVMC_MPI=1 JULIA_MVMC_INNER_THREADS=1 JULIA_MVMC_PFAPACK_THREADS=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE UCX_MEMTYPE_CACHE=no \
/opt/mpich/bin/mpiexec -n 1 julia +1.13.1 --project=/path/to/isolated-project \
  --startup-file=no benchmark/runtime_gap_profile.jl \
  /path/to/L32/namelist.def /path/to/profile-output 300
```

This performs a full warmup, a sampled CPU/allocation timing, then a separately
warmed CTimer run. It writes rank-local timing and profile files. Use
`runtime_gap_allocations.jl INPUT OUTPUT` with the same one-process environment
for a separate 20-step allocation-site probe. Neither instrumented probe supplies
the headline medians. `runtime_gap_child.jl WARNTYPE_OUTPUT` measures an isolated
fixed-input child kernel, with preallocated buffers, one warmup and three batches
of 10,000 calls; it does not measure the whole optimizer.

`runtime_gap_prefix.jl PREPARED_INPUT_ROOT OUTPUT_ROOT RANKS` compares 20-step
production outputs and non-consuming SFMT blocks. Its prepared input tree is
`L{16,24,32}-ranks{1,2,4}-threads{4,2,1}/inputs/namelist.def`. Run each matching
MPI/thread layout on the baseline and candidate. Then compare the two output
roots through `uv run --no-project python benchmark/compare_runtime_gap_prefixes.py
BEFORE AFTER SUMMARY_JSON`; the roots contain `ranksN-threadsM/` directories.
The comparison uses the existing abs=rel=1e-11 budget and exact integer RNG blocks.

`runtime_gap_rng_trace.jl` takes the same three arguments. It is a separate,
optional developer audit that forwards the original C SFMT primitives, hashes
their integer words and primitive/seed tags, and records draw counts per rank.
The real2 conversion is checked against C `to_real2` for the consumed word. Its
runtime method instrumentation is excluded from all performance measurements
and does not modify the SFMT library or source. Compare each baseline/candidate
rank's trace text exactly; the published parity JSON retains counts and hashes.
