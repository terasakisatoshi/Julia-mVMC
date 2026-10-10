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
