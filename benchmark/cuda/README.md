# CUDA benchmarks

Run `test/gpu/benchmark.jl` with its standalone GPU environment. See the
[GPU guide](../../docs/src/en/gpu.md) and the
[Linux RTX3060 report](results/linux-x86_64-rtx3060-2026-10-12.md).
GPU0 is the debug/parity device; GPU1 is the benchmark device. Results distinguish
resident kernel work, host transfers and single-thread CPU comparisons.
The [L128 follow-up report](results/linux-x86_64-rtx3060-L128-2026-10-12.md)
records the large-lattice optimizations, numerical cancellation regression and
two-device measurements against the20× target.

L128 inputs are checked in under `inputs/hubbard_chain_L128` with generator
provenance. For a synchronized single-device run:

```sh
JULIA_MVMC_GPU_DEVICE=1 JULIA_MVMC_GPU_WALKERS=256 \
JULIA_MVMC_GPU_BENCH_STEPS=1 JULIA_MVMC_GPU_PROFILE=1 \
julia +1.13.1 --project=test/gpu test/gpu/benchmark.jl \
  benchmark/cuda/inputs/hubbard_chain_L128/namelist.def
```

`JULIA_MVMC_GPU_BENCH_STEPS` controls attempted proposals per walker in each
sampling repetition (default20). Sampling and fixed-configuration measurement
are timed separately; the reported CPU/GPU ratio uses identical energy, Greens
and derivatives. Setup, compilation and SR updates are excluded from that ratio.

The two-device script splits the same fixed batch across devices0/1, uses
persistent Julia workers and includes worker dispatch/synchronization time:

```sh
JULIA_MVMC_GPU_WALKERS=256 julia +1.13.1 --threads=2 --project=test/gpu \
  test/gpu/benchmark_multi_gpu.jl
```

Large inverse matrices use resident cuBLAS batched LU/inversion by default.
`JULIA_MVMC_GPU_INVERSE=cooperative` selects the Gauss-Jordan debug comparison
path; its inverse scratch is allocated only when selected. Neither mode changes
CPU-only behavior or the shared-tape sampling contract.
