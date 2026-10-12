# CUDA benchmarks

Run `test/gpu/benchmark.jl` with its standalone GPU environment. See the
[GPU guide](../../docs/src/en/gpu.md) and the
[Linux RTX3060 report](results/linux-x86_64-rtx3060-2026-10-12.md).
GPU0 is the debug/parity device; GPU1 is the benchmark device. Results distinguish
resident kernel work, host transfers and single-thread CPU comparisons.
