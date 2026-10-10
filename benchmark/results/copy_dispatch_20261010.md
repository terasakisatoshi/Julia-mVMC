# Conversion-copy dispatch measurements (2026-10-10)

Related to https://github.com/tmisawa/Julia-mVMC/issues/71. Baseline: 8d815db0eec0ba12dea88aa9edb3d5f74d7b5ccd. Candidate: the copy-specific 65,536-item gate in this branch; generic numerical-kernel gate stays at 64. Conversion operations, bounds and RNG are unchanged.

Short static thread dispatch dominates SIMD copies: at 128 items and 4 workers, real-to-complex takes 61 ns serial versus 3041 ns with the baseline threaded flag. At 2048 items it is 365 ns versus 2912 ns. The two-worker 32768-item case does not improve (5046 ns serial versus 5229 ns parallel); 65536 items improves (11640 ns versus 8078 ns). We use a conservative common threshold, not a universal crossover claim. Small copies now follow the serial SIMD path; large copies retain threading. Raw microbenchmark CSVs and the probe are included.

## Production workload

L16/L24/L32 Hubbard chains, 300 SR steps,total 300 samples/step, 1 warmup, 3 measured repetitions per cell. Times are slowest-rank production API times, including parse/initialization/optimization/output, excluding process startup/JIT/warmup and separate 20-step profiling. Sequential runs with no concurrent heavy builds/tests. Rank-local samples are 300/150/75. All worker settings verified: MVMC frames observed on 4/2/1 threads per rank. Native PfaPack threading stays off.

| Model | Layout | Before min / median / max (s) | After min / median / max (s) | Median reduction |
|---|---|---:|---:|---:|
| L16 | 1×4 | 5.150224 / 5.165813 / 5.193660 | 4.873635 / 4.874976 / 4.876528 | 5.63% |
| L16 | 2×2 | 3.079024 / 3.085645 / 3.092858 | 2.865805 / 2.871831 / 2.877717 | 6.93% |
| L16 | 4×1 | 1.732902 / 1.741198 / 1.759906 | 1.715236 / 1.722909 / 1.738506 | 1.05% |
| L24 | 1×4 | 9.695936 / 9.774402 / 9.882514 | 9.368180 / 9.418530 / 9.477980 | 3.64% |
| L24 | 2×2 | 5.748818 / 5.761951 / 5.872163 | 5.464219 / 5.551135 / 5.589936 | 3.66% |
| L24 | 4×1 | 3.455773 / 3.471335 / 3.494779 | 3.460235 / 3.462333 / 3.463843 | 0.26% |
| L32 | 1×4 | 16.818156 / 16.879773 / 16.888224 | 16.615545 / 16.772030 / 16.807614 | 0.64% |
| L32 | 2×2 | 9.862322 / 9.915609 / 9.925222 | 9.497598 / 9.616061 / 9.806468 | 3.02% |
| L32 | 4×1 | 6.065782 / 6.075963 / 6.121723 | 5.934062 / 6.035221 / 6.131077 | 0.67% |

L16 hybrid median reductions are 5.6% (1×4) and 6.9% (2×2). L24 is about 3.6%; L32 1×4 changes only 0.6%. Three repetitions and near-unchanged single-thread controls do not establish small differences as significant. All models are fastest at 4×1; these timings do not prove convergence or effective-sample gains. Changing rank counts changes seeds/chains/trajectories.

## Environment and validation

Linux x86_64 Dev Container; AMD Ryzen 9 PRO 8945HS (8 cores/16 logical); Julia 1.13.1; MPICH 4.2.0 ch4:ucx/Hydra/PMI1 at `/opt/mpich`. LinearAlgebra OpenBLAS 0.3.30 ILP64 and native helper OpenBLAS 0.3.26 LP64, both Cooperlake kernels and 1 thread per process. JULIA_NUM_THREADS=N,0; JULIA_NUM_GC_THREADS=1; JULIA_MVMC_INNER_THREADS=1; JULIA_MVMC_PFAPACK_THREADS=0. Startup UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE and UCX_MEMTYPE_CACHE=no avoid GC-safepoint signal conflicts. Original source/submodules unchanged; existing Julia 1.13 lock copied into isolated source projects with system-MPI preferences.

Full optimizer tests: 22,008 passed at 1 thread; 22,009 at 4 threads. Added tests cover gate boundary, environment/explicit disable, short/long/empty copies, return identity and untouched tails. Separate before/after 20-step checks cover all 9 model/layout cases: 42 initial/final SFMT 624-word blocks match exactly; 88794 finite computed fields pass abs+rel 1e-11,maximum absolute difference 0.0. Computed floats are not compared bitwise. No tolerance widening or RNG reseeding to mask differences.

The microbenchmark command is documented in ../README.md. Workload commands/data are preserved by the Rust harness at https://github.com/AtelierArith/mvmc-rs (scripts/bench_mpi.py, --layout 1x4 2x2 4x1 --steps 300 --samples 300 --sample-policy total --warmups 1 --reps 3, --julia-source for candidate). The associated dated report records full source/binary/library hashes. This change was measured on Linux only; larger models and other machines require separate measurements.

Full environment, executed-source and linked-library hashes: [JSON](copy_dispatch_20261010_environment.json). The candidate was measured before committing; its source diff and SHA-256 hashes are recorded there.
