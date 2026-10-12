# Linux Dev Container: L128 whole-run CPU, MPI and single-GPU comparison

Measured on 2026-10-12 using the frozen numerical source from commit
`dcb36b570389a2426369361b41f27b3d8e0e6528`. The primary cohort retains **320
samples globally** at every measurement step, runs **20 optimization steps**,
and measures one fixed-parameter PhysCal run. Warmup and interval are each one
sweep, seed is 79, and each cell has one cold execution plus three warmed
executions. All calls returned success and finite summaries. Measurements ran
sequentially without competing builds or other benchmark processes.

The single GPU beats MPI with four or eight ranks here. **MPI with sixteen
physical cores is fastest:** its warmed median is 11.1744 s for optimization
and 0.762520 s for PhysCal, versus the GPU's 12.5482 s and 0.933714 s. The GPU
is 6.15× / 4.78× faster than the one-rank CPU baseline; the requested 20×
whole-run target was not reached. These are observed lifecycle timings with
the sampling and output differences below, not effective-sample speedups.

A subsequent [phase profile](linux-x86_64-devcontainer-L128-whole-2026-10-12/phase-profile.md)
of the same source attributes 36.7% of optimization time to sampling, 35.3%
to post-SR inverse refresh, and 16.3% to observables. Its synchronized diagnostic
timings supplement this table; they do not replace the public-runner medians.

## Primary cohort: 320 samples globally

CPU cells use genuine MPI worlds, one Julia worker thread and one BLAS thread
per rank, `NSplitSize=1`, and independent sampling groups. The baseline is a
real one-rank MPI launch. Every rank is bound to its own physical core.

| Backend | Global samples | Samples/group or GPU walkers | Opt median (s) | PhysCal median (s) | CPU1/Opt | CPU1/PhysCal |
|---|---:|---:|---:|---:|---:|---:|
| CPU, 1 MPI rank | 320 | 320 | 77.137908551 | 4.465393414 | 1.00× | 1.00× |
| CPU, 4 MPI ranks | 320 | 80 | 22.501600746 | 1.382951334 | 3.43× | 3.23× |
| CPU, 8 MPI ranks | 320 | 40 | 14.729498005 | 0.984994268 | 5.24× | 4.53× |
| CPU, 16 MPI ranks | 320 | 20 | 11.174382908 | 0.762519904 | 6.90× | 5.86× |
| GPU, one RTX 3060 | 320 | 320 independent walkers | 12.548162391 | 0.933714220 | 6.15× | 4.78× |

MPI16 is 1.123× faster than GPU1 for Opt and 1.225× for PhysCal. MPI8/GPU1
elapsed ratios are 1.174× and 1.055×; the PhysCal margin is small on this
machine and workload.

| Backend | Opt warmed min–max (s) | PhysCal warmed min–max (s) | Cold Opt (s) | Cold PhysCal (s) |
|---|---:|---:|---:|---:|
| CPU1 | 76.665950383–77.179709223 | 4.408157762–4.466237027 | 102.511210009 | 6.330361634 |
| MPI4 | 22.479790613–22.675199254 | 1.376226567–1.385175600 | 50.750598263 | 3.462813889 |
| MPI8 | 14.723369822–14.773077905 | 0.972728802–1.015504389 | 46.582524145 | 3.261234174 |
| MPI16 | 11.171947536–11.490397516 | 0.758886651–0.765171021 | 46.237714130 | 3.200527157 |
| GPU1 | 12.545820295–12.578782317 | 0.931140795–0.935244332 | 58.109173250 | 1.387615357 |

Cold execution includes first-call compilation and real work. Process launch,
module imports and dependency/bootstrap builds are outside these clocks.
Cold values are not included in the warmed medians.

## Timing and Monte Carlo contracts

Each elapsed measurement wraps a complete public optimization or PhysCal
runner: input parsing, parameter/state setup, initialization, Monte Carlo,
observables, SR in Opt, ordinary synchronization and final results/output.
CPU output files are included; the GPU returns numerical arrays and final
summaries instead of those files. Fixed PhysCal checkpoint preparation is
outside timing. GPU timing synchronizes before and after the public call.
MPI timing reports the maximum rank wall time; the preceding barrier,
subsequent timing/status allreduces and the driver's extra PhysCal energy-file
readback are outside the clock. MPI collectives and the replicated direct SR
solve performed by the public runner remain inside timing.

CPU uses one sequential SFMT chain per group with the existing base+group seed
convention. GPU uses independent XORWOW walkers, legal empty-destination
proposals and resident rank updates. Their trajectories, burn-state reuse,
accepted moves and initial warmup work differ. Equal retained samples and SR
steps do not establish equal effective sample sizes. No cross-backend
trajectory or bitwise floating-point equality is claimed.

The following are configured proposal-attempt budgets derived from the
sampling loops, not measured acceptance counts or effective samples:

| Backend | Opt attempts, all 20 steps | PhysCal attempts, one run |
|---|---:|---:|
| CPU1 | 821,760 | 41,088 |
| MPI4 groups | 829,440 | 41,472 |
| MPI8 groups | 839,680 | 41,984 |
| MPI16 groups | 860,160 | 43,008 |
| GPU320 walkers | 860,160 | 81,920 |

The GPU PhysCal budget includes one initial warmup per independent walker.
Each backend reproduced its reported final energies across the three warmed
calls. Different group/walker structures produce different final energies;
these timing runs are not deterministic cross-backend parity tests.

## Supplemental cohort: 300 samples globally

This cohort was measured before the requested 4/8/16-rank comparison. It is
kept separate and was not used to compute the primary 320-sample ratios.

| Backend | Opt median (s) | PhysCal median (s) |
|---|---:|---:|
| CPU serial, one thread | 72.372369801 | 4.010374841 |
| GPU1, 300 walkers | 11.787850719 | 0.916507099 |
| MPI4, QP split, `NSplitSize=4`, one shared 300-sample chain | 23.683208837 | 1.452011373 |

The serial300/GPU300 elapsed ratios are 6.14× and 4.38×. QP mode distributes
QP sectors during sampling and the 300 configurations during measurement.
Its reported final CPU energies match the serial CPU300 output at printed
precision. Only QP4 was measured; QP8/QP16 were not. This input has eight QP
sectors, which limits useful QP partitioning.

## Container and numerical provenance

- Architecture: Linux x86_64; Ubuntu 24.04; Intel Xeon E5-2699 v3 @ 2.30 GHz,
  one socket, 18 physical cores and 36 logical CPUs. Cgroup `cpu.max=max 100000`,
  `cpuset.cpus.effective=0-35`; no container CPU or memory quota override.
- GPU: one NVIDIA GeForce RTX 3060, device 1, driver 580.178.04. Two GPUs were
  passed through, but each GPU timing uses only device 1.
- Julia 1.13.1, pinned `test/gpu` manifest; clean writable container depot with
  host packages/artifacts as read-only secondary storage, no host compiled
  cache. Julia BLAS: ILP64 `libopenblas64_.so` via libblastrampoline.
- CUDA.jl/CUDACore 6.4.2, MPI.jl 0.20.27, CUDA artifact runtime 13.4.0
  and compiler 13.4.92; complete library versions are recorded in
  [CUDA metadata](linux-x86_64-devcontainer-L128-whole-2026-10-12/CUDA-metadata.log).
- Native PfaPack rebuilt inside the container using GNU Fortran/G++ 13.3.0 and
  unchanged canonical flags; shared libraries link the container's LP64
  system OpenBLAS/BLAS/LAPACK. Native and Julia BLAS providers differ.
- CUDA.jl 6.4.2; artifact CUDA runtime 13.4.0/compiler 13.4.92,
  forward-compatible driver API for CUDA 13.3; MPI.jl 0.20.27.
- MPI: genuine system MPICH 4.2.0, `/opt/mpich/lib/libmpi.so` and
  `/opt/mpich/bin/mpiexec`; isolated project `MPIPreferences`. Native MPI
  startup tests verified real worlds 2 and 4; benchmark drivers verified
  requested worlds 1,4,8,16. Hydra `HYDRA_BINDING=core` placed rank r on logical
  CPUs r and r+18, i.e. one distinct physical core per rank. GPU host control
  and the serial300 baseline used `taskset -c 0,18`.
- Image ID: `sha256:b34e1fed725f040b2ef0e1b713edb187132ebaf1323c07fc3c652dada365811a`.
  Container ID: `c59c18feaaca252a5d8bacef5290d27288641e8b6abd07afc2cba9a0fc0ce248`.
- The temporary Dev Container configuration enabled GPU passthrough and
  read-only source/depot mounts and skipped postCreate to preserve the user's
  original submodule checkout. Sources were copied to a writable container
  cache. The original Rust workspace was not modified.
- The temporary Dockerfile changed only MPICH `make -j4` to available-CPU
  `make -j"$(nproc)"`; 36 build workers were verified. The initial four-worker
  build was cancelled cleanly before the replacement started. Julia package
  precompile used 36 tasks; the three independent canonical PfaPack compiler
  commands ran concurrently. Build parallelism does not change runtime
  Julia/BLAS thread limits.

The original Dockerfile SHA-256 is
`635a382af05e04dd55f368daf9bc0bace9eef5267a671b3b3e7dd064d957cafb`;
the measured temporary Dockerfile SHA-256 is
`e3e278b12003789cd850a3ec8cece94c27f1ce9962d30b6d24ae989caedbc90b`;
temporary configuration SHA-256 is
`884a724b12bb9bebfbfcf5d26919d7e22bff722b61f6ba417b651ac93d6f03de`.

All cells share fixed checkpoint SHA-256
`36cef06347facd5645882d229405bc81095d31731454d821df5fc1bcad014db8`.
The per-cell cloned ModPara hashes deliberately differ because sample counts
and NSplitSize select each work budget; they are recorded in the raw input
manifest. GPU320 and CPU1-rank320 use identical cloned ModPara contents.
The staged numerical-source manifest includes every Julia kernel/source and
native PfaPack source. Representative frozen hashes:

- `cuda_sr.jl`: `d67b26cd86b1ca8cf89eeaba7a2c7cf757106e027f75b990b17987de0ed0e7f0`
- `cuda_runners.jl`: `9ed158e38bd6945ea572931ea7262ca235fc23a83805b3c9cb769369353141d9`
- `cuda_sampling_update.jl`: `064b3e14a626309fe25556359a582995bf5e45266fd157e1aea822092eeed3b1`

The frozen host GPU regression suite passed 7,062 assertions in 44 testsets.
These container runs additionally verify successful finite public workloads,
real MPI worlds, fixed checkpoint hashes and same-backend repeatability; they
do not claim to rerun all host numerical regressions in the container.

## Reproduction and raw evidence

See [MPI/Dev Container reproduction](../../../test/gpu/README_MPI.md) for the
exact launcher, pinned environment, system-MPI preference, parallel native
build helper and commands. The validated MPI driver and launcher are committed
under `test/gpu`; the Linux build helper checks the canonical build-script
SHA before using its unchanged commands and honors `MVMC_BUILD_JOBS`.

Raw logs and source/input manifests are in
[the accompanying artifact directory](linux-x86_64-devcontainer-L128-whole-2026-10-12/):
`mpi-groups-{1,4,8,16}-total320.log`, `whole-gpu-total320.log`,
`whole-CPU1-GPU1-total300.log`, `mpi-qp-4-total300.log`,
`bootstrap.log`, `mpi-health.log`, `mpi-smoke-L6.log`,
`runtime-metadata.log`, `CUDA-metadata.log`, `build-helper-validation.log`,
`source-sha256.txt` and `input-sha256.txt`.

The bootstrap helper was hardened after timing to reject a changed canonical
build-script SHA and honor an affinity-aware positive `MVMC_BUILD_JOBS` budget.
Its compiler commands remained identical. The initial helper hash is retained
in runtime metadata alongside the final validated helper hash; post-timing
validation covers serial override, default36/min3 scheduling and both failure
guards. This build-only change does not alter any measured numerical source.

Archived text logs have trailing whitespace removed; timing values and source/input hashes are unchanged. `verified-medians.json` records medians independently checked against all three logged warm repetitions.

[Build-helper validation](linux-x86_64-devcontainer-L128-whole-2026-10-12/build-helper-validation.log) records successful one-job/default-job native builds and rejection of a zero budget or changed canonical build SHA.
