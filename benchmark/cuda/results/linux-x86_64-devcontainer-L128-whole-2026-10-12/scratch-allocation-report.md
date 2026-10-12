# Scratch allocation verification and setup measurements

The candidate removes zero-fills for three large buffers that are fully gathered before use: parallel-initializer factors and the sampling/observable batched LU buffers. Initial PF tables remain zero-initialized because masked walkers must retain defined values. It also omits two unused transform buffers for rank-one Opt observables; general transforms and compact PhysCal retain their required storage.

At L128, 320 walkers and 8 QPs, each full matrix buffer contains 335,544,320 bytes (320 MiB). The two unused Opt buffers total 671,088,640 bytes (640 MiB). The three scratch changes avoid zero-filling 960 MiB during PhysCal setup; those scratch allocations themselves remain necessary.

The isolated candidate was based on `49c9bc7` and tested before production integration. Its four numerical source changes were subsequently integrated unchanged in `4eb9c69`. The baseline and candidate full source inventories, input/checkpoint hashes and exact tested patch accompany this report.

The candidate inventory was checked against production `4eb9c69`: all numerical source and test files match. Only `docs/make.jl` differs because the production branch added report rendering/link handling after the frozen baseline. Original inventory receipts are preserved unchanged.

## Correctness

All **367 assertions across 8 testsets passed**. Tests poison scratch factors/LU with NaN and cover full matrix gathers, including singular QPs; both initial-PF reuse and normal-refactor branches; repeated refresh; independent CPU inverse, residual and energy checks; and exact future RNG on fixed control paths. The explicit two-phase masked-to-retry test confirms that skipped QP matrices remain poisoned with untouched zero PF values, then initializes them on activation without clearing the workspace. Rank-one tests verify that unused internal transform buffers are empty while general transforms retain their buffers.

## Allocation-inclusive setup measurements

Measurements used GPU1 in the running Linux x86_64 Dev Container, Julia 1.13.1, one Julia/BLAS/OMP thread, `taskset -c 0,18` and `UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE`. Baseline workspace: `/home/vscode/.cache/mvmc/julia-whole-49c9bc7`; isolated candidate: `/tmp/julia-undef-scratch-test`. Input/checkpoint were the canonical fixed PhysCal files used by earlier phase profiling. B320, seed79 and all initialization work remained unchanged.

Each row reports seven warmed synchronized-wall repetitions. Parsing and compilation were excluded; constructors and their allocations were included. Full GC, synchronization and CUDA memory-pool reclaim ran outside clocks. Initialization, workspace construction and their combined path were measured separately, so their medians need not add exactly. No sampling, inverse refresh or SR step was timed here.

|Kind|Phase|Before median (ms)|After median (ms)|
|---|---|---:|---:|
|PhysCal|Initialization|108.738|108.595|
|PhysCal|Workspaces|7.168|7.203|
|PhysCal|Combined setup|117.358|116.424|
|Opt|Initialization|142.300|142.472|
|Opt|Workspaces|9.461|6.807|
|Opt|Combined setup|151.385|148.948|

PhysCal's combined median decreased by 0.934 ms, but the ranges overlap (before 117.196–121.967 ms; after 115.070–122.729 ms). This is **not a robust PhysCal performance gain**. Opt's combined median decreased by 2.436 ms, with nonoverlapping measured ranges (before 151.181–151.877 ms; after 148.717–149.396 ms). These are setup measurements, not whole-run speed claims. Configuration and future UInt32 RNG hashes match exactly before/after for both workloads.

## Excluded harness failure

The first microbenchmark retained temporary workspace results long enough to exhaust device memory during repeated setup. It failed after the first initialization phase. The corrected harness returns only elapsed time from a short-lived measurement function, drops temporary references, and performs GC/reclaim outside clocks. The first script and log are preserved as `scratch-allocation-excluded-oom-*`; **all measurements from that failed run are excluded**. This was a benchmark-lifetime failure, not a production numerical failure. Both corrected before/after processes exited successfully.

## Artifacts

- `scratch-allocation-focused-gates.log` and `scratch-allocation-focused-driver.jl`
- `scratch-allocation-setup-benchmark.jl`, `scratch-allocation-setup-before.log`, `scratch-allocation-setup-after.log`
- `scratch-allocation-before-source-sha256.txt`, `scratch-allocation-after-source-sha256.txt`, `scratch-allocation-input-sha256.txt`
- `scratch-allocation-tested.patch.b64` preserves the exact tested patch, including
  its blank context-line markers. Restore it with
  `base64 --decode scratch-allocation-tested.patch.b64 > scratch-allocation-tested.patch`.
- `scratch-allocation-excluded-oom.log`, `scratch-allocation-excluded-oom-benchmark.jl`
