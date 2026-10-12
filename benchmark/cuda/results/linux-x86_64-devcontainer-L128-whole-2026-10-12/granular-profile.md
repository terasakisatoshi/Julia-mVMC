# Dev Container L128 B320 granular refresh and observables

Temporary diagnostic scripts only; no production algorithm or source changed. Measurements use the staged numerical source corresponding to the earlier frozen `63ddd84` profile checkpoint (numerical implementation `dcb36b570389a2426369361b41f27b3d8e0e6528`), the same 320-sample cloned input, and an actual state produced by 20 public optimization steps. Representative source hashes remain cuda_runners=9ed158e38bd6945ea572931ea7262ca235fc23a83805b3c9cb769369353141d9, cuda_sr=d67b26cd86b1ca8cf89eeaba7a2c7cf757106e027f75b990b17987de0ed0e7f0, cuda_sampling_update=064b3e14a626309fe25556359a582995bf5e45266fd157e1aea822092eeed3b1.

Container c59c18feaaca, Julia 1.13.1, GPU 1 RTX3060, CPU affinity 0,18, one Julia/BLAS thread, FP64. All reported stage/candidate times are synchronized wall seconds, not CUDA event durations. Five warmed repetitions use unchanged resident matrices; copies/reset/parity checks occur outside candidate clocks. Main whole-run baseline medians must not be reattributed to these scratch variants.

## Native refresh

There are 2,560 matrices (320 walkers × eight QPs), each 128×128.

| Stage | Median synchronized wall milliseconds |
|---|---:|
| Fill LU and PF factors | 2.177 |
| Cooperative Pfaffians, production 128 threads / cap64 | 98.770 |
| Native getrf batched | 41.960 |
| Native getri batched | 69.109 |
| Project inverse to skew form | 3.647 |
| Publish packed inverse/valid/PF cache | 2.105 |
| Publish overlap plus full correlation | 1.861 |
| Sum of production stages | 219.629 |

The independent diagnostics split the final publication into 0.038 ms overlap/counter work and 1.681 ms correlation work. These diagnostic timings are not added again to the production-stage sum. The result agrees with the whole-run 4.439 s inverse-refresh budget over 20 steps (about 222 ms per boundary).

## PF launch/static-index experiment: no adopted winner

The unchanged cooperative helper was tested at 32/64/128/256 threads and register caps32/48/64/128. A scratch wrapper also supplied constant128 packed matrix/pivot strides and a constant n while calling the identical RN/pivot helper. No numerical arithmetic was rewritten.

Production-reference candidate128/cap64 measured99.357 ms. Dynamic64/cap64 measured98.523 ms, only0.84% lower and within the baseline variation seen across probes. This is not a convincing adoption result; retain128/cap64. The static-stride wrapper at128/cap64 measured131.214 ms, so discard it. Larger blocks or tighter caps did not provide a material win.

All32 candidates passed96 assertions: exact integer pivot arrays, PF values at atol1e-10/rtol1e-12, and full factor normalized error<=1e-12. The observed GPU factor normalized error was0.0 for every candidate; floating-point results were not asserted bitwise. Six resident matrices were also checked against independent CPU C-order factorization/products/pivots, adding24 assertions. This is matrix-kernel parity, not a new Monte Carlo trajectory gate. No sampler or RNG entrypoint is called during refresh/PF tuning, and no production variant was adopted.

## Observable attribution

| Stage | Opt, milliseconds | Phys-style Greens, milliseconds |
|---|---:|---:|
| Publish sampler cache to observable/native layouts | 10.393 | 10.394 |
| Target-site map | 0.132 | 0.130 |
| Fill unoccupied Slater rows | 1.905 | 1.902 |
| Transformed-Slater batched GEMM | 57.692 | 57.693 |
| Publish transformed cache | 26.936 | 26.766 |
| Parallel operator evaluation | 5.171 | 11.146 |
| Ordered reduction | 0.094 | 0.329 |
| Scalar singular-QP fallback launch | 0.032 | 0.043 |

The Opt workspace has640 terms and one output row. The Phys-style workspace has1,664 terms and1,025 output rows. Every QP cache was valid and all output values finite for this frozen evolved optimization state. The Phys-style column attributes the full Green workspace on that state; it does not replace the independently initialized fixed-parameter PhysCal whole-run benchmark.

The two layout-publication stages total37.3 ms per evaluation, substantially more than the operator stage. A coalesced/tiled data-movement prototype is therefore a measured opportunity, while a target_map electron-label lookup can affect only the5.17/11.15 ms operator portion. Neither improvement is assumed until parity and actual timing pass. Native LU/inversion remains another measured target; PF tuning alone found no material gain.

## Runtime and scratch failures

The initial CUDA-event diagnostic process hit the known UCX SIGSEGV-handler/Julia-GC collision before collecting stage results. The successful wall-only run uses UCX_ERROR_SIGNALS=SIGILL,SIGBUS,SIGFPE, excluding SIGSEGV. A scratch launch=false call initially included a launch-time threads keyword; CUDA rejected it, and it was corrected by supplying threads only when invoking the compiled kernel. These are runtime/developer-probe failures, not production numerical failures. Neither failed attempt supplies any adopted performance conclusion.

Successful evidence:

- [Refresh/PF probe source](refresh-pf-probe-source.jl)
- [Refresh/PF probe log](refresh-pf-probe.log)
- [Observable probe source](observable-parts-source.jl)
- [Observable probe log](observable-parts.log)

These are exact scratch-source snapshots with measured temporary input paths,
not public benchmark drivers. [SHA-256 records](granular-profile-sha256.txt)
retain the original `/tmp` names.

Separate failed/partial evidence:

- /tmp/profile-L128-refresh-pf.jl and /tmp/profile-L128-refresh-pf.log (runtime signal collision)
- /tmp/profile-L128-refresh-pf-wall.jl and its container log (stage results/24 checks followed by scratch macro-contract error)
