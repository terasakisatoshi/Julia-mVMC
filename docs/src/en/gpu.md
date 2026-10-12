# Optional CUDA execution

CUDA.jl 6 enables an experimental, device-resident backend for both parameter
optimization and physical quantities. The existing CPU/SFMT runners are unchanged
and do not install or load CUDA. This backend targets batching many walkers; it
updates sampling Pfaffians and inverses on the GPU and reuses them for Green
functions and analytic derivatives. Large lattices distribute projection sectors
across CUDA lanes, with walker-first workspaces for coalesced memory access.
See the benchmark report for complete-run timings and their sampling scope.

## Install and run

Use Julia 1.13.1 with this checkout and initialized submodules. The GPU test project
keeps CUDA separate from the CPU workspace:

```sh
julia +1.13.1 --project=test/gpu -e 'using Pkg; Pkg.instantiate(); Pkg.build(["SFMT", "PfaPack"])'
julia +1.13.1 --project=test/gpu
```

```julia
using CUDA, MVMCOptimizers
CUDA.allowscalar(false)
input = "examples/inputs/hubbard_chain_real/namelist.def"
opt = run_gpu_para_opt_from_namelist(input;
    nsteps=20, walkers=256, device=0, seed=79, warmup=100, interval=20)
phys = run_gpu_phys_cal_from_namelist(input;
    nsamples=20, walkers=256, device=0, seed=79, warmup=100, interval=20)
```

`initial_def=:auto` reads a sibling `initial.def` when present, followed by the
namelist's In*.def overlays. PhysCal additionally accepts `opt_para="zqp_opt.dat"`
for fixed optimized parameters. An omitted seed follows the input's `RndSeed` and
the CPU serial seed-resolution rule. `device` selects a zero-based CUDA device.

`walkers` is the number of resident independent chains. `warmup` and `interval`
are attempted proposals **per walker** when explicitly provided. When omitted,
the namelist runners use `walkers=NVMCSample`,
`interval=NVMCInterval*Nsite`, and
`warmup=NVMCWarmUp*NVMCInterval*Nsite`; PhysCal defaults its iteration count to
`NDataQtySmp`. Each independent chain receives its own thermalization.
Each optimization step uses the current batch of walkers. Each PhysCal sampling
iteration averages that batch, and the final result averages these iterations.
Autocorrelation and equilibration still require choosing suitable lengths.

Optimization returns `status`, per-step `energies`, final `parameters`, `data`,
and the resident `state`. PhysCal returns `energy`, `green_one`, `green_two`,
`green_two_ex`, `data`, and `state`. Green arrays retain the input term order;
factored two-body values average products of per-configuration one-body values.
These GPU runners return numerical results rather than writing the CPU runners'
C-format output files. Optional callbacks receive resident buffers, and callers
control any callback transfers. `gpu_para_opt!` and `gpu_phys_cal!` also accept
already initialized `ExpertModeData` with a host matrix of initial configurations
(zero-based sites; up electrons followed by down electrons; one walker per column).

## Supported input contract

The first GPU runner supports real, sz-conserved orbitals with `2Sz=0`, Gutzwiller
and Jastrow factors, fixed spin/translation quantum projections, and the ordinary
Transfer, CoulombIntra/Inter, Hund, PairHop, Exchange and spin-conserving InterAll
Hamiltonian terms. One-body, direct two-body, and factored/product two-body Green
functions are supported. Local spins stay singly occupied. The direct SR path
uses `NSRCG=0`, the existing optimization flags, DSROptRedCut variance filtering,
diagonal stabilization and step width, and C-compatible parameter gauge shifts.

FSZ/general orbitals, complex runner inputs, RBM, doublon-holon, variational
OptTrans, Lanczos, and MPI split execution are rejected. Complex batched Pfaffian
kernels are available independently and tested against the CPU factorization;
they do not imply support for complex end-to-end GPU runs.

## Residency and arithmetic

Setup parses inputs and initializes parameters on CPU. Large lattices generate
and validate initial configurations on GPU, using unbiased UInt32 bounded draws
and the C startup requirement that every projection sector be nonsingular.
Only a pending-walker count is read at initialization retries. The large-lattice
Slater derivative basis is constructed directly as CSC from the C four-block
coefficients; the L128 basis occupies about 12.44 MB instead of 3.24 GB. Small
lattices retain the CPU configuration and dense basis reference paths. Afterwards, configurations, projection
counts, Slater elements, Pfaffian scratch, observables, log derivatives, and SR
matrices stay on the device. Parameter changes rebuild Slater elements with a
resident kernel using C's four spin-block expressions and explicit round-to-nearest
operations. This avoids reassociation from basis GEMV and implicit multiply/add
contraction. Small lattices contract cofactors with the dense basis through
cuBLAS GEMM; large lattices use resident CSC coefficients and fuse cofactor
evaluation with contraction. SR moments use cuBLAS, and direct
SR uses cuSOLVER Cholesky.

Large cached hopping updates compute the upper inverse triangle and publish the
lower triangle by its skew sign. Matrix publication uses shared-memory tiles.
Optimization with only one-body and diagonal two-body terms skips the full
Slater transform. PhysCal omits the derivative-only basis and, when only a few
transform columns are required, computes those columns and obtains other
entries from ordered inverse-row dot products. General operators and singular
sectors retain their full-transform or cofactor paths. Large PhysCal batches
initialize projection sectors in parallel, retaining the ordered projection
sum and the original random draw sequence.

Variance cutoffs, ordered parameter gauge shifts, solver status and energy
history stay on GPU throughout optimization. Native cuSOLVER status is checked
once before final reporting; failed solves do not update parameters. The final
parameter and energy summaries are transferred on return. PhysCal transfers its
summary once at the end. No configuration, sampled O
matrix, Green matrix or SR matrix is copied to the host in the numerical loops.
No CPU scalar indexing of CuArrays is allowed in the verification suite.

Log derivatives are analytic: each projection sector uses Pfaffian cofactors
obtained from a pivoted inverse. A singular individual sector uses exact Pfaffian
minors so its nonzero derivative is retained inside the projected sum.
Floating-point comparisons use explicit tolerances, not bitwise equality. The
GPU inverse algorithm, cuBLAS reductions, and cuSOLVER differ from CPU operation
order; numerical acceptance differences can subsequently change trajectories.

## RNG and compatibility verification

In the verified CUDA.jl 6.4.2 environment, `CUDA.rand(UInt32, ...)` uses the
cuRAND library's default XORWOW generator. Five UInt32 words per attempted
proposal are generated directly into a device tape (electron, spin, destination,
move type, acceptance). Rejected and invalid proposals consume that fixed packet.
UInt32 acceptance words map to `[0,1)` by division by 2^32. This tape is never
copied to the host during production sampling.

Large-lattice runners select an electron and rank a destination among allowed
empty sites, or select opposite-spin electrons for an exchange and reject invalid
exchanges. Exact integer projection-count deltas avoid full recounting per hop.
A resident Pfaffian/inverse cache applies C-order hopping updates; batched native
LU refreshes it at parameter boundaries. Observables reuse those inverse and
Pfaffian values, retaining the exact minor fallback for singular sectors. `NExUpdatePath=0/1/2` selects hopping,
a 50/50 mixture, or exchange. The fixed-word GPU stream differs from C's draw-until-empty SFMT stream.
The uniform-empty hopping distribution agrees for the L128 half-filled Hubbard
workload; independent chains and thermalization budgets differ from the CPU
sequential sampling lifecycle. Equal seeds **do not imply equal CPU/CUDA or
Julia/Rust trajectories**. Match generator state initialization, stream/output
ordering and conversion before claiming cross-language RNG equality.

`gpu_sample!` accepts a device UInt32 tape for deterministic verification. The
CPU oracle uses existing CPU C-order Pfaffian/inverse, Hamiltonian, Green and
Slater-derivative routines, with the same configurations and proposal packets.
Tests compare every proposal/acceptance step on 20-step shared-tape paths, exact
configuration/draw packets, numerically close overlaps/observables/SR updates,
solver residuals and 20-step same-implementation production repeatability.

Sparse block-skew matrices and individual nodal QP sectors additionally use
closed-form mathematical checks. The standalone Pfaffian kernel retains zero
odd elimination columns (only even pivots enter its product); the CPU full-normal
factorization may report a nonzero `info` for such columns. These synthetic
mathematical cases are distinguished from the CPU-reference parity fixtures.

The L128 regression includes a configuration with a projected-sum cancellation
condition number above 2.5 million. Explicitly rounded real Pfaffian updates and
ordered weight accumulation retain the existing 1e-9 comparison threshold.
Raw tiny orbital parameters remain linear in the GPU/C expressions; the historical
Julia orbital-value builder's 1e-14 filtering is tested separately using C closed
forms and is not presented as a CPU-reference match for that exceptional case.

```sh
JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --project=test/gpu test/gpu/pfaffians.jl
JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --project=test/gpu test/gpu/wavefunctions.jl
JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --project=test/gpu test/gpu/parity.jl
# All GPU gates, including operator coincidences and nodal derivatives:
JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --project=test/gpu test/gpu/runtests.jl
```

## Two-device validation and benchmarking

Use device 0 for correctness/debugging and device 1 for profiling/benchmarking.
A separate process selects device 1, warms up compilation and all measured
library paths, synchronizes and records CUDA-event as well as host-wall timings.
Avoid concurrent jobs on that same GPU. Each optimization/PhysCal runner owns
one device. `test/gpu/benchmark_multi_gpu.jl` additionally splits an identical
fixed configuration batch across devices0/1 with persistent Julia workers;
its wall time includes dispatch and synchronization. It does not measure a
distributed SR solver or a complete two-device Monte Carlo run.

```sh
JULIA_MVMC_GPU_DEVICE=1 JULIA_MVMC_GPU_PROFILE=1 \
  julia +1.13.1 --project=test/gpu test/gpu/benchmark.jl
# Optional larger ordinary input:
JULIA_MVMC_GPU_DEVICE=1 julia +1.13.1 --project=test/gpu \
  test/gpu/benchmark.jl /path/to/hubbard_L16/namelist.def
```

See the [Linux RTX3060 measurements](gpu_benchmark.md) for timings and provenance.
The checked-in L128 Hubbard input and `JULIA_MVMC_GPU_BENCH_STEPS` option support
large-lattice measurements. Large matrices use a cooperative Pfaffian, resident
cuBLAS batched LU/inversion, transformed unoccupied Slater columns and independent
operator evaluation. Occupied columns use the exact identity `A^-1*S_occ = -I`.
Sparse analytic derivatives contract cofactors directly from device inverse
tables; singular QPs retain exact minor calculations. Large lattices build and
upload the sparse Slater derivative basis during setup.
`JULIA_MVMC_GPU_INVERSE=cooperative` selects the Gauss-Jordan debug comparison.
The warmed profile covers sampling, observables and analytic derivatives and
checks the transfer-free inner region. Benchmark results record Julia, CUDA,
GPU/RNG and actual CPU BLAS configuration. The historical kernel benchmark evaluates identical fixed configurations. The
complete-run benchmark additionally measures initialization, sampling, SR and
final results, using equal saved sample totals but different CPU/GPU chains and
thermalization budgets.

Large-lattice debug paths can be selected independently with
`JULIA_MVMC_GPU_INITIALIZATION=cpu`, `JULIA_MVMC_GPU_SAMPLING=refactor`,
`JULIA_MVMC_GPU_PROPOSALS=self-loop`, `JULIA_MVMC_GPU_CORRELATION=full`, and
`JULIA_MVMC_GPU_OBSERVABLES=refactor`. Defaults are `gpu`, `update`, `empty`,
`delta`, and `reuse` for the large runner. Shared tape tests keep the low-level
legacy proposal mode as their independent reference.

The whole-run Dev Container comparison uses L128, 320 retained samples globally,
20 optimization steps and one PhysCal run. The initial three-repeat GPU result
was 12.55 s / 0.934 s versus a one-rank CPU baseline of 77.14 s / 4.465 s.
After the additional optimizations, a separate five-repeat confirmation measured
one GPU at 10.48 s / 0.761 s and MPI with 16 independent groups at
11.21 s / 0.781 s. Both GPU medians are lower in that confirmation, but the
PhysCal timing ranges overlap. The 20× CPU target remains unmet.
See the [complete measurements](gpu_benchmark.md) for every revision and
repetition, short-warmup, chain, output and proposal-budget differences, and
the recorded source hashes.
