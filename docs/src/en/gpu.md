# Optional CUDA execution

CUDA.jl 6 enables an experimental, device-resident backend for both parameter
optimization and physical quantities. The existing CPU/SFMT runners are unchanged
and do not install or load CUDA. This backend targets batching many walkers; it
recomputes sampling Pfaffians on the GPU and reuses resident inverses for Green
functions and analytic derivatives. It assigns one CUDA lane to each walker, with
walker-first workspaces for coalesced memory access. See the
benchmark report for measured performance and the limits of this first kernel.

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
are attempted proposals **per walker**, not CPU NVMCWarmUp/NVMCInterval sweeps.
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

Setup parses inputs, initializes parameters and initial configurations on CPU,
constructs a linear Slater-element basis once using the established CPU algorithm,
and uploads the basis and model tables. Afterwards configurations, projection
counts, Slater elements, Pfaffian scratch, observables, log derivatives, and SR
matrices stay on the device. Parameter changes rebuild Slater elements with a
cuBLAS GEMV. Pfaffian cofactor sensitivities contract with the Slater basis using
cuBLAS GEMM, with resident alpha/beta scalars. SR moments use cuBLAS, and direct
SR uses cuSOLVER Cholesky.

Each optimization boundary reads one maximum-variance scalar, solver status and
an energy summary. C-compatible gauge shifts copy O(NPara) parameters to CPU and
back. PhysCal transfers its summary once at the end. No configuration, sampled O
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

The CUDA proposal is symmetric: select an electron and an arbitrary destination,
reject occupied/forbidden destinations, or select opposite-spin electrons for an
exchange and reject invalid exchanges. `NExUpdatePath=0/1/2` selects hopping,
a 50/50 mixture, or exchange. It differs from C's draw-until-empty proposal and
from SFMT stream initialization. Equal seeds **do not imply equal CPU/CUDA or
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
Avoid concurrent jobs on that same GPU. The scripts do not shard a single run
across two GPUs or claim multi-GPU speedup.

```sh
JULIA_MVMC_GPU_DEVICE=1 JULIA_MVMC_GPU_PROFILE=1 \
  julia +1.13.1 --project=test/gpu test/gpu/benchmark.jl
# Optional larger ordinary input:
JULIA_MVMC_GPU_DEVICE=1 julia +1.13.1 --project=test/gpu \
  test/gpu/benchmark.jl /path/to/hubbard_L16/namelist.def
```

See the [Linux RTX3060 measurements](gpu_benchmark.md) for timings and provenance.
The warmed profile covers sampling, observables and analytic derivatives and
checks the transfer-free inner region. Benchmark results record Julia, CUDA,
GPU/RNG and actual CPU BLAS configuration. CPU timing evaluates identical fixed
configurations and quantities, rather than comparing different MC trajectories.
