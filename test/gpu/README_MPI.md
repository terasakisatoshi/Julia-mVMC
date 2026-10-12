# CPU MPI whole-run benchmark

These optional drivers measure the public CPU optimization and PhysCal runners.
Use the pinned `test/gpu` environment with Julia 1.13.1 and one Julia/BLAS thread.
The launcher uses the MPI provider selected by that project and verifies the
requested world size; it records rank affinity and the actual MPI/BLAS versions.

The Linux Dev Container supplies MPICH 4.2.0. Install Julia with the Rust
workspace's `.devcontainer/install-julia.sh`, copy the Julia checkout into a
writable container directory, and select system MPI only in that isolated copy:

```sh
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)
julia --project=test/gpu -e 'using Pkg; Pkg.instantiate(;allow_build=false,allow_autoprecomp=false); P=Base.require(Base.PkgId(Base.UUID("3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"),"MPIPreferences")); P.use_system_binary(;library_names=["/opt/mpich/lib/libmpi.so"],mpiexec="/opt/mpich/bin/mpiexec",abi="MPICH",force=true)'
julia --project=test/gpu test/gpu/build_pfapack_parallel.jl PfaPack.jl/deps
julia --project=test/gpu -e 'using Pkg; Pkg.precompile()'
```

The optional Linux native build helper runs the three independent canonical
PfaPack compiler commands concurrently, preserving their sources, flags and
link order. Package precompile uses all available CPUs; benchmark execution
uses one thread per rank. Keep all builds and other benchmarks idle during
measurements. Do not run a host compiled-cache directory inside the container.

Run a small two-rank smoke first, then the L128 workload:

```sh
julia --project=test/gpu --threads=1 test/gpu/launch_mpi.jl 2 test/gpu/benchmark_mpi_end_to_end.jl \
  --input test/integration/reference/hubbard_chain_real/inputs/namelist.def \
  --mode qp --samples 8 --steps 2 --reps 1 --warmup 1 --interval 1 \
  --checkpoint auto --output /tmp/mpi-smoke-L6

for ranks in 1 4 8 16; do
  julia --project=test/gpu --threads=1 test/gpu/launch_mpi.jl "$ranks" test/gpu/benchmark_mpi_end_to_end.jl \
    --input benchmark/cuda/inputs/hubbard_chain_L128/namelist.def \
    --mode groups --samples 320 --steps 20 --reps 3 --warmup 1 --interval 1 \
    --checkpoint auto --output "/tmp/mpi-groups-L128-p$ranks"
done
```

Use a fresh output directory for each invocation. The primary report uses independent groups at ranks4,8,16 with320 retained
samples globally; QP mode at4 ranks and300 samples is supplemental.
`--mode qp` sets `NSplitSize=P`: one shared chain with 300 retained samples per
step globally, QP sectors split during sampling, and samples split during
measurement. MPI collectives and the replicated direct SR solve remain inside
the timed calls. Eight QP sectors limit useful sampling partitioning to eight
ranks for this input.

For the main independent-group comparison use `--mode groups --samples 320`
and fresh output directories at ranks4,8,16. `NSplitSize=1` assigns80,40,20
samples to each group respectively, preserving320 retained samples globally
and the existing C base+group seed convention. Compare these cells against
the separately measured CPU1/GPU1 320-sample baseline, not the300-sample
supplement. The cloned ModPara deliberately changes NSplitSize and NVMCSample
to select that budget; its hash therefore differs from the original input
and from QP mode.

Each timing wraps a full public runner call, including parsing, setup, Monte
Carlo, observables, SR for optimization, ordinary synchronization and output.
The driver reports the maximum rank wall time. The initial barrier, final
MPI timing/status reductions and extra PhysCal output-file readback are outside
the clock. Module import/MPI launch are excluded; one cold execution with
compilation and three warmed executions are reported separately.

CPU/MPI use one sequential SFMT chain per group; GPU uses independent XORWOW
walkers. Initial warmup, accepted moves, RNG trajectories and output behavior
differ. Equal retained-sample/optimization budgets support lifecycle elapsed
time comparisons, not claims of equivalent trajectories or effective sample
speedup. The driver copies inputs and prepares a fixed PhysCal checkpoint
outside timing; it does not modify the input checkout.

Measure the matching single-GPU 320-sample cell separately:

```sh
JULIA_MVMC_MPI=0 JULIA_MVMC_E2E_BACKEND=gpu JULIA_MVMC_GPU_DEVICE=1 \
JULIA_MVMC_E2E_SAMPLES=320 JULIA_MVMC_E2E_STEPS=20 JULIA_MVMC_E2E_REPS=3 \
JULIA_MVMC_E2E_WARMUP=1 JULIA_MVMC_E2E_INTERVAL=1 \
taskset -c 0,18 julia --project=test/gpu --threads=1 test/gpu/benchmark_end_to_end.jl \
  benchmark/cuda/inputs/hubbard_chain_L128/namelist.def
```

The example affinity matches the report's machine: CPUs0,18 are two logical
threads of physical core0. Adapt binding to the measured machine topology.
Hydra defaults to `HYDRA_BINDING=core`; inspect the logged per-rank CPU lists.
Compare the logged fixed-parameter checkpoint hashes across all cells.
