# C CG small-SPD kernel fixture

Upstream mVMC1.3.0 src/mVMC/stcopt_cg_impl.c:258–351, complete
`fn_StochasticOptCG_Main` definition copied unchanged into the companion probe.
Source SHA256 41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9.
Probe SHA256 d7a0c22250081c9130a358380493afabfa865b05b682bde48c7c0aea481a4be1.
Origin https://github.com/issp-center-dev/mVMC; GPL-3.0-or-later, copyright
2016 University of Tokyo (inline notice included).

Only MPI rank/size and a one-dimensional sampled operator are adapters.
O=g=2^-54, mean O=0, W=1, stabilization=0, tolerance1e-30, limit1.
S=O*O=2^-108 is strictly SPD, condition number1. delta=2^-108 is above the
convergence threshold1e-60; d*S*d=2^-216 is nonzero. C:310 has no absolute
denominator guard. C:333/336 computes beta=dot(r,r)/delta then delta=beta*delta.
Output columns: iteration count, solution, residual, search direction.
Independent analytical solution is x=2^54.

Optional reproduction from repository root:

```sh
cc -std=c11 -O0 -ffp-contract=off \
  MVMCOptimizers.jl/test_unit/fixtures/c_cg_small_spd_probe.c -lm -o /tmp/c_cg_small_spd
/tmp/c_cg_small_spd
```

Linux x86_64, GCC13.3.0. Probe invokes no BLAS, no sampling, no RNG and no
MPI runtime; this is not a complete C sampled SR/MPI execution claim.
Regression uses Julia1.13.1, Manifest-v1.13 locally (not committed), actual
OpenBLAS0.3.30 ILP64, one thread. Both real and complex helper dispatches are
checked. Original Julia973184d gives zero solution and residual2^-54 (2 pass,
6 fail): first divergence is the premature denominator guard, not roundoff.
Numerical solution bound4.0 is one ulp at2^54; residual/direction bounds
eps(Float64)*2^-54 cover this single binary scalar step. No tolerance changes
are used to repair the premature exit. Longer recurrence/residual-refresh and
full MPI validation remain separate evidence requirements.
