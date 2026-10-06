# Native DSKR2 cancellation boundary

Independent Linux native acquisition: `/tmp/mvmc-issue176-initializer.0RKnLG`.
Command: `OPENBLAS_NUM_THREADS=1 ./probe --rank2-boundary`.
The command above describes the historical acquisition. A checkout-only
optional reproduction is `bash c_toolbox/issue176_rank2/reproduce.sh`; see
that toolbox README for extraction boundaries, licenses, ABI and flags.
Its smaller adapter retains the same native kernels/input without requiring
the excluded initializer toolbox. It does not replace the original provenance.
The frozen environment and complete source hashes accompany `native.txt`.
Original `fortran/dskr2.f` SHA-256:
`57bf85cfabc22055af6c95d5765200ef5c27c9b5dff074da264fc76bbf3a56f2`.
Its lines 158–176 evaluate `(A + X*TEMP1) - Y*TEMP2`.

The independent input uses upper-triangle entries
`[2^54, 1, -2^54, 2^55, 1, 2^55]`. Native factor INFO is zero
and the Pfaffian is -1. Grouping the two update products first instead
reports INFO=1: the intended nonzero final pivot is lost. The Rust regression
therefore checks factor success as an algorithm contract, not computed-bitwise
parity. Its scalar Pfaffian check allows four machine epsilons, accounting for
the two dyadic diagonal factors and extraction sign; it is not a general inverse
or factorization error budget.

Native binary SHA-256:
`0b9fb96506341c1ee247fd1ea39699516f1680c52ed248194eeec3d275618716`.
This is a standalone kernel acquisition, not a full initializer/MPI proof.
Cargo reads checked-in expectations only and does not invoke the toolbox.
