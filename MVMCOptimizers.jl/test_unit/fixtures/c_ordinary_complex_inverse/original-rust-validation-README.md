# Shared C quotient and ordinary complex direct solve

This milestone changes ordinary PfaPack complex `sktdsmx` from a reciprocal
followed by multiplication to a numerator/denominator quotient. C's supported
arithmetic order is authoritative; the earlier difference is not attributed
solely to BLAS or mathematical-function rounding. Solver quotient selection is
separate from the FSZ triangular-matrix backend flag. Ordinary real keeps its
direct quotient; FSZ keeps its explicit caller-supplied quotient and backends.
No LTL arithmetic is modified by this ownership scope.

The previously reviewed core quotient is shared in the lower PfaPack crate,
without a core dependency or C FFI:

- GNU/Linux: GCC 13.3.0 `libgcc/libgcc2.c`, `L_divdc3`, GPL-3.0-or-later
  WITH GCC-exception-3.1. Original core source SHA
  `928a0fc7cde1b507af1efa99407b908fe899631307ef134d45c304ddcb93b448`;
  moved source SHA
  `81ce68241a6cabe9fec4aa4d11d586cfeca1e177140fe34b3e1759bee82cc077`.
- Other platforms: LLVM 17 compiler-rt `divdc3.c`, Apache-2.0 WITH
  LLVM-exception, preserving the existing exponent scaling, recovery and
  macOS ARM FMA selection. Original core source SHA
  `a01be8efe7e9276109742bb10013b1276504ac501e1305d21d506c32d89b7ae3`;
  moved production source SHA
  `7076f47e8676b6b5ae2cc069874806e4a9d7bf4f795e657516bbb0c16a93f45d`.

Before deleting the old core GNU file, its entire text was compared with the
new file after reversing the visibility/doc-only edits: equality was true.
The moved LLVM production text likewise equals the old pre-test text after
reversing visibility and module-path changes and removing the original single
trailing blank line before the retained tests. Arithmetic expressions, cfgs and
constants are unchanged. Both full license copies compare byte-identical:
GCC exception SHA `28e85c5aa4af9b4f1dfe6b4817aa3eefb3eaaee7fd735045016f29ccf50276a1`,
LLVM SHA `3340babe8ac7bc6ae294d93aa01c310a250d43d5b760e5c12954882d4e5c83c7`.
Core retains its notices and re-exports the same numerical utility; its original
fixture test is unchanged. Fixtures under `tests/fixtures/interall/` are used
only for quotient utility checks, not an InterAll implementation claim.

The new offline integration test retains the existing quotient policy
`16*EPSILON` relative plus four subnormal quanta absolute, with exact infinity
classification/sign and NaN classification. It adds ordinary public complex
2×2 analytic finite cases with divisor `(2^1023,2^1023)` and
`(2^-1023,2^-1023)`, whose inverses have components `±2^-1024` and `±2^1022`.
Existing independent inverse fixtures check all A/M/vT planes, poisoned/reused
workspaces, nonidentity pivots, actual factorization and operator residuals.
No new tolerance or Rust-generated expectations are introduced.

Initial shared-checkout results (not complete immutable compiler closure):
default lib/inverse `9d69dc0f-9d14-4358-ad1e-c446a33f9663` 16/16;
new quotient/public-range `5b13ffda-4c96-4831-bf27-8c5122332309` 2/2;
retained core quotient `98362e46-f33e-4b5a-9028-50efe7c1c5c8` 1/1.
Shared feature gates `7e35438d`, `c180a6fb`, `c4f0c0e9` passed 31/34/34;
these include another owner's uncommitted rank2 test and exclude the unapproved
loop/panel test. They are not the isolated proof below.

Isolated snapshot `/tmp/mvmc-complex-direct-final.tz2M0x` has been checked against
commit `7e7cea64c72dfaf91e4330113bc229158b8981b1`: exactly eight tracked
deviations (quotient move/re-export, PfaPack policy/lib/manifest, notices, and
Ram's separately approved f64 LTL association), plus the five new quotient,
license and integration-test files. Unapproved loop/panel and core176 drafts
are absent. Compiled snapshot utu2 SHA is
`4349ee2ae8c5c1b53eeac6345d0fab79c300de493c1016db1db4f359f7399f4a`.
Its exclusive target is inside that snapshot.
Subsequent rustfmt-only call layout gives shared source SHA
`225c3d0d055c820c57280a920599c708bbcf7325f4886ee063d674b334b7666a`;
the existing snapshot is unchanged and results are not retrospectively relabelled.

Terminal-0 isolated PfaPack commands:
`cargo nextest run -p pfapack --cargo-profile test-fast --locked
--no-fail-fast --retries 0`, adding each requested feature selection:
default `92190005-5bf2-438e-988c-1d68347241d5` 30/30 (0.022s),
SIMD `ffeb6fe1-0d26-4947-9022-49eb24a8bb6c` 30/30 (0.029s),
BLAS `b064a594-ffc2-4d81-9891-3c08d4937276` 33/33 (0.063s),
combined `9871a83b-eee0-4f98-9cfd-81f76e3487fc` 33/33 (0.055s).
Each has eight existing feature-selected exclusions; no unapproved draft is
present to be filtered. PfaPack all-feature doctest command exited 0 with zero
doctest cases (compilation only). Isolated core quotient command
`cargo nextest run -p mvmc-core --lib --cargo-profile test-fast --locked
-E 'test(scaled_complex_quotients_and_range_recovery_match_native_c)'
--no-fail-fast --retries 0` exited 0:
`87ff996f-a9b5-4e27-a5eb-9591cccddae2`, 1/1 (216 unselected), 0.019s.
The existing inverse consumers command
`cargo nextest run -p mvmc-core --test calc_m_all_vs_julia --test pfaffian_cg
--cargo-profile test-fast --locked --no-fail-fast --retries 0` exited 0:
`ff1434eb-0465-4baa-ab85-f2d75a9911b8`, 12/12, 0.024s.
PfaPack `cargo clippy -p pfapack --all-targets --all-features --locked
-- -D warnings` exited 0 (5.49s). All 13 retained entries in external
`owned-source-sha256.txt` still match, including the unchanged backend.
That manifest was captured during compilation, not a full before-all-gates
compiler/library closure. External `gates.log` SHA is
`e3c66baf36ffe6dc41e9546ad53c792471e7ca561266db128845efbd23087f72`;
it retains the four PfaPack runs. Core and Clippy terminal results above were
captured by the execution tool, not falsely labelled a native log file.
Isolated `cargo test -p mvmc-core --doc --locked` also exited 0 after
1m18s build, zero doctest cases (compilation only).
Core176 final numerical checks are owned separately and are not claimed here. Native macOS results,
full workspace success, general panel error budgets and performance improvement
are not claimed. Earlier reciprocal scalar traces/conditional majorants describe
pre-repair arithmetic and are not adopted for the new direct solver.
