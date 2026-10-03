# Ordinary real direct-division repair (real-only checkpoint)

The ordinary real inverse now selects numerator/denominator division separately
from the FSZ matrix-backend policy. At this checkpoint ordinary complex retained
reciprocal/multiply; that operation-order gap was still open. FSZ retains its supplied complex quotient
and existing triangular backends. No LTL implementation is changed here.

`real_direct_solver_probe.cc` includes unchanged MPL-2.0 native
`extern/mVMC-1.3.0/src/ltl2inv/invert.tcc`, lines 11–28 (`sktdsmx`) and
the full `utu2inv` template. Source SHA-256:
`d477c7d2bb6d2b0c29872d06ec75fd7e804e1e30ace00888b77d31d2c74b821c`.
Probe SHA-256:
`ad44d79244708a1a067381d76c683bd0b52ac0a4cea728466787617f5f63446d`.
This optional developer probe is not invoked by Cargo.

Reproduce in the Linux reference container, with a fresh external directory:

```sh
task_stage=$(mktemp -d /tmp/mvmc-real-direct-native.XXXXXX)
gfortran -O0 -ffp-contract=off -J"$task_stage" -c extern/mVMC-1.3.0/src/ltl2inv/ilaenv_wrap.f90 -o "$task_stage/ilaenv.o"
g++ -std=c++17 -O0 -ffp-contract=off -DBLAS_EXTERNAL -Iextern/mVMC-1.3.0/src/ltl2inv -Iextern/mVMC-1.3.0/src/common -Iextern/mVMC-1.3.0/src/common/deps c_toolbox/issue184_inverse/real_direct_solver_probe.cc extern/mVMC-1.3.0/src/ltl2inv/ilaenv_lauum.cc "$task_stage/ilaenv.o" -lopenblas -lgfortran -o "$task_stage/probe"
OPENBLAS_NUM_THREADS=1 "$task_stage/probe"
```

Actual acquisition `/tmp/mvmc-real-direct-native.t1yw04`, GCC 13.3.0,
exit 0: binary SHA `4ddcd8af4ac0ddadad97888b0bd368e11d03187ddbe912569136f1849ac80675`,
native stdout SHA `af41a2a2b841fef51ddd4576535f2bc280c02c635e1b8147019ed480bd833619`.
Earlier compilations K5BlzP/fhrixa failed on missing include paths; not numerical
failures. This bounded acquisition does not capture a complete compiler/library
closure and is not full native model validation.

Six independent power-of-two/IEEE solver cases cover a huge finite divisor,
smallest subnormal divisor with zero and equal numerator, infinity divisor,
zero divisor, and infinity/infinity. Analytic expectations agree with native
classification. Two public native 2×2 inverses additionally show finite
`2^1023` has off-diagonal `±2^-1023` and zero diagonals, while the subnormal
divisor produces infinite off-diagonals and NaN diagonals after OpenBLAS TRMM.
The latter is not a supported model with finite inverse. Rust scalar TRMM's
first row has no dot product and retains zero; the BLAS feature follows the
observed native NaN propagation in that Linux acquisition only, not a universal
provider contract. These are explicit range classifications,
not a portable computed-bit acceptance policy or a tolerance change.

Old-public-path RED: immutable f296b2f1 checkout
`/tmp/mvmc-real-direct-red.PSrcaF/source`, isolated target, test-only addition,
nextest `b3480853-cd1f-4cfb-83a4-cba10b672b7e`, exit 100, 0/1 pass:
first scalar diagonal was NaN rather than zero. Current focused public test
passed `16db7b07-aed7-4af2-aad9-b23e63a9f4c1` before adding the explicitly
native-backed BLAS classification. That initial common-diagonal expectation
failed combined features (`061c3519-eb4f-4493-b7cc-d9833889c0de`, 17/18 pass);
the native probe above justified the backend-specific classification.

Final lib + existing inverse-contract gates, test-fast/locked/no-fail-fast/retries0:
default `cc8be0fe-44d5-457b-afe9-ac87199e7751` 16/16;
SIMD `c5fb56d2-aaf1-4c2b-990e-c55908891b62` 16/16;
BLAS `7294eb2e-b45c-4e94-8fcd-7bad680be092` 18/18;
combined `f3fbf5b4-9f21-4a0b-a7cb-b6c7dca07a5f` 18/18, all exit 0.
These gates exclude the unapproved loop/panel integration test and are not
whole-PfaPack gates. Core issue176 two integration binaries passed 16/16,
`607f0672-7f73-4067-8f90-da61edcfedf2` (before final public-test adjustment;
production policy unchanged). Shared checkout evidence is not a complete
immutable compiler-input closure. No performance improvement is claimed.

Final test design supersedes the temporary universal BLAS-NaN assertion above:
public finite divisors `2^1023` and `2^-1023` are tested with every feature profile.
The unsupported nonfinite tiny-divisor public diagnostic is scalar-only; internal
direct-solver IEEE cases still run in every profile. No supported finite case is
excluded. Final source SHA is
`f125cbd19cb11eba6605fb6c58a2d7531b11da573d5e6f3e561b11280cdb0553`;
Fresh isolated gates below are separate from the historical results above.
The snapshot `/tmp/mvmc-real-direct-final.Ypucyh` starts from committed
`384eebd8b6c3812b3f1e2c32b9c17248d22ea2f6` with only this utu2 source and
Ram's separately approved f64 LTL association source
`3496d3854253c45bf711a0d80a4633982a2add4fd762748877c1c60a03e00986`
overlaid. Unapproved loop/panel and uncommitted core176 drafts are absent.

Isolated terminal-0 PfaPack gates: default
`df5aea74-ea5c-49f4-92a0-c9c469c89b5b` 28/28 (0.010s), SIMD
`06d5742b-f124-4414-acd0-21b9efda2cc9` 28/28 (0.026s), BLAS
`b303bc71-1b60-4e67-9cc1-ce5b728e063b` 31/31 (0.082s), combined
`f37c072f-d8e8-4f5a-a3ec-7bd24a7d295d` 31/31 (0.064s). Each reports eight
existing feature-selected exclusions; no loop/panel draft exists in this
snapshot. Commands: `cargo nextest run -p pfapack --cargo-profile test-fast
--locked --no-fail-fast --retries 0`, adding `--features simd-backend`,
`--features blas-backend`, or `--features 'simd-backend blas-backend'`.
`CARGO_TARGET_DIR` points exclusively to the snapshot's target directory.
All-target/all-feature PfaPack Clippy with `--locked -- -D warnings` exited 0
(6.29s). The utu2, LTL and backend source hashes were unchanged at the end;
this is not a captured full compiler/library before-after closure. Feature
output is retained externally in the snapshot's `features.log`.

The complex follow-up required at this checkpoint was to decouple the same solver policy from FSZ
backends, using a reviewed scaled C quotient in the lower PfaPack crate, not
unscaled num-complex division. Sharing the existing GNU/compiler-rt pure-Rust
implementation requires preserving its license/provenance and reviewing its
platform selection, huge-finite, subnormal and nonfinite recovery tests.
This historical checkpoint did not change ordinary complex arithmetic.
The later implementation and bounded evidence are recorded in
[complex-direct-solver-README.md](complex-direct-solver-README.md); no broad parity claim is made.
