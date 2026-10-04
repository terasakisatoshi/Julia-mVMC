# Independent C sampled CG refresh records

These three files are imported without modification from the Rust port's
optional C kernel oracle, documented in `c_toolbox/ctest_cg_refresh.md` of
https://github.com/AtelierArith/mvmc-rs . They are NOT Rust-generated results
or re-labelled historical Julia solver expectations. Input operand records
originated in archived Julia1.13.1 records; only means, diagonal, stored
samples and gradients are read by the C adapter, never archived solver output.

Authority: mVMC1.3.0 `src/mVMC/stcopt_cg_impl.c`, unchanged Main/operator
definitions (lines255–426) plus original license (lines1–21), SHA256
41452de5fe766409431c6e1cf73cd2b12485faaeb4bfbcdec1e53444e9af1cf9.
Extracted include SHA256
9f941cc5f9208eed0efc80937f292837c48b7bc7d078f467715ca001d9e9f43f.
Unchanged dot definition `stcopt_cg.c:26–34`, upstream SHA256
66a36cdff6f22e23367e12c08fee505daff6194447a667f5b89a65eb30130700;
include SHA256 f36c3f7d0aa0fc22e7e3cbffde3c95b79808112979e1e0ebc8fdd853d995d214.
Upstream https://github.com/issp-center-dev/mVMC; GPL-3.0-or-later,
copyright2016 University of Tokyo. Adapter SHA256
5e41bb737a018fda5d1ae014672954d5544f4f78fe0882d76be38e81ba77548e.
The post-allocation-guard adapter regenerated all three files with identical
hashes. No numerical C function body is patched.

Serial fixed-input kernel, NOT C initialization/sampling or MPI execution.
Generation: Dev Container73c57e563c61, Linuxx86_64, GCC13.3.0,
-O0 -ffp-contract=off, LP64 OpenBLAS pthread0.3.26, one thread;
library SHA256 bfc7492adbf84a8f567720a9e1fae2afc18f3d817da233e7f4d453683485308e.
Real executable c72f45946fab01d6de2b2afa66357b5149439428672a8a29c2add4762102783a;
complex executable969413856298b78745fcf74c3882fa5d9863bbc844a36bee91e0a31df77f1631.

| File | Input SHA256 | C output SHA256 |
| --- | --- | --- |
| real.txt | 95765162416342c86bace224de36dd315e5a0efa9ee1dd501556b8074df7c0a4 | e9c1372e2aa7b1ff12e15c44c74f5875030e622b80c985eacdee35c853122844 |
| complex.txt | 8258518a952cfe34db3a409ee9034a5c25905b7cf87ca0799b71ed4360c64eb5 | bd48dd44525c79082f6db667ff6dc6103f7c363328198044c8c839df230b80ea |
| sampled_complex.txt | 87829b292761dada69121c4d1fd175c0b4a71e029b5c6ae6dded2dcd2b80894f | 6036d436653df6c02846827194cfe82f43597e93f392f5dd52b932f2e166b590 |

Each file has shape n/sample-count/complex, six hexadecimal operand/product
rows, then solution/residual/direction for restarted limits1–41. Hex records
retain acquisition precision; computed assertions are numerical, not bitwise.
Tolerance0, shift1e-5, weight=sample count, both20/40 residual refreshes.

The exact adapter and both licensed upstream extracts are bundled alongside
these records, with the hashes above. This makes regeneration independent of
an unpublished Rust toolbox commit. Each output contains the original input
operands as its first six rows; the adapter consumes those operands only.
Optional regeneration from this directory in the recorded LP64 environment:

```sh
cc -O0 -ffp-contract=off -DMVMC_SRCG_REAL ctest_cg_refresh.c -lopenblas -lm -o /tmp/real-cg
cc -O0 -ffp-contract=off ctest_cg_refresh.c -lopenblas -lm -o /tmp/complex-cg
OPENBLAS_NUM_THREADS=1 /tmp/real-cg real.txt /tmp/real.txt
OPENBLAS_NUM_THREADS=1 /tmp/complex-cg complex.txt /tmp/complex.txt
OPENBLAS_NUM_THREADS=1 /tmp/complex-cg sampled_complex.txt /tmp/sampled_complex.txt
```

Normal Julia tests read these checked-in fixtures only, without a C compiler,
oracle invocation or Rust checkout. Julia1.13.1 validation uses actual
OpenBLAS0.3.30 ILP64/Haswell, one thread, local uncommitted Manifest-v1.13.
Unchanged C-refresh budgets: operator abs/rel8*(n+samples)*eps; iterate
abs/rel8*(n+samples)*limit*eps; independent explicit-Gram backward residual
abs16*(n+samples)*limit*eps*(abs(g)+sum(abs(A*x))), rel0.
There are14,673 assertions across all three cases and41 limits, including
explicit fixture shape and iteration-bound checks. The first
computed C-Julia difference was none on the observed Linux environment;
different BLAS environments must still pass the justified numerical bounds.
No new full sampling/MPI residual acceptance claim follows from this fixture.
