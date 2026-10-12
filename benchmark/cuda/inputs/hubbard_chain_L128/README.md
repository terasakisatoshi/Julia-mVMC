# L128 Hubbard chain input provenance

Half-filled128-site periodic Hubbard chain, U=4, t=1, Lsub=4, eight spin-projection
quadrature points, 512 orbital and260 projection parameters. Ordinary measured
operators:256 one-body and768 direct two-body Green functions, plus the energy.

Generated from `StdFace.def` with the pure-Rust StdFace port at
AtelierArith/mvmc-rs commit dd402cdc (base b1d617bccfb983d051202afe8990c833e697a275).
The source input is the committed L64 standard input at that base commit with
L and Ncond changed from64 to128. Generation is a separate developer command;
normal Julia GPU tests consume the checked-in Expert input files directly.

```sh
/path/to/mvmc --dry-run StdFace.def --out-dir .
```

Only the Expert inputs used by `namelist.def` and the generator input are kept.
Geometry/plot files are omitted. Benchmark scripts override walker batch size;
the300-step/300-sample values in ModPara are input provenance, not the number of
iterations measured by the fixed-configuration benchmark.
