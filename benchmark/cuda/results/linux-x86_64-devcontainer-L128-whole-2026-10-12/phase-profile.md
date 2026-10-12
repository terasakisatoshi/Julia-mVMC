# L128 B320 phase budget in the measured Dev Container

Container c59c18feaaca; staged source from `63ddd84e2f47d535811b505227716ed9abce2e81` (numerical implementation `dcb36b570389a2426369361b41f27b3d8e0e6528`); Julia1.13.1, FP64, GPU1, CPU affinity0,18, Julia/BLAS1thread, XORWOW seed79. Same cloned320 input/fixed PhysCal checkpoint as the main comparison. No production source was edited. Every phase ends in CUDA.synchronize; these measurements are diagnostic rather than replacements for the public whole-run table.

Three instrumented warmed repetitions followed one instrumented cold compilation run. The first uninstrumented public wrapper invocation includes first-use wrapper compilation and must be excluded from warmed overhead comparison. The remaining two fully warmed public calls have medians12.571435s Opt and0.950203s Phys. Instrumented medians are12.586871s and0.939047s. Opt difference is15.44ms (0.12%); the negative11.16ms Phys difference is ordinary measurement variation, not an instrumentation speedup. The original primary report records12.548162s and0.933714s.

| Stage | Opt20steps median seconds | Phys1run median seconds |
|---|---:|---:|
| Parse/parameter setup |0.079357|0.079730|
| Initial GPU walkers |0.151015|0.150708|
| Observable workspace |0.006057|0.006096|
| Sampling workspace |0.002286|0.002288|
| SR workspace |0.004087|—|
| Initial inverse refresh |0.121236|0.121837|
| Initial128-proposal warmup |0.235443|0.237548|
| Sampling intervals |4.622742|0.231644|
| Observables |2.053979|0.108665|
| Derivatives |0.713904|—|
| Moments |0.054656|—|
| Native SR |0.095652|—|
| Parameter gauge |0.001179|—|
| Slater refresh |0.001352|—|
| Post-SR inverse refresh |4.439187|—|
| Final summary |0.000413|0.000033|
| Other validation/history/accumulation |<0.0011|<0.0004|
| Instrumented full wall |12.586871|0.939047|

Optimization sampling is36.7% of the total, inverse refresh35.3%, observables16.3%, derivatives5.7%. These four stages account for94.0%. SR and moment computation together are only1.2%; eliminating that work could not close the1.37s gap to measuredMPI16.

The first experiments should therefore target sampling register/occupancy behavior and repeated Pfaffian/inverse refresh, with unchanged RNG controls and FP64 numerical checks. Observables/transformed-Slater work is the next measured target; profile inside that phase before choosing an implementation. Gauge and Slater refresh are already negligible.

The subsequent [granular profile](granular-profile.md) separates native
factorization/inversion, Pfaffians, observable GEMM, and layout publication.
It records the launch variants that did not justify adoption.

The20× budgets against the measured CPU1 baseline are3.856895s Opt and0.223270s Phys. Sampling plus initial warmup alone currently cost4.858185s Opt and0.469192s Phys. Optimizing inverse refresh alone cannot meet20×. For PhysCal, parsing and initial walker setup also cost0.230437s before sampling; reusable prepared-state workflows would need a separately labelled timing contract instead of silently removing those public-runner costs.

Every instrumented run had finite results and identical reported energy:99.78135142945789 total Opt energy and116.86444093776177 Phys, matching the primary GPU320 energies/site after division by128. Phase medians need not sum exactly to the full-wall median because they come from different repetitions.

Evidence: [raw log](phase-profile.log) and [exact diagnostic source](phase-profile-source.jl). The source snapshot uses the measured container's temporary input/checkpoint paths; recreate these from the main report before rerunning, or replace only those two path constants. Raw log also remains in the container's `/home/vscode/.cache/mvmc/benchmark-results/` directory. Representative source hashes and checkpoint SHA are emitted at its beginning. The log's automatic overhead summary includes the first public-wrapper compilation; the fully warmed comparison above explicitly excludes that invocation.
