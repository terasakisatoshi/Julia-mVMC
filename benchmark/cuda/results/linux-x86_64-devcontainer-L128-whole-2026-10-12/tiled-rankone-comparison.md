# Tiled publication and rank-one optimization comparison

Warm whole public-runner medians in the same Linux x86_64 Dev Container,
Julia 1.13.1, one Julia/OpenBLAS thread, NVIDIA RTX 3060 device 1. Each
measurement uses seed 79, 320 total configurations, 20 optimization steps,
one PhysCal run, one warmup sweep, one interval sweep and three timed repeats.
Cold compilation/execution is excluded from these medians.

| Backend/source | Optimization (s) | PhysCal (s) |
| --- | ---: | ---: |
| CPU, one genuine MPI rank, frozen baseline | 77.137908551 | 4.465393414 |
| MPI4 independent groups, frozen baseline | 22.501600746 | 1.382951334 |
| MPI8 independent groups, frozen baseline | 14.729498005 | 0.984994268 |
| MPI16 independent groups, historical baseline | 11.174382908 | 0.762519904 |
| GPU1, historical baseline | 12.548162391 | 0.933714220 |
| GPU1, `020f95d` tiled publication and static rank-one Opt | 11.003444561 | 0.908276458 |
| GPU1, `eb6ab29` plus lazy PhysCal basis and initialization cap64 | 11.005270854 | 0.869055315 |
| GPU1, `760d811` plus triangle sampling and compact Phys transform | 10.503924950 | 0.764970380 |
| GPU1, `49c9bc7` plus parallel-QP initialization, original 3 repeats | 10.488677221 | 0.763737508 |
| GPU1, `49c9bc7`, separate 5-repeat confirmation | 10.493063189 | 0.762302185 |
| MPI16 independent groups, fresh 5-repeat confirmation | 11.211987096 | 0.780964168 |

At `020f95d`, optimization is 1.55% faster than the frozen MPI16 median;
PhysCal remains 19.12% slower. At `eb6ab29`, the PhysCal gap falls to 13.97%.
The goal of beating MPI16 in both workloads is only partly
achieved. The same existing sampling/output workload differences remain;
see the parent whole-run report for proposal budgets and timing scope.
Historical logs and medians were preserved.

Optimization repeats: 11.025083843, 10.991350072, 11.003444561 seconds.
PhysCal repeats: 0.900176203, 0.908276458, 0.910750031 seconds.
All repeats returned identical reported energies and last-interval accepted
counts for their respective workloads.

The unchanged public `benchmark_end_to_end.jl` driver SHA-256 is
`7c3ad955163cf6b2d31aa6265d8aafa0cffd11ea8f179be69a8dd07c8a71d65b`.
The fixed-parameter checkpoint remains
`36cef06347facd5645882d229405bc81095d31731454d821df5fc1bcad014db8`;
the cloned ModPara remains
`41dde0457bfe9e533a98c4017796fc56864202429e8d215c25b1907624bd6fbc`.
Extension files were staged using `git archive 020f95d` into an independent
writable container copy; no numerical source changed during timing.

Evidence: [whole public-runner log](whole-gpu-total320-020f95d.log),
[all-extension source inventory](whole-gpu-total320-020f95d-source-sha256.txt),
[runtime receipt](whole-gpu-total320-020f95d-runtime.txt).

Focused validation passed 42 transport checks, 20 production static-rank-one
checks including independently computed CPU energies and the strongly
cancelling configuration, and seven copied-runner repeatability checks over
two 20-step optimization runs. The latter included exact configurations,
projection counts and future UInt32 RNG words; computed energies and
parameters retained explicit numerical tolerances. The complete production
GPU suite passed 7,124 assertions in 47 testsets against `020f95d`, including
the public 20-step optimization/PhysCal repeatability tests; see the
[full suite log](tiled-rankone-full-gpu-suite.log).

The subsequent `eb6ab29` measurement uses the same driver, checkpoint, input,
threads, affinity and repetition budget. Its PhysCal median improves by
39.22 ms over `020f95d`, but remains 13.97% slower than MPI16. Optimization
still beats MPI16 narrowly. Its warm optimization repeats are 10.969458987,
11.043901621 and 11.005270854 seconds; PhysCal repeats are 0.875638090,
0.863599731 and 0.869055315 seconds. Reported energies and accepted counts
match across repeats and the previous GPU candidate. This revision's complete
GPU suite passed 7,150 assertions in 49 testsets; see the
[full suite log](eb6ab29-full-gpu-suite.log).
Evidence: [public-runner log](whole-gpu-total320-eb6ab29.log),
[source inventory](whole-gpu-total320-eb6ab29-source-sha256.txt),
[runtime receipt](whole-gpu-total320-eb6ab29-runtime.txt).

The `760d811` public-runner measurement uses the same unchanged driver and
canonical inputs. Its PhysCal median remains 2.450476 ms (0.3214%) slower than
MPI16; the goal of beating MPI16 in both workloads remains unmet.
Optimization repeats are 10.468586644, 10.503924950 and 10.516912672 seconds;
PhysCal repeats are 0.764160519, 0.764970380 and 0.767874493 seconds.
Reported energies and accepted counts are unchanged across repetitions and
from the earlier GPU revisions. The compact implementation passed 126 focused
assertions covering independent CPU Greens, strongly cancelling configurations,
mixed singular-QP fallback, transported packed inverse entries, and two public
20-save PhysCal runs with exact configurations/counts/future UInt32 words.
The complete `760d811` production GPU suite passed 8,241 assertions across
56 testsets in the Dev Container; see the
[full suite log](760d811-full-container-gpu-suite.log) and
[source inventory](full-760d811-source-sha256.txt).
Evidence: [public-runner log](whole-gpu-total320-760d811.log),
[source inventory](whole-gpu-total320-760d811-source-sha256.txt),
[runtime receipt](whole-gpu-total320-760d811-runtime.txt),
[focused gate](compact-production-cpu-repeat-gate.log).

## Parallel-QP initialization and confirmation runs

The original three-repeat public `49c9bc7` run returned optimization times
10.459272943, 10.488677221 and 10.507864427 seconds; PhysCal times were
0.755391768, 0.763737508 and 0.774663017 seconds. Its PhysCal median remained
1.217604 ms slower than the historical MPI16 median. All three results were
retained, rather than selecting the fastest PhysCal repetition.

Because the PhysCal difference was small, a separate five-repeat GPU run and
a fresh five-repeat MPI16 run followed, sequentially without competing builds
or GPU correctness jobs. Drivers, workload, seed, checkpoint, thread counts,
affinity and CPU numerical source remained unchanged; only repetition counts
and output paths changed.

| Confirmation | All optimization repetitions (s) | All PhysCal repetitions (s) |
| --- | --- | --- |
| GPU1 `49c9bc7` | 10.453616551, 10.471221754, 10.522433694, 10.505393634, 10.493063189 | 0.760588361, 0.761598662, 0.764200743, 0.780407637, 0.762302185 |
| MPI16 | 11.497685819, 11.252425522, 11.156991061, 11.211987096, 11.051598019 | 0.770273582, 0.766651869, 0.791917308, 0.787567029, 0.780964168 |

GPU medians are 6.41% faster for optimization and 2.39% faster for PhysCal
than these fresh MPI16 medians. PhysCal ranges overlap: GPU 0.760588361–
0.780407637 s and MPI16 0.766651869–0.791917308 s. The GPU PhysCal confirmation
median differs from the historical MPI16 median by only 0.217719 ms, so a
robust advantage over the earlier baseline has not been established. Both
the original and confirmation results are reported; no repetitions were
discarded. Returned energies and last-interval accepted counts remained
identical across the GPU repetitions.

Evidence: [original GPU 3-repeat log](whole-gpu-total320-49c9bc7.log),
[GPU 5-repeat log](whole-gpu-total320-49c9bc7-confirm5.log),
[GPU source inventory](whole-gpu-total320-49c9bc7-source-sha256.txt),
[GPU runtime receipt](whole-gpu-total320-49c9bc7-runtime.txt),
[fresh MPI16 5-repeat log](mpi-groups-16-total320-confirm5.log),
[MPI source inventory](mpi-groups-16-total320-confirm5-source-sha256.txt),
[MPI input/checkpoint inventory](mpi-groups-16-total320-confirm5-input-sha256.txt).
The final combined source's full GPU suite remains to be run.
