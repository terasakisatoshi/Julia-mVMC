module InitialParallelQP
include(joinpath(pwd(),"test","gpu","initial_parallel_qp.jl"))
end
module BatchedInverse
include(joinpath(pwd(),"test","gpu","batched_inverse.jl"))
end
module SamplingInverse
include(joinpath(pwd(),"test","gpu","sampling_inverse.jl"))
end
module RankoneObservables
include(joinpath(pwd(),"test","gpu","rankone_observables.jl"))
end
