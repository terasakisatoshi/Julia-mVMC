using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers, MVMCExpertModeParsers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
const P = MVMCExpertModeParsers
const GPU = Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
@testset "All ordinary one/two-body operator coincidences and signs" begin
    path = joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng = GPU.prepared_gpu_data(path;seed=79)
    cfg = GPU.initial_configurations(data,8,rng)
    ns = data.modpara.nsite
    data.green_one_terms = [P.GreenOneTerm(i,j,s,s) for i in 0:ns-1 for j in 0:ns-1 for s in (:up,:down)]
    data.green_two_terms = [P.GreenTwoTerm(i,j,k,l,s,s,t,t) for i in 0:ns-1 for j in 0:ns-1 for k in 0:ns-1 for l in 0:ns-1 for s in (:up,:down) for t in (:up,:down)]
    data.green_two_ex_terms = [P.GreenTwoExTerm(0,0,1,0,2,1,3,1),P.GreenTwoExTerm(1,0,0,0,3,1,2,1)]
    s = gpu_state(data;configurations=cfg)
    o = gpu_observables_workspace(s,data)
    gpu_observables!(s,o)
    expected = hcat([cpu_observables(data,cpu_state(data),Int.(cfg[:,b]))[1] for b in 1:8]...)
    @test Array(o.values) ≈ expected atol=1e-9 rtol=1e-9
    @test Array(s.configurations) == cfg
    @info "Exhaustive Green comparison" quantities=size(expected,1) max_error=maximum(abs,Array(o.values)-expected)
end
