using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers, MVMCExpertModeParsers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
const P = MVMCExpertModeParsers
const GPU = Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")

function fallback_observables!(s,o)
    batch=size(s.configurations,2)
    @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false GPU.observable_kernel!(
        o.values,o.ops,o.coeffs,o.one,o.two,o.ex1,o.ex2,
        s.configurations,s.counts,s.occupancy,s.overlaps,s.logcorrelations,
        s.gutz,s.jastrow,s.parameters,s.slater,s.weights,GPU.factor_view(s.factors),
        GPU.pivot_view(s.pivots),s.pfaffians,GPU.cache_view(o.inverse),
        o.pfaffians,o.valid,GPU.cache_view(o.transformed),
        s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch,true)
end

@testset "Independent operator threads preserve all coincidences and fallback" begin
    path=joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,8,rng)
    ns=data.modpara.nsite
    data.green_one_terms=[P.GreenOneTerm(i,j,s,s) for i in 0:ns-1 for j in 0:ns-1 for s in (:up,:down)]
    data.green_two_terms=[P.GreenTwoTerm(i,j,k,l,s,s,t,t) for i in 0:ns-1 for j in 0:ns-1 for k in 0:ns-1 for l in 0:ns-1 for s in (:up,:down) for t in (:up,:down)]
    data.green_two_ex_terms=[P.GreenTwoExTerm(0,0,1,0,2,1,3,1),P.GreenTwoExTerm(1,0,0,0,3,1,2,1)]
    s=gpu_state(data;configurations=cfg)
    o=gpu_observables_workspace(s,data)
    gpu_observables!(s,o)
    # Small production states retain the scalar path and allocate no parallel
    # term scratch. This focused test deliberately exercises the new kernel.
    count=length(o.ops)+length(o.one)+length(o.two)+length(o.ex1)
    fields=map(fieldnames(typeof(o))) do name
        name == :termvalues ? CUDA.zeros(Float64,count,size(cfg,2)) : getfield(o,name)
    end
    o=GPU.CUDAObservables(fields...)
    original_valid=Array(o.valid)
    expected=hcat([cpu_observables(data,cpu_state(data),Int.(cfg[:,b]))[1] for b in 1:8]...)
    for force_fallback in (false,true)
        valid=copy(original_valid)
        force_fallback && (valid[1,1]=false)
        copyto!(o.valid,valid)
        fill!(o.values,NaN)
        GPU.parallel_observables!(s,o)
        fallback_observables!(s,o)
        @test Array(o.values) ≈ expected atol=1e-9 rtol=1e-9
        @test Array(s.configurations) == cfg
    end
end
