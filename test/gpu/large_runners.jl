using Test, CUDA, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
const CPU=MVMCOptimizers
const P=MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
include("cpu_oracle.jl")
@testset "L128 complete optimization and PhysCal repeat on resident paths" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    runs=[];future=[]
    for _=1:2
        result=run_gpu_para_opt_from_namelist(path;nsteps=20,walkers=8,
            device=parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")),seed=79,warmup=128,interval=128)
        push!(runs,result);push!(future,Array(CUDA.rand(UInt32,64)))
    end
    a,b=runs
    @test a.status==b.status==0
    @test length(a.energies)==20
    @test all(isfinite,a.parameters) && all(isfinite,a.energies)
    @test a.energies≈b.energies atol=1e-9 rtol=1e-9
    @test a.parameters≈b.parameters atol=1e-9 rtol=1e-9
    @test Array(a.state.configurations)==Array(b.state.configurations)
    @test Array(a.state.counts)==Array(b.state.counts)
    @test future[1]==future[2]
    @test real.(CPU.pack_parameters(a.data))≈a.parameters atol=1e-13 rtol=1e-13
    o=GPU.CUDAObservables(a.state,a.data)
    gpu_observables!(a.state,o)
    c=cpu_state(a.data)
    cfg=Int.(Array(a.state.configurations))
    expected=hcat([cpu_observables(a.data,c,cfg[:,i])[1] for i=1:8]...)
    @test all(isapprox.(Array(o.values),expected;atol=1e-9,rtol=1e-9))
    phys=[];future=[]
    for _=1:2
        result=run_gpu_phys_cal_from_namelist(path;nsamples=2,walkers=8,
            device=parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")),seed=79,warmup=128,interval=128)
        push!(phys,result);push!(future,Array(CUDA.rand(UInt32,64)))
    end
    p,q=phys
    @test p.status==q.status==0
    @test p.energy≈q.energy atol=1e-9 rtol=1e-9
    @test p.green_one≈q.green_one atol=1e-9 rtol=1e-9
    @test p.green_two≈q.green_two atol=1e-9 rtol=1e-9
    @test Array(p.state.configurations)==Array(q.state.configurations)
    @test future[1]==future[2]
end
