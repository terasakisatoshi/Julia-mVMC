using Test, CUDA, MVMCOptimizers, LinearAlgebra
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
@testset "Native sampling boundary publishes independent inverse and Pfaffians" begin
    paths=(joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def"),
           joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def"))
    for path in paths
        data,rng=GPU.prepared_gpu_data(path;seed=79)
        cfg=GPU.initial_configurations(data,4,rng)
        s=gpu_state(data;configurations=cfg)
        expected=Array(s.overlaps)
        cache=GPU.GPUSamplingCache(s)
        work=GPU.BatchedSamplingInverse(s)
        GPU.refresh_sampling_state!(s,cache,work)
        @test Array(s.overlaps)≈expected atol=1e-9 rtol=1e-9
        @test all(Array(cache.valid))
        inverse=Array(cache.inverse)
        slater=Array(s.slater)
        n=2s.nelec
        for b=1:4,q=1:s.nqp
            sites=[cfg[i,b]+(i>s.nelec ? s.nsite : 0)+1 for i=1:n]
            matrix=-slater[sites,sites,q]
            @test norm(matrix*inverse[:,:,q,b]-I,Inf)<1e-9
            @test norm(inverse[:,:,q,b]+transpose(inverse[:,:,q,b]),Inf)<1e-12
        end
        @test Array(cache.accepted_since_refactor)==zeros(Int32,4)
        # Refresh is deterministic and resets accumulated acceptance counts.
        fill!(cache.accepted_since_refactor,5)
        GPU.refresh_sampling_state!(s,cache,work)
        @test Array(cache.accepted_since_refactor)==zeros(Int32,4)
    end
end

@testset "Startup Pfaffians reuse preserves native boundary results and RNG" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    results=[];future=[]
    for reuse in (false,true)
        s,pf=GPU.initial_gpu_state(data,8;seed=79,return_pfaffians=true)
        cache=GPU.GPUSamplingCache(s)
        work=GPU.BatchedSamplingInverse(s)
        GPU.refresh_sampling_state!(s,cache,work;initial_pfaffians=reuse ? pf : nothing)
        push!(results,(;cfg=Array(s.configurations),pf=Array(cache.pfaffians),
            inverse=Array(cache.inverse),valid=Array(cache.valid),overlap=Array(s.overlaps)))
        push!(future,Array(CUDA.rand(UInt32,64)))
    end
    a,b=results
    @test a.cfg==b.cfg
    @test a.pf≈b.pf atol=1e-9 rtol=1e-9
    @test a.inverse≈b.inverse atol=1e-9 rtol=1e-9
    @test a.valid==b.valid
    @test all(isapprox.(a.overlap,b.overlap;atol=1e-9,rtol=1e-9))
    @test future[1]==future[2]
end
