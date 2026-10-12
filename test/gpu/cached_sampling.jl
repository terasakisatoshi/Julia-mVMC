using Test,Random,LinearAlgebra,CUDA,MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)

function cache_test_state(ns,cfg,slater)
    ne,batch=size(cfg,1)÷2,size(cfg,2)
    nq=size(slater,3)
    # Synthetic dense geometry is deliberate: no Expert input preparation or
    # production-basis construction participates in this sampling unit gate.
    return GPU.CUDAState(ns,ne,1,nq,32,CuArray([0.1]),CUDA.zeros(Float64,0,1),
        CuArray(slater),CuArray([1.0,0.4]),CuArray(cfg),
        CUDA.zeros(Int32,2ns,batch),CUDA.zeros(Int32,1,batch),
        CuArray(zeros(Int32,ns)),CuArray(fill(Int32(-1),ns,ns)),1,0,
        CUDA.zeros(Float64,batch,2ne,2ne),CUDA.zeros(Int32,batch,2ne),
        CUDA.zeros(Float64,batch),CUDA.zeros(Float64,batch),CUDA.zeros(Float64,batch),
        CuArray(zeros(Int32,ns)),CUDA.zeros(Int32,batch),
        CUDA.zeros(Float64,2ne,2ne,batch),nothing)
end

@testset "Resident hopping inverse cache preserves fixed control paths" begin
    for ns in (64,128)
        ne,batch=ns÷2,4
        rng=MersenneTwister(12801+ns)
        cfg=repeat(reshape(Int32.(0:ns-1),ns,1),1,batch)
        cfg[ne+1,:].=0
        slater=randn(rng,2ns,2ns,2)
        for q in 1:2
            slater[:,:,q]-=transpose(slater[:,:,q])
        end
        for fraction in (0.0,1.0,0.37)
            fresh=cache_test_state(ns,cfg,slater)
            cached=cache_test_state(ns,cfg,slater)
            gpu_wavefunctions!(fresh);gpu_wavefunctions!(cached)
            work=GPU.GPUSamplingCache(cached)
            tape=rand(rng,UInt32,5,batch,20)
            for step in 1:20
                packet=CuArray(tape[:,:,step:step])
                gpu_sample!(fresh;steps=1,exchange_fraction=fraction,tape=packet)
                gpu_sample!(cached;steps=1,exchange_fraction=fraction,tape=packet,cache=work)
                @test Array(cached.configurations)==Array(fresh.configurations)
                @test Array(cached.accepted)==Array(fresh.accepted)
                @test Array(cached.counts)==Array(fresh.counts)
                @test Array(cached.occupancy)==Array(fresh.occupancy)
                # Rank updates and fresh factorization are independent numeric
                # algorithms. The1e-9 gate also checks their cumulative drift.
                @test all(isapprox.(Array(cached.overlaps),Array(fresh.overlaps);atol=1e-9,rtol=1e-9))
                @test Array(cached.logcorrelations) ≈ Array(fresh.logcorrelations) atol=1e-13 rtol=1e-13
            end
            inv=Array(work.inverse)
            fixed=Array(cached.configurations)
            for b in 1:batch,q in 1:2
                matrix=[-slater[fixed[i,b]+(i>ne ? ns : 0)+1,
                    fixed[j,b]+(j>ne ? ns : 0)+1,q] for i in 1:2ne,j in 1:2ne]
                residual=norm(matrix*inv[:,:,q,b]-I,Inf)
                @test residual<1e-9
            end
        end
        # cuRAND consumption and repeatability remain exact on identical cached
        # paths, independently of fresh-vs-incremental numerical differences.
        states=[cache_test_state(ns,cfg,slater) for _ in 1:2]
        future=Vector{UInt32}[]
        for s in states
            gpu_wavefunctions!(s)
            cache=GPU.GPUSamplingCache(s)
            CUDA.seed!(178)
            gpu_sample!(s;steps=20,exchange_fraction=0.37,cache)
            push!(future,Array(CUDA.rand(UInt32,64)))
        end
        @test Array(states[1].configurations)==Array(states[2].configurations)
        @test Array(states[1].accepted)==Array(states[2].accepted)
        @test future[1]==future[2]
        @test Array(states[1].overlaps) ≈ Array(states[2].overlaps) atol=1e-13 rtol=1e-13
    end
end

@testset "QP groups retain ordered sampling and inactive-group fallbacks" begin
    ns=64;batch=3
    rng=MersenneTwister(821)
    cfg=repeat(reshape(Int32.(0:ns-1),ns,1),1,batch)
    slater=randn(rng,2ns,2ns,2)
    for q in 1:2
        slater[:,:,q]-=transpose(slater[:,:,q])
    end
    for fraction in (0.0,0.37)
        tape=CuArray(rand(rng,UInt32,5,batch,20))
        reference=nothing
        future=nothing
        for (groups,cap) in ((1,255),(1,64),(2,64),(4,64),(8,64))
            s=cache_test_state(ns,cfg,slater)
            gpu_wavefunctions!(s)
            cache=GPU.GPUSamplingCache(s)
            CUDA.seed!(982)
            GPU.sample_with_cache!(s,cache,tape,fraction;qp_groups=groups,kernel_maxregs=cap)
            result=map(Array,(s.configurations,s.counts,s.occupancy,s.accepted,
                s.overlaps,s.logcorrelations,cache.inverse,cache.pfaffians,cache.valid))
            nextwords=Array(CUDA.rand(UInt32,64))
            if reference===nothing
                reference=result;future=nextwords
            else
                for k in (1,2,3,4,9)
                    @test result[k]==reference[k]
                end
                for k in (5,6,7,8)
                    @test all(isapprox.(result[k],reference[k];atol=1e-9,rtol=1e-9))
                end
                @test nextwords==future
            end
        end
    end
end
