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

@testset "Tiled triangular update preserves packet boundaries" begin
    ns=64;batch=3
    rng=MersenneTwister(88101)
    cfg=repeat(reshape(Int32.(0:ns-1),ns,1),1,batch)
    base=randn(rng,2ns,2ns,2)
    for q in 1:2
        base[:,:,q]-=transpose(base[:,:,q])
    end
    for scenario in (:normal,:invalid,:exchange,:threshold,:last_threshold,:zero_qp),
        empty in (false,true),incremental in (false,true)
        slater=copy(base)
        if scenario==:zero_qp
            # Initially unoccupied up-site32 becomes the proposed first row.
            # Its first QP Pfaffian is exactlyzero; the second remains finite.
            slater[33,:,1].=0.0;slater[:,33,1].=0.0
        end
        steps=scenario in (:last_threshold,:zero_qp) ? 1 : 20
        words=rand(rng,UInt32,5,batch,steps)
        words[1,:,1].=0;words[2,:,1].=0
        words[3,:,1].=empty ? 0 : 32
        words[5,:,1].=0
        fraction=scenario==:exchange ? 1.0 : 0.0
        fraction==1.0 && (words[3,:,1].=0)
        tape=CuArray(words)
        reference=nothing;future=nothing;second_reference=nothing
        for triangle in (false,true)
            s=cache_test_state(ns,cfg,slater)
            gpu_wavefunctions!(s)
            c=GPU.GPUSamplingCache(s)
            GPU.refresh_sampling_state!(s,c)
            if scenario==:invalid
                flags=Array(c.valid);flags[1,1]=false;copyto!(c.valid,flags)
            elseif scenario in (:threshold,:last_threshold)
                fill!(c.accepted_since_refactor,Int32(ns))
            end
            CUDA.seed!(9281)
            GPU.sample_with_cache!(s,c,tape,fraction;empty_destinations=empty,
                incremental_correlation=incremental,triangle_update=triangle)
            result=map(Array,(s.configurations,s.counts,s.occupancy,s.accepted,
                s.overlaps,s.logcorrelations,c.inverse,c.pfaffians,c.valid,
                c.accepted_since_refactor))
            nextwords=Array(CUDA.rand(UInt32,64))
            @test sum(result[4])>0
            if !triangle
                reference=result;future=nextwords
            else
                for k in (1,2,3,4,9,10)
                    @test result[k]==reference[k]
                end
                for k in (5,6,7,8)
                    @test all(isapprox.(result[k],reference[k];atol=1e-9,rtol=1e-9))
                end
                @test nextwords==future
                for q in 1:2,b in 1:batch
                    matrix=result[7][:,:,q,b]
                    @test maximum(abs.(matrix+transpose(matrix)))<1e-12
                    if result[9][q,b]
                        sites=Int.(result[1][:,b]).+1
                        sites[ns÷2+1:end].+=ns
                        selected=-slater[sites,sites,q]
                        @test norm(selected*matrix-I,Inf)<1e-9
                    end
                end
            end
            # Reusing the same cache must retain the ordered control path
            # after a preceding singular or threshold rebuild.
            GPU.sample_with_cache!(s,c,tape,fraction;empty_destinations=empty,
                incremental_correlation=incremental,triangle_update=triangle)
            second=map(Array,(s.configurations,s.counts,s.occupancy,s.accepted,
                s.overlaps,s.logcorrelations,c.inverse,c.pfaffians,c.valid,
                c.accepted_since_refactor))
            if !triangle
                second_reference=second
            else
                for k in (1,2,3,4,9,10)
                    @test second[k]==second_reference[k]
                end
                for k in (5,6,7,8)
                    @test all(isapprox.(second[k],second_reference[k];atol=1e-9,rtol=1e-9))
                end
            end
        end
    end
end
@testset "Triangular grouped fallback and inactive lanes" begin
    ns=64;batch=3;rng=MersenneTwister(17201)
    cfg=repeat(reshape(Int32.(0:ns-1),ns,1),1,batch)
    slater=randn(rng,2ns,2ns,2)
    for q in 1:2
        slater[:,:,q]-=transpose(slater[:,:,q])
    end
    for fraction in (0.0,0.37,1.0)
        tape=CuArray(rand(rng,UInt32,5,batch,20))
        reference=nothing;future=nothing
        for (triangle,groups) in ((false,1),(true,1),(true,2),(true,4),(true,8))
            s=cache_test_state(ns,cfg,slater);gpu_wavefunctions!(s)
            c=GPU.GPUSamplingCache(s);GPU.refresh_sampling_state!(s,c)
            flags=Array(c.valid);flags[2,1]=false;copyto!(c.valid,flags)
            counters=Array(c.accepted_since_refactor);counters[2]=ns;copyto!(c.accepted_since_refactor,counters)
            CUDA.seed!(7101)
            GPU.sample_with_cache!(s,c,tape,fraction;qp_groups=groups,triangle_update=triangle,
                empty_destinations=true,incremental_correlation=true)
            result=map(Array,(s.configurations,s.counts,s.occupancy,s.accepted,
                s.overlaps,s.logcorrelations,c.inverse,c.pfaffians,c.valid,c.accepted_since_refactor))
            nextwords=Array(CUDA.rand(UInt32,64))
            if !triangle
                reference=result;future=nextwords
            else
                for k in (1,2,3,4,9,10)
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

@testset "Runner selects triangular updates only for large resident caches" begin
    @test !GPU.runner_sampling_options(nothing).triangle_update
    for ns in (64,128)
        cfg=reshape(Int32.(0:ns-1),ns,1)
        s=cache_test_state(ns,cfg,zeros(Float64,2ns,2ns,2))
        c=GPU.GPUSamplingCache(s)
        @test GPU.runner_sampling_options(c).triangle_update==(ns>=128)
        @test_throws ArgumentError gpu_sample!(s;steps=0,triangle_update=true)
    end
end
