using Test,Random,LinearAlgebra,CUDA,MVMCOptimizers,MVMCExpertModeParsers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU=MVMCOptimizers
const P=MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
include("cancellation_configuration.jl")

for delayed in (false,true)
@testset "Cached L128 projected sampler follows shared20-step controls" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,4,rng)
    cfg[:,1].=CANCELLATION_CFG
    fresh=gpu_state(data;configurations=cfg)
    cached=gpu_state(data;configurations=cfg)
    work=GPU.GPUSamplingCache(cached;delayed)
    GPU.refresh_sampling_state!(cached,work,GPU.BatchedSamplingInverse(cached))
    @test all(isapprox.(Array(cached.overlaps),Array(fresh.overlaps);atol=1e-9,rtol=1e-9))
    c=cpu_state(data)
    x=cpu_overlap(data,c,CANCELLATION_CFG)
    expected_pf=copy(c.slater_matrix.pf_m_real)
    @test sum(abs,real.(data.qp_weights.qp_full_weight).*expected_pf)/abs(x.ip)>1e6
    tape=rand(MersenneTwister(128791),UInt32,5,4,20)
    for step in 1:20
        packet=CuArray(tape[:,:,step:step])
        gpu_sample!(fresh;steps=1,tape=packet)
        gpu_sample!(cached;steps=1,tape=packet,cache=work,incremental_correlation=true,triangle_update=true)
        @test Array(cached.configurations)==Array(fresh.configurations)
        @test Array(cached.accepted)==Array(fresh.accepted)
        @test Array(cached.counts)==Array(fresh.counts)
        @test Array(cached.occupancy)==Array(fresh.occupancy)
        @test all(isapprox.(Array(cached.overlaps),Array(fresh.overlaps);atol=1e-9,rtol=1e-9))
        @test Array(cached.logcorrelations) ≈ Array(fresh.logcorrelations) atol=1e-13 rtol=1e-13
    end
    # A forced invalid sector exercises the independent full-Pf/rebuild path.
    flags=Array(work.valid);flags[1,1]=false;copyto!(work.valid,flags)
    packet_data=copy(tape[:,:,1:1])
    final_cfg=Array(fresh.configurations)
    destination=first(setdiff(0:127,final_cfg[1:64,1]))
    packet_data[:,1,1].=UInt32.([0,0,destination,typemax(UInt32),0])
    packet=CuArray(packet_data)
    gpu_sample!(fresh;steps=1,tape=packet)
    gpu_sample!(cached;steps=1,tape=packet,cache=work,incremental_correlation=true,triangle_update=true)
    @test Array(cached.configurations)==Array(fresh.configurations)
    @test all(isapprox.(Array(cached.overlaps),Array(fresh.overlaps);atol=1e-9,rtol=1e-9))
    @test all(Array(work.valid)[:,1])
end

@testset "Empty-rank hopping and integer count deltas match translated tapes" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,4,rng)
    cfg[:,1].=CANCELLATION_CFG
    fresh=gpu_state(data;configurations=cfg)
    cached=gpu_state(data;configurations=cfg)
    work=GPU.GPUSamplingCache(cached;delayed)
    GPU.refresh_sampling_state!(cached,work,GPU.BatchedSamplingInverse(cached))
    localspins=Array(fresh.localspins)
    tape=rand(MersenneTwister(128792),UInt32,5,4,20)
    for step in 1:20
        fixed=Array(fresh.configurations)
        original=tape[:,:,step:step]
        translated=copy(original)
        for b in 1:4
            spin=original[2,b,1]<UInt32(0x80000000) ? 0 : 1
            occupied=Set(fixed[spin*64+1:(spin+1)*64,b])
            # Independent host list selects the same physically valid target
            # without sharing the device selector's scan implementation.
            choices=[r for r in 0:127 if r ∉ occupied && localspins[r+1]==0]
            target=choices[Int(original[3,b,1]%UInt32(length(choices)))+1]
            translated[3,b,1]=UInt32(target)
        end
        gpu_sample!(fresh;steps=1,tape=CuArray(translated))
        gpu_sample!(cached;steps=1,tape=CuArray(original),cache=work,
            empty_destinations=true,incremental_correlation=true,triangle_update=true)
        @test Array(cached.configurations)==Array(fresh.configurations)
        @test Array(cached.accepted)==Array(fresh.accepted)
        @test Array(cached.counts)==Array(fresh.counts)
        @test Array(cached.occupancy)==Array(fresh.occupancy)
        @test all(isapprox.(Array(cached.overlaps),Array(fresh.overlaps);atol=1e-9,rtol=1e-9))
        @test all(isapprox.(Array(cached.logcorrelations),Array(fresh.logcorrelations);atol=1e-13,rtol=1e-13))
    end
end

end # update strategy
