using Test, CUDA, MVMCOptimizers, LinearAlgebra
const CPU=MVMCOptimizers
const P=MVMCOptimizers.MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
include("cpu_oracle.jl")
include("cancellation_configuration.jl")
@testset "Static Hamiltonian gate preserves general and PhysCal transforms" begin
    simple=[GPU.op1(1,2,0),GPU.op2(1,1,0,2,2,1)]
    @test !GPU.requires_transformed_slater(simple;measure_greens=false)
    @test GPU.requires_transformed_slater(simple;measure_greens=true)
    for op in (GPU.op2(1,2,0,1,2,1), # Pair hopping.
               GPU.op2(1,2,0,2,1,1), # Exchange.
               GPU.op2(1,2,0,3,4,0)) # General two moving pairs.
        @test GPU.requires_transformed_slater([op];measure_greens=false)
    end
end
function check_rankone(data,s,cache)
    cfg=Array(s.configurations)
    direct=GPU.CUDAObservables(s,data;measure_greens=false)
    full=GPU.CUDAObservables(s,data;measure_greens=true,compact_transform=false)
    @test direct.warp_hamiltonian
    @test !full.warp_hamiltonian
    @test isempty(direct.transformed)
    @test isempty(direct.batched_inverse.slater_rows)
    @test isempty(direct.batched_inverse.transformed)
    @test !isempty(full.transformed)
    @test !isempty(full.batched_inverse.slater_rows)
    @test !isempty(full.batched_inverse.transformed)
    for source in (nothing,cache)
        gpu_observables!(s,direct;sampling_cache=source)
        c=cpu_state(data)
        expected=[cpu_observables(data,c,cfg[:,b])[1][1] for b in axes(cfg,2)]
        @test all(isapprox.(vec(Array(direct.values)),expected;atol=1e-9,rtol=1e-9))
    end
    gpu_observables!(s,full;sampling_cache=cache)
    # The ordered row dot and cuBLAS transform have different accumulation paths.
    # Computed quantities retain the independent observable comparison policy.
    @test all(isapprox.(vec(Array(direct.values)),Array(full.values)[1,:];atol=1e-9,rtol=1e-9))
    c=cpu_state(data)
    expected=[cpu_observables(data,c,cfg[:,b])[1][1] for b in axes(cfg,2)]
    @test all(isapprox.(vec(Array(direct.values)),expected;atol=1e-9,rtol=1e-9))
    @test Array(s.configurations)==cfg
end
@testset "L128 rank-one energies match CPU after twenty cached proposals" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    s=GPU.initial_gpu_state(data,8;seed=123)
    cache=GPU.GPUSamplingCache(s)
    GPU.refresh_sampling_state!(s,cache,GPU.BatchedSamplingInverse(s))
    gpu_sample!(s;steps=20,cache,empty_destinations=true,incremental_correlation=true)
    check_rankone(data,s,cache)
    strong=gpu_state(data;configurations=reshape(Int32.(CANCELLATION_CFG),:,1))
    strongcache=GPU.GPUSamplingCache(strong)
    GPU.refresh_sampling_state!(strong,strongcache,GPU.BatchedSamplingInverse(strong))
    check_rankone(data,strong,strongcache)
    gpu_sample!(strong;steps=20,cache=strongcache,empty_destinations=true,incremental_correlation=true)
    check_rankone(data,strong,strongcache)
end
