using Test,Random,LinearAlgebra,CUDA,MVMCOptimizers,MVMCExpertModeParsers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU=MVMCOptimizers
const P=MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
# A fixed L128 configuration from the seed-79 production benchmark. It has a
# strongly cancelling projected overlap; it is an input, not a generated oracle.
include("cancellation_configuration.jl")

function projected_rounding_kernel!(overlaps,state)
    b=blockIdx().x
    ip=GPU.cooperative_overlap!(state,b)
    if threadIdx().x==1
        @inbounds overlaps[b]=ip
    end
    return nothing
end

@testset "Cooperative C-order arithmetic survives projected cancellation" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    c=cpu_state(data)
    x=cpu_overlap(data,c,CANCELLATION_CFG)
    expected=copy(c.slater_matrix.pf_m_real)
    ns,ne,nq=data.modpara.nsite,data.modpara.nelec,CPU.get_n_qp_full(data)
    slater=reshape(c.slater_matrix.slater_elm_real,2ns,2ns,nq)
    matrices=Array{Float64}(undef,2ne,2ne,nq)
    for q in 1:nq,i in 1:2ne,j in 1:2ne
        ri=CANCELLATION_CFG[i]+(i>ne ? ns : 0)+1
        rj=CANCELLATION_CFG[j]+(j>ne ? ns : 0)+1
        matrices[i,j,q]=-slater[ri,rj,q]
    end
    values=CUDA.zeros(Float64,nq)
    GPU.cooperative_pfaffians!(values,CuArray(matrices),CUDA.zeros(Int32,2ne,nq))
    actual=Array(values)
    weights=real.(data.qp_weights.qp_full_weight)
    @test all(isapprox.(actual,expected;atol=1e-10,rtol=1e-12))
    kappa=sum(abs,weights.*expected)/abs(x.ip)
    @test kappa>1e6
    # Keep the existing1e-9 portable tolerance on the ill-conditioned projected
    # value. A tiny relative PF error must not be silently amplified past it.
    projected=0.0
    for q in 1:nq
        projected+=weights[q]*actual[q]
    end
    @test projected ≈ x.ip atol=1e-9 rtol=1e-9
    # Exercise the resident wavefunction's weighted projection as well: it
    # must not reintroduce contraction after the precise Pfaffian updates.
    state=(;nelec=ne,nsite=ns,nqp=nq,
        configurations=CuArray(reshape(CANCELLATION_CFG,:,1)),slater=CuArray(slater),
        weights=CuArray(weights),cooperative_factors=CUDA.zeros(Float64,2ne,2ne,1),
        pivots=CUDA.zeros(Int32,2ne,1))
    overlap=CUDA.zeros(Float64,1)
    @cuda threads=128 blocks=1 fastmath=false projected_rounding_kernel!(overlap,state)
    @test only(Array(overlap)) ≈ x.ip atol=1e-9 rtol=1e-9
end
