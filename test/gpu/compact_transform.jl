using Test,CUDA,MVMCOptimizers,LinearAlgebra
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU=MVMCOptimizers
const P=MVMCOptimizers.MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
include("cancellation_configuration.jl")
@testset "Compact eligibility retains general and small matrix paths" begin
    large=(nelec=64,nsite=128)
    @test GPU.compact_transform_eligible(large,Int32[1,129])
    @test !GPU.compact_transform_eligible(large,Int32[])
    @test !GPU.compact_transform_eligible(large,Int32.(1:33))
    @test !GPU.compact_transform_eligible((nelec=8,nsite=16),Int32[1])
end
@testset "Compact Phys twenty saves repeat RNG and independent CPU Greens" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    results=[];future=[];lastvalues=[]
    for repeat in 1:2
        last=Ref{Any}(nothing)
        callback=(sample,state,o)->(sample==20 && (last[]=Array(o.values)))
        result=run_gpu_phys_cal_from_namelist(path;nsamples=20,walkers=8,
            device=parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")),seed=79,
            warmup=128,interval=128,callback)
        push!(results,result);push!(future,Array(CUDA.rand(UInt32,64)));push!(lastvalues,last[])
    end
    a,b=results
    @test a.status==b.status==0
    @test a.energy≈b.energy atol=1e-9 rtol=1e-9
    @test a.green_one≈b.green_one atol=1e-9 rtol=1e-9
    @test a.green_two≈b.green_two atol=1e-9 rtol=1e-9
    @test Array(a.state.configurations)==Array(b.state.configurations)
    @test Array(a.state.counts)==Array(b.state.counts)
    @test future[1]==future[2]
    c=cpu_state(a.data);cfg=Int.(Array(a.state.configurations))
    expected=hcat([cpu_observables(a.data,c,cfg[:,w])[1] for w in axes(cfg,2)]...)
    @test all(isapprox.(lastvalues[1],expected;atol=1e-9,rtol=1e-9))
end
function check_cache_observables(data,s,cache)
    o=GPU.CUDAObservables(s,data;compact_transform=false)
    sr=GPU.CUDASR(s,data)
    gpu_observables!(s,o;sampling_cache=cache)
    baseline=Array(o.values)
    o=GPU.CUDAObservables(s,data)
    compact=o.compact
    @test Array(compact.sites)==Int32[1,129]
    println("compact physical columns=",Array(compact.sites))
    gpu_observables!(s,o;sampling_cache=cache)
    @test isempty(o.transformed)
    @test isempty(o.batched_inverse.slater_rows)
    @test isempty(o.batched_inverse.transformed)
    # Publication transports these already computed entries without arithmetic.
    packed=reshape(Array(o.batched_inverse.inverse),2s.nelec,2s.nelec,size(s.configurations,2),s.nqp)
    resident=permutedims(Array(o.inverse),(2,3,1,4))
    @test reinterpret(UInt64,vec(packed))==reinterpret(UInt64,vec(resident))
    actualcompact=Array(o.values)
    differences=abs.(actualcompact-baseline)
    println("compact versus full GEMM maximum absolute difference=",maximum(differences))
    firstdifference=findfirst(!iszero,differences)
    println("first compact versus full difference index=",firstdifference,
        " values=",firstdifference===nothing ? nothing : (actualcompact[firstdifference],baseline[firstdifference]))
    @test all(isapprox.(actualcompact,baseline;atol=1e-9,rtol=1e-9))
    GPU.derivatives!(s,sr,o)
    cfg=Array(s.configurations)
    c=cpu_state(data)
    reference=[cpu_observables(data,c,cfg[:,b]) for b in axes(cfg,2)]
    expected=hcat(first.(reference)...)
    expectedO=hcat(last.(reference)...)
    actual=Array(o.values);actualO=Array(sr.derivatives)
    println("cache reuse maxerrors obs=",maximum(abs,actual-expected)," O=",maximum(abs,actualO-expectedO))
    @test all(isapprox.(actual,expected;atol=1e-9,rtol=1e-9))
    @test all(isapprox.(actualO,expectedO;atol=1e-9,rtol=1e-9))
    inverse=Array(cache.inverse);slater=Array(s.slater);valid=Array(cache.valid)
    worst=0.0
    for b in axes(cfg,2),q in 1:s.nqp
        valid[q,b] || continue
        sites=vcat(Int.(cfg[1:s.nelec,b]).+1,Int.(cfg[s.nelec+1:end,b]).+s.nsite.+1)
        A=-slater[sites,sites,q]
        G=inverse[:,:,q,b]
        residual=opnorm(A*G-I,Inf)/(opnorm(A,Inf)*opnorm(G,Inf))
        worst=max(worst,residual)
        # Twenty rank updates accumulate row-dot and outer-product rounding;
        # this conservative O(n*eps) backward bound is separate from the
        # unchanged independent CPU observable and derivative1e-9 gate.
        @test residual <= 128(2s.nelec)*eps(Float64)
    end
    println("cache inverse maximum normalized residual=",worst)
    @test Array(o.valid)==valid
    @test Array(o.pfaffians) ≈ Array(cache.pfaffians) atol=eps(Float64) rtol=eps(Float64)
    return nothing
end
@testset "L128 sampled inverse reuse independent CPU observables" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    s=GPU.initial_gpu_state(data,8;seed=123)
    cache=GPU.GPUSamplingCache(s)
    GPU.refresh_sampling_state!(s,cache,GPU.BatchedSamplingInverse(s))
    gpu_sample!(s;steps=20,cache,empty_destinations=true,incremental_correlation=true)
    check_cache_observables(data,s,cache)
    # The persistent strongly cancelling walker from the merged projection
    # regression is independent of any new initializer or sampling trajectory.
    strong=gpu_state(data;configurations=reshape(Int32.(CANCELLATION_CFG),:,1))
    strongcache=GPU.GPUSamplingCache(strong)
    GPU.refresh_sampling_state!(strong,strongcache,GPU.BatchedSamplingInverse(strong))
    check_cache_observables(data,strong,strongcache)
    gpu_sample!(strong;steps=20,cache=strongcache,empty_destinations=true,incremental_correlation=true)
    check_cache_observables(data,strong,strongcache)
end
@testset "Sampler inverse reuse retains singular QP cofactor fallback" begin
    ns,ne,nq,batch=64,32,2,2
    n=2ne
    cfg=hcat(vcat(collect(0:ne-1),collect(0:ne-1)),vcat(collect(ne:ns-1),collect(ne:ns-1)))
    A=zeros(Float64,2ns,2ns)
    for i in 1:2:2ns
        A[i,i+1]=1;A[i+1,i]=-1
    end
    slater=cat(-A,-A;dims=3)
    ri,rj=ns+ne-1,ns+ne
    slater[ri,rj,1]=slater[rj,ri,1]=0
    basis=zeros(Float64,4ns*ns*nq,2)
    for q in 1:nq
        basis[ri+(rj-1)*2ns+(q-1)*4ns*ns,2]=-1
        basis[rj+(ri-1)*2ns+(q-1)*4ns*ns,2]=1
    end
    s=GPU.CUDAState(ns,ne,1,nq,32,CuArray([0.0,1.0]),CuArray(basis),CuArray(slater),
        CuArray([0.25,0.75]),CuArray(Int32.(cfg)),CUDA.zeros(Int32,2ns,batch),
        CUDA.zeros(Int32,1,batch),CuArray(vcat(zeros(Int32,ne),fill(Int32(-1),ne))),CuArray(fill(Int32(-1),ns,ns)),
        1,0,CUDA.zeros(Float64,batch,n,n),CUDA.zeros(Int32,batch,n),CUDA.zeros(Float64,batch),
        CUDA.zeros(Float64,batch),CUDA.zeros(Float64,batch),CUDA.zeros(Int32,ns),
        CUDA.zeros(Int32,batch),CUDA.zeros(Float64,n,n,batch),nothing)
    compressed=CUDA.cuSPARSE.CuSparseMatrixCSC(s.basis)
    sr=GPU.CUDASR(CUDA.zeros(Float64,2,batch),similar(s.factors),CuArray([true,true]),
        CUDA.zeros(Float64,size(basis,1),batch),CuArray([1.0]),CuArray([0.0]),
        compressed,GPU.decoded_sparse_basis(compressed,ns))
    empty=CuArray(GPU.GPUOperator[])
    o=GPU.CUDAObservables(CUDA.zeros(Float64,3,batch),empty,CuArray(Float64[]),
        CuArray([GPU.op1(ne-1,ne-1,1),GPU.op1(ne,ne-1,1)]),empty,empty,empty,
        CUDA.zeros(Float64,batch,n,n,nq),CUDA.zeros(Float64,nq,batch),CUDA.zeros(Bool,nq,batch),
        CUDA.zeros(Float64,n,n,0),CUDA.zeros(Float64,n,n,0),
        CUDA.zeros(Float64,batch,n,2ns,nq),CUDA.zeros(Float64,2,batch),GPU.BatchedInverse(s),nothing)
    cache=GPU.GPUSamplingCache(s)
    GPU.refresh_sampling_state!(s,cache)
    gpu_observables!(s,o;sampling_cache=cache)
    original=Array(o.values)
    columns=zeros(Int32,2ns);columns[ns+ne+1]=1
    compact=GPU.CompactTransform(CuArray(Int32[ns+ne+1]),CuArray(columns),similar(s.configurations),
        CUDA.zeros(Float64,n,1,batch*nq),CUDA.zeros(Float64,n,1,batch*nq))
    GPU.compact_observables!(s,o,compact;sampling_cache=cache)
    @test all(isapprox.(Array(o.values),original;atol=1e-12,rtol=1e-12))
    GPU.derivatives!(s,sr,o)
    @test Array(o.valid)==Bool[false true;true true]
    @test Array(o.values) ≈ [0.0 0.0;1.0 0.0;0.0 0.0] atol=1e-12 rtol=1e-12
    @test Array(sr.derivatives) ≈ [32.0 0.0;4/3 0.0] atol=1e-12 rtol=1e-12
end
