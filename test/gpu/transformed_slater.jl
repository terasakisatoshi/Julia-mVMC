using CUDA, MVMCOptimizers, Random, LinearAlgebra, Test
using PfaPack: utu2pfa
CUDA.device!(parse(Int, get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)

function transformed_ratio_kernel!(out,cfg,ops,slater,inv,transformed,pf,valid,weights,
                                   factors,pivots,pfs,emptytrans,ns,ne,batch)
    b=(blockIdx().x-1)*blockDim().x+threadIdx().x
    b > batch && return nothing
    op=ops[b]
    c=op[7] == 2 ? GPU.move_operator!(cfg,op[4],op[5],op[6],ne,b) : 0
    a=GPU.move_operator!(cfg,op[1],op[2],op[3],ne,b)
    @inbounds out[1,b]=GPU.updated_overlap!(slater,weights,cfg,factors,pivots,pfs,
        inv,pf,valid,transformed,a,c,op,ns,ne,1,batch,b)
    @inbounds out[2,b]=GPU.updated_overlap!(slater,weights,cfg,factors,pivots,pfs,
        inv,pf,valid,emptytrans,a,c,op,ns,ne,1,batch,b)
    return nothing
end

function apply_valid(base,op,ne)
    cfg=copy(base)
    pairs=op[7] == 2 ? ((op[4],op[5],op[6]),(op[1],op[2],op[3])) : ((op[1],op[2],op[3]),)
    for (i,j,spin) in pairs
        range=spin*ne+1:(spin+1)*ne
        electron=findfirst(k->cfg[k] == j,range)
        electron === nothing && return nothing
        i != j && any(k->cfg[k] == i,range) && return nothing
        cfg[range[electron]]=i
    end
    return cfg
end

function reference_pf(a)
    f=copy(a); n=size(a,1); piv=zeros(Int32,n)
    MVMCOptimizers._ordinary_dsktf2_c_order!(f,piv)
    return utu2pfa(n,f,n,piv)
end

@testset "Resident transformed Slater columns and move corrections" begin
    rng=MersenneTwister(193)
    ns=64; ne=32; n=2ne
    base=Int32.(vcat(0:2:ns-1,1:2:ns-1))
    ops=GPU.GPUOperator[]; moved=Vector{Int32}[]
    for i in 0:5,j in 0:5,s in 0:1
        candidates=[GPU.op1(i,j,s)]
        for k in 0:5,l in 0:5,t in 0:1
            push!(candidates,GPU.op2(i,j,s,k,l,t))
        end
        for op in candidates
            cfg=apply_valid(base,op,ne)
            cfg === nothing && continue
            push!(ops,op);push!(moved,cfg)
        end
    end
    batch=length(ops)
    raw=randn(rng,2ns,2ns);slater=raw-raw'
    rows=Int.(base).+vcat(ones(Int,ne),fill(ns+1,ne))
    a=-slater[rows,rows]
    inverse=inv(a)
    cache=GPU.cache_view(CuArray(repeat(reshape(inverse,1,n,n,1),batch,1,1,1)))
    # Reshape above must preserve the mathematical row/column indexing.
    cfg=CuArray(repeat(base,1,batch)); valid=CUDA.ones(Bool,1,batch)
    transformed=GPU.cache_view(CUDA.zeros(Float64,batch,n,2ns,1))
    ss=CuArray(reshape(slater,2ns,2ns,1))
    entries=batch*n*2ns
    @cuda threads=256 blocks=cld(entries,256) GPU.transformed_slater_kernel!(
        transformed,cache,valid,cfg,ss,ns,ne,1,batch)
    out=CUDA.zeros(Float64,2,batch)
    @cuda threads=128 blocks=cld(batch,128) transformed_ratio_kernel!(out,cfg,CuArray(ops),ss,
        cache,transformed,CUDA.ones(Float64,1,batch),valid,CUDA.ones(Float64,1),
        CUDA.zeros(Float64,n,n,batch),CUDA.zeros(Int32,n,batch),CUDA.zeros(Float64,batch),
        GPU.cache_view(CUDA.zeros(Float64,0,0,0,0)),ns,ne,batch)
    actual=Array(out)
    expected=[reference_pf(-slater[Int.(idx).+vcat(ones(Int,ne),fill(ns+1,ne)),
                                  Int.(idx).+vcat(ones(Int,ne),fill(ns+1,ne))])/reference_pf(a) for idx in moved]
    @test all(isapprox.(actual[1,:],actual[2,:];atol=1e-9,rtol=1e-9))
    @test all(isapprox.(actual[1,:],expected;atol=1e-9,rtol=1e-9))
    println("transformed Slater valid_operators=$batch max_old_error=",maximum(abs,actual[1,:]-actual[2,:]),
            " max_CPU_error=",maximum(abs,actual[1,:]-expected))
end
