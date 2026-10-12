using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)

@testset "Warp CSC contraction obeys a floating-point reduction bound" begin
    rng=MersenneTwister(12832)
    counts=(0,1,31,32,33,127,1024,8192)
    nrows,np,batch=16384,length(counts),7
    basis=zeros(Float64,nrows,np)
    for p in 1:np
        basis[1:counts[p],p] .= randn(rng,counts[p])
    end
    sensitivity=randn(rng,nrows,batch)
    sparse=CUDA.cuSPARSE.CuSparseMatrixCSC(CuArray(basis))
    output=CUDA.zeros(Float64,np,batch)
    GPU.warp_sparse_contraction!(output,CuArray(sensitivity),sparse)
    actual=Array(output)
    @test actual ≈ transpose(basis)*sensitivity atol=1e-11 rtol=1e-11
    @test all(iszero,actual[1,:])
    # Each lane performs at most ceil(nnz/32) products/additions, followed by
    # five tree additions. This standard gamma bound uses two rounding steps
    # per local term and allows the final BigFloat-to-Float64 reference rounding.
    setprecision(BigFloat,256) do
        for p in 1:np, b in 1:batch
            reference=BigFloat(0)
            absolute_sum=BigFloat(0)
            for k in 1:counts[p]
                term=BigFloat(basis[k,p])*BigFloat(sensitivity[k,b])
                reference += term
                absolute_sum += abs(term)
            end
            depth=2cld(counts[p],32)+5
            gamma=depth*eps(Float64)/(1-depth*eps(Float64))
            bound=gamma*Float64(absolute_sum)+eps(Float64)*abs(Float64(reference))
            @test abs(actual[p,b]-Float64(reference)) <= bound
        end
    end
end
