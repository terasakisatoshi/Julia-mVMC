using CUDA, MVMCOptimizers, LinearAlgebra, Random, Test
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
Random.seed!(817)
BLAS.set_num_threads(1)
@testset "Cooperative large-matrix inverse cache" begin
for n in (64,128), batch in (1,8), nq in (1,3)
    ne=n÷2; ns=n
    slater=zeros(2ns,2ns,nq)
    rows=vcat(1:ne,ns+1:ns+ne)
    matrices=[]
    for q in 1:nq
        r=randn(n,n); a=r-r'
        push!(matrices,a)
        slater[rows,rows,q] = -a
    end
    cfg=CuArray(repeat(Int32.(vcat(0:ne-1,0:ne-1)),1,batch))
    ss=CuArray(slater)
    cache=CUDA.zeros(Float64,batch,n,n,nq)
    pf=CUDA.zeros(Float64,nq,batch); valid=CUDA.zeros(Bool,nq,batch)
    a=CUDA.zeros(Float64,n,n,batch); ii=similar(a); piv=CUDA.zeros(Int32,n,batch)
    @cuda threads=256 blocks=batch shmem=8n GPU.cooperative_inverse_cache_kernel!(GPU.cache_view(cache),pf,valid,cfg,ss,a,ii,piv,ns,ne,nq,batch)
    CUDA.synchronize()
    got=Array(GPU.cache_view(cache))
    @test all(Array(valid))
    for q in 1:nq,b in 1:batch
        @test isapprox(got[:,:,q,b],inv(matrices[q]);atol=1e-9,rtol=1e-9)
        @test norm(matrices[q]*got[:,:,q,b]-I,Inf) < 1e-9
    end
    f=CuArray(repeat(reshape(matrices[1],n,n,1),1,1,batch)); pp=CUDA.zeros(Float64,batch)
    MVMCOptimizers.gpu_pfaffians!(pp,f,piv)
    @test isapprox(Array(pf)[1,:],Array(pp);atol=1e-9,rtol=1e-9)
end
end
