using CUDA,Test,MVMCOptimizers,Random
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","1")))
CUDA.allowscalar(false)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
# These kernels only transport existing payloads. Byte identity checks copies,
# not independently computed floating-point results.
bits(x)=reinterpret(UInt64,vec(Array(x)))
Random.seed!(734)
@testset "Tiled publication preserves transport bytes and partial tiles" begin
for (n,ns,batch,nq) in [(6,6,3,2),(64,64,65,3),(80,80,17,2),(128,128,33,2),(130,130,17,3),(64,32,5,2)]
    count=batch*nq
    source=CUDA.randn(Float64,n,n,nq,batch)
    source_pf=CUDA.randn(Float64,nq,batch)
    flags=CuArray(rand(Bool,nq,batch))
    a=CUDA.zeros(Float64,batch,n,n,nq); b=similar(a)
    pa=CUDA.zeros(Float64,n,n,count);pb=similar(pa)
    fa=CUDA.zeros(Float64,nq,batch);fb=similar(fa)
    va=CUDA.zeros(Bool,nq,batch);vb=similar(va)
    @cuda threads=256 blocks=cld(n*n*count,256) GPU.publish_sampling_inverse_kernel!(GPU.cache_view(a),pa,fa,va,source,source_pf,flags,n,batch,count)
    @cuda threads=(32,8) blocks=(cld(n*n,32),cld(batch,32),nq) GPU.tiled_publish_sampling_inverse_kernel!(GPU.cache_view(b),pb,fb,vb,source,source_pf,flags,n,batch,count)
    @test bits(a)==bits(b)
    @test bits(pa)==bits(pb)
    @test bits(fa)==bits(fb)
    @test Array(va)==Array(vb)==Array(flags)
    expected=permutedims(Array(source),(4,1,2,3))
    for q in 1:nq,w in 1:batch
        Array(flags)[q,w] || (expected[w,:,:,q].=0)
    end
    @test bits(a)==reinterpret(UInt64,vec(expected))
    map=zeros(Int32,2ns,batch)
    for w in 1:batch
        sites=randperm(2ns)
        for i in 1:n; map[sites[i],w]=-i; end
        for (j,r) in enumerate(sites[n+1:end]);map[r,w]=j;end
    end
    packed=CUDA.randn(Float64,n,2ns-n,count)
    mapping=CuArray(map)
    ta=CUDA.zeros(Float64,batch,n,2ns,nq);tb=similar(ta)
    @cuda threads=256 blocks=cld(n*2ns*count,256) GPU.publish_batched_transform!(GPU.cache_view(ta),packed,mapping,n,ns,batch,count)
    @cuda threads=(32,8) blocks=(cld(n*2ns,32),cld(batch,32),nq) GPU.tiled_publish_batched_transform!(GPU.cache_view(tb),packed,mapping,n,ns,batch,count)
    @test bits(ta)==bits(tb)
    hostpacked=Array(packed)
    expectedt=zeros(Float64,batch,n,2ns,nq)
    for q in 1:nq,w in 1:batch,r in 1:2ns,i in 1:n
        col=map[r,w]
        expectedt[w,i,r,q]=col>0 ? hostpacked[i,col,(q-1)*batch+w] : (i == -col ? -1.0 : 0.0)
    end
    @test bits(ta)==reinterpret(UInt64,vec(expectedt))
end
end
