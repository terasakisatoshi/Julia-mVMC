using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers, PfaPack
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)

@testset "Batched LU inverse residuals, singular status and transform" begin
    rng = MersenneTwister(81)
    for n in (64,128), holes in (0,3)
        ne = n÷2
        ns = ne+holes
        full = 2ns
        batch,nq = 3,2
        count = batch*nq
        Q = Matrix(qr(randn(rng,full,full)).Q)
        J = zeros(Float64,full,full)
        for i in 1:2:full
            J[i,i+1] = 1
            J[i+1,i] = -1
        end
        raw = Q*J*transpose(Q)
        A = (raw-transpose(raw))/2
        # Orthogonal skew matrices have condition number approximately one.
        # QP2 is exactly singular; no inverse result is asserted for that plane.
        slater = cat(-A,zeros(Float64,full,full);dims=3)
        cfg = hcat([vcat(randperm(rng,ns)[1:ne].-1,randperm(rng,ns)[1:ne].-1) for _ in 1:batch]...)
        device_cfg = CuArray(Int32.(cfg))
        device_slater = CuArray(slater)
        lu = CUDA.zeros(Float64,n,n,count)
        inverse = similar(lu)
        work = GPU.BatchedInverse(lu,similar(lu),inverse,
            CuArray([pointer(lu,(m-1)*n*n+1) for m in 1:count]),
            CuArray([pointer(inverse,(m-1)*n*n+1) for m in 1:count]),
            CUDA.zeros(Int32,n,count),CUDA.zeros(Int32,n,count),CUDA.zeros(Float64,count),
            CUDA.zeros(Int32,count),CUDA.zeros(Int32,count),
            CUDA.zeros(Float64,n,2ns-n,count),CUDA.zeros(Float64,n,2ns-n,count),
            CuArray([1.0]),CuArray([0.0]),CUDA.zeros(Int32,2ns-n,batch),CUDA.zeros(Int32,2ns,batch))
        entries = n*n*count
        CUDA.@cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.fill_batched_slater!(
            work.lu,work.factors,device_cfg,device_slater,ns,ne,batch,count)
        packed = Array(work.lu)
        CUDA.@cuda threads=128 blocks=count fastmath=false GPU.cooperative_pfaffian_kernel!(
            work.pfaffians,work.factors,work.pf_pivots,n)
        CUDA.cuBLAS.cublasDgetrfBatched(CUDA.cuBLAS.handle(),n,work.lu_ptrs,n,
            work.pivots,work.lu_info,count)
        CUDA.cuBLAS.cublasDgetriBatched(CUDA.cuBLAS.handle(),n,work.lu_ptrs,n,
            work.pivots,work.inverse_ptrs,n,work.inverse_info,count)
        cache = CUDA.zeros(Float64,batch,n,n,nq)
        pf = CUDA.zeros(Float64,nq,batch)
        valid = CUDA.zeros(Bool,nq,batch)
        CUDA.@cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.publish_batched_inverse!(
            GPU.cache_view(cache),pf,valid,work.inverse,work.pfaffians,
            work.lu_info,work.inverse_info,n,batch,count)
        output = Array(cache)
        lu_info, inverse_info = Array(work.lu_info),Array(work.inverse_info)
        validity, pfaffians = Array(valid),Array(pf)
        @test all(iszero,lu_info[1:batch])
        @test all(iszero,inverse_info[1:batch])
        @test all(>(0),lu_info[batch+1:end])
        @test all(>(0),inverse_info[batch+1:end])
        @test all(validity[1,:])
        @test !any(validity[2,:])
        transformed = CUDA.zeros(Float64,batch,n,2ns,nq)
        state = (;nelec=ne,nsite=ns,nqp=nq,configurations=device_cfg,slater=device_slater)
        GPU.batched_transform_cache!(state,(;transformed),work)
        transform = Array(transformed)
        for b in 1:batch
            selected = cfg[:,b] .+ vcat(zeros(Int,ne),fill(ns,ne)) .+ 1
            expectedA = A[selected,selected]
            expectedInverse = inv(expectedA)
            @test packed[:,:,b] ≈ expectedA atol=0 rtol=0 # gather, no arithmetic
            @test output[b,:,:,1] ≈ expectedInverse atol=1e-11 rtol=1e-11
            # LU is backward stable here; an O(n eps) bound includes pivot/update
            # rounding. The independent forward check above uses cond2≈1.
            residual = opnorm(expectedA*output[b,:,:,1]-I,Inf)
            backward = residual/(opnorm(expectedA,Inf)*opnorm(output[b,:,:,1],Inf))
            @test backward <= 64n*eps(Float64)
            inversions = sum(selected[i]>selected[j] for i in 1:n for j in i+1:n)
            expectedPf = holes == 0 ? (isodd(inversions) ? -1 : 1)*det(Q) : PfaPack.pfaffian_ltl!(copy(expectedA))
            @test pfaffians[1,b] ≈ expectedPf atol=1e-11 rtol=1e-11
            @test transform[b,:,:,1] ≈ expectedInverse*slater[selected,:,1] atol=1e-11 rtol=1e-11
            # Occupied columns use the exact A^-1*S_occ=-I identity; the
            # remaining columns retain the independently checked GEMM above.
            @test transform[b,:,selected,1] ≈ -Matrix{Float64}(I,n,n) atol=0 rtol=0
            @test size(work.slater_rows,2) == 2ns-n
            @test length(work.target_sites) == (2ns-n)*batch
        end
    end
end
