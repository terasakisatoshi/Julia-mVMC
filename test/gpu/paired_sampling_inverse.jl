using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers, PfaPack
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
include("cancellation_configuration.jl")

@testset "Paired factors preserve independently known Pfaffians and inverses" begin
    rng=MersenneTwister(548)
    for n in (2,6,32,64,128,130), capacity in 0:4
        batch=4
        matrices=zeros(n,n,batch)
        expected_pf=zeros(batch)
        for b in 1:batch
            Q=b==1 ? Matrix{Float64}(I,n,n) : Matrix(qr(randn(rng,n,n)).Q)
            D=zeros(n,n)
            # Cover identity pairs, swaps/ties, and a controlled condition
            # number up to 1e6 without overflow/underflow of the Pfaffian.
            ds=b==3 && n>2 ? exp10.(range(-3,3;length=n÷2)) : ones(n÷2)
            for k in 1:n÷2
                D[2k-1,2k]=ds[k];D[2k,2k-1]=-ds[k]
            end
            A=Q*D*transpose(Q)
            matrices[:,:,b]=(A-transpose(A))/2
            expected_pf[b]=det(Q)*prod(ds)
        end
        matrices[:,:,4].=0 # Exactly singular, with poisoned reusable buffers.
        lu=CuArray(matrices);inverse=similar(lu)
        native=GPU.BatchedSamplingInverse(lu,copy(lu),inverse,
            CuArray([pointer(lu,(b-1)*n*n+1) for b in 1:batch]),
            CuArray([pointer(inverse,(b-1)*n*n+1) for b in 1:batch]),
            CUDA.zeros(Int32,n,batch),CUDA.zeros(Int32,n,batch),CUDA.zeros(Float64,batch),
            CUDA.zeros(Int32,batch),CUDA.zeros(Int32,batch))
        work=GPU.PairedSamplingInverse(native,batch;capacity)
        fill!(work.diagonal,NaN)
        GPU.paired_factor_inverse!(work,n,batch)
        CUDA.@cuda threads=128 blocks=batch GPU.paired_inverse_status!(native.inverse_info,native.inverse,n)
        actual=Array(inverse);pf=Array(native.pfaffians)
        @test pf[4]==0
        # The native route can report singularity through either LAPACK
        # status array; the paired route reports its nonfinite inverse.
        @test capacity==0 ? (Array(native.lu_info)[4]!=0 || Array(native.inverse_info)[4]!=0) : Array(native.inverse_info)[4]!=0
        @test all(iszero,Array(native.inverse_info)[1:3])
        for b in 1:3
            A=matrices[:,:,b];X=actual[:,:,b];oracle=inv(A)
            residual=opnorm(A*X-I,Inf)/(opnorm(A,Inf)*opnorm(X,Inf))
            condition=opnorm(A,Inf)*opnorm(oracle,Inf)
            # Backward residual is dimension-scaled roundoff. The forward
            # bound includes the independently computed matrix condition.
            @test residual <= 64n*eps(Float64)
            @test opnorm(X-oracle,Inf)/opnorm(oracle,Inf) <= 64n*eps(Float64)*condition
            @test maximum(abs,X+transpose(X))==0
            @test isapprox(pf[b],expected_pf[b];atol=0,rtol=64n*eps(Float64)*condition)
            cpu_pf=PfaPack.pfaffian_ltl!(copy(A))
            @test isapprox(pf[b],cpu_pf;atol=0,rtol=64n*eps(Float64)*condition)
        end
    end
end

@testset "Paired boundary cancellation, singular sectors and startup reuse" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,4,rng)
    cfg[:,1].=CANCELLATION_CFG
    s=gpu_state(data;configurations=cfg)
    cache=GPU.GPUSamplingCache(s)
    work=GPU.PairedSamplingInverse(s)
    expected_overlap=Array(s.overlaps)
    GPU.refresh_sampling_state!(s,cache,work)
    @test Array(work.fallback)[1]
    @test all(isapprox.(Array(s.overlaps),expected_overlap;atol=1e-9,rtol=1e-9))
    original_slater=Array(s.slater)
    for singular in (false,true)
        slater=copy(original_slater)
        singular && (slater[:,:,1].=0)
        copyto!(s.slater,slater)
        fill!(work.diagonal,NaN)
        GPU.refresh_sampling_state!(s,cache,work)
        inverse=Array(cache.inverse);pf=Array(cache.pfaffians)
        valid=Array(cache.valid)
        singular && @test all(Array(work.fallback))
        for b in 1:4,q in 1:s.nqp
            sites=cfg[:,b].+vcat(zeros(Int,s.nelec),fill(s.nsite,s.nelec)).+1
            A=-slater[sites,sites,q]
            @test isapprox(pf[q,b],PfaPack.pfaffian_ltl!(copy(A));atol=1e-9,rtol=1e-9)
            if singular && q==1
                @test !valid[q,b]
            else
                @test valid[q,b]
                X=inverse[:,:,q,b]
                @test opnorm(A*X-I,Inf)/(opnorm(A,Inf)*opnorm(X,Inf)) <= 64size(A,1)*eps(Float64)
            end
        end
    end
    s,pf=GPU.initial_gpu_state(data,4;seed=79,return_pfaffians=true)
    a,b=GPU.GPUSamplingCache(s),GPU.GPUSamplingCache(s)
    GPU.refresh_sampling_state!(s,a,GPU.BatchedSamplingInverse(s);initial_pfaffians=pf)
    GPU.refresh_sampling_state!(s,b,GPU.PairedSamplingInverse(s);initial_pfaffians=pf)
    @test Array(a.valid)==Array(b.valid)
    @test all(isapprox.(Array(a.pfaffians),Array(b.pfaffians);atol=1e-9,rtol=1e-9))
    @test all(isapprox.(Array(a.inverse),Array(b.inverse);atol=1e-9,rtol=1e-9))
end

@testset "Paired factor capacity follows the device scratch budget" begin
    @test GPU.paired_factor_capacity(128,32*128*4)==4
    @test GPU.paired_factor_capacity(128,32*128*3)==3
    @test GPU.paired_factor_capacity(128,32*128-1)==0
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    s=GPU.initial_gpu_state(data,4;seed=79)
    a,b=GPU.GPUSamplingCache(s),GPU.GPUSamplingCache(s)
    native=GPU.BatchedSamplingInverse(s)
    work=GPU.PairedSamplingInverse(GPU.BatchedSamplingInverse(s),4;capacity=0)
    GPU.refresh_sampling_state!(s,a,native)
    GPU.refresh_sampling_state!(s,b,work)
    @test all(Array(work.fallback))
    @test Array(a.valid)==Array(b.valid)
    @test all(isapprox.(Array(a.pfaffians),Array(b.pfaffians);atol=1e-9,rtol=1e-9))
    @test all(isapprox.(Array(a.inverse),Array(b.inverse);atol=1e-9,rtol=1e-9))
    @test_throws ArgumentError GPU.PairedSamplingInverse(native,4;capacity=5)
end
