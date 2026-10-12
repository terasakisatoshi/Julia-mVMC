using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
const P = MVMCOptimizers.MVMCExpertModeParsers
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
@testset "Resident sparse Slater derivative contraction" begin
    rng = MersenneTwister(81)
    rows, np, batch = 2048, 65, 7
    basis = zeros(Float64,rows,np)
    for p in 2:np, k in 1:32
        basis[mod(157p+13k,rows)+1,p] = randn(rng)
    end
    sensitivity = randn(rng,rows,batch)
    sparse = CUDA.cuSPARSE.CuSparseMatrixCSC(CuArray(basis))
    output = CUDA.zeros(Float64,np,batch)
    device_sensitivity = CuArray(sensitivity)
    CUDA.@cuda threads=128 blocks=cld(np*batch,128) fastmath=false GPU.sparse_derivative_contraction!(
        output,device_sensitivity,sparse.colPtr,sparse.rowVal,sparse.nzVal,np,batch)
    actual = Array(output)
    # CPU dense BLAS is an independent contraction oracle. At most32 unit-scale
    # terms contribute per output; the rounding bound is comfortably below1e-11.
    @test actual ≈ transpose(basis)*sensitivity atol=1e-11 rtol=1e-11
    @test actual[1,:] == zeros(batch) # empty projection column
    @test length(sparse.nzVal) == 32*(np-1)
end

include("cpu_oracle.jl")
@testset "L128 sparse analytic derivatives match CPU" begin
    path = joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,rng = GPU.prepared_gpu_data(path;seed=81)
    cfg = GPU.initial_configurations(data,1,rng)
    state = gpu_state(data;configurations=cfg)
    sr = GPU.CUDASR(state,data)
    @test state.nsite == 128
    @test sr.sparse_basis !== nothing
    GPU.derivatives!(state,sr)
    c = cpu_state(data)
    x = cpu_overlap(data,c,cfg[:,1])
    CPU.convert_from_real_arrays!(c)
    c.slater_matrix.pf_m .= c.slater_matrix.pf_m_real
    no = CPU.count_total_parameters(data)-length(x.counts)
    orbital = zeros(ComplexF64,2no)
    CPU.slater_elm_diff_fcmp!(orbital,ComplexF64(x.ip),Int.(cfg[:,1]),data,c;real_inverse=true)
    expected = vcat(Float64.(x.counts),real.(orbital[1:2:end]))
    actual = vec(Array(sr.derivatives))
    @test all(isfinite,expected)
    @test actual ≈ expected atol=1e-9 rtol=1e-9
    # cuSPARSE matrix multiplication independently checks the custom contraction;
    # finish_derivatives! separately overwrites projection rows with counts.
    dense = similar(sr.derivatives)
    mul!(dense,transpose(state.basis),sr.sensitivity)
    @test vec(Array(dense))[state.nproj+1:end] ≈ actual[state.nproj+1:end] atol=1e-9 rtol=1e-9
    @test actual[1:state.nproj] ≈ Float64.(vec(Array(state.counts))) atol=0 rtol=0
    # Build only the inverse cache; avoid the unrelated full Green workload.
    o = GPU.CUDAObservables(state,data;measure_greens=false)
    if o.batched_inverse !== nothing
        GPU.batched_inverse_cache!(state,o,o.batched_inverse)
    else
        CUDA.@cuda threads=256 blocks=size(cfg,2) shmem=16state.nelec fastmath=false GPU.cooperative_inverse_cache_kernel!(
            GPU.cache_view(o.inverse),o.pfaffians,o.valid,state.configurations,state.slater,
            o.packed_factors,o.packed_inverse,GPU.pivot_view(state.pivots),state.nsite,state.nelec,
            state.nqp,size(cfg,2))
    end
    @test all(Array(o.valid))
    GPU.derivatives!(state,sr,o)
    @test vec(Array(sr.derivatives)) ≈ expected atol=1e-9 rtol=1e-9
end

@testset "Large cached sparse path retains singular-sector exact cofactors" begin
    ns,ne,nq,batch = 64,32,2,2
    n = 2ne
    cfg = hcat(vcat(collect(0:ne-1),collect(0:ne-1)),
               vcat(collect(ne:ns-1),collect(ne:ns-1)))
    A = zeros(Float64,2ns,2ns)
    for i in 1:2:2ns
        A[i,i+1] = 1
        A[i+1,i] = -1
    end
    slater = cat(-A,-A;dims=3)
    # Walker1's first QP has rank62. The only nonzero Pfaffian cofactor
    # removes its last electron pair. Walker2 does not occupy that pair.
    ri,rj = ns+ne-1,ns+ne
    slater[ri,rj,1] = slater[rj,ri,1] = 0
    basis = zeros(Float64,4ns*ns*nq,2)
    for q in 1:nq
        basis[ri+(rj-1)*2ns+(q-1)*4ns*ns,2] = -1
        basis[rj+(ri-1)*2ns+(q-1)*4ns*ns,2] = 1
    end
    state = GPU.CUDAState(ns,ne,1,nq,32,CuArray([0.0,1.0]),CuArray(basis),CuArray(slater),
        CuArray([0.25,0.75]),CuArray(Int32.(cfg)),CUDA.zeros(Int32,2ns,batch),
        CuArray(reshape(Int32[ne,0],1,batch)),CuArray(vcat(zeros(Int32,ne),fill(Int32(-1),ne))),CuArray(fill(Int32(-1),ns,ns)),
        1,0,CUDA.zeros(Float64,batch,n,n),CUDA.zeros(Int32,batch,n),CUDA.zeros(Float64,batch),
        CuArray([0.75,1.0]),CUDA.zeros(Float64,batch),CUDA.zeros(Int32,ns),
        CUDA.zeros(Int32,batch),CUDA.zeros(Float64,n,n,batch),nothing)
    compressed = CUDA.cuSPARSE.CuSparseMatrixCSC(state.basis)
    sr = GPU.CUDASR(CuArray(fill(NaN,2,batch)),similar(state.factors),CuArray([true,true]),
        CuArray(fill(NaN,size(basis,1),batch)),CuArray([1.0]),CuArray([0.0]),
        compressed,GPU.decoded_sparse_basis(compressed,ns))
    work = GPU.BatchedInverse(state)
    o = (;inverse=CUDA.zeros(Float64,batch,n,n,nq),pfaffians=CUDA.zeros(Float64,nq,batch),
         valid=CUDA.zeros(Bool,nq,batch),batched_inverse=work)
    GPU.batched_inverse_cache!(state,o,work)
    @test Array(o.valid) == Bool[false true;true true]
    # The dispatch method requires the normal workspace type; no observable
    # kernels run, so empty operator tables suffice.
    workspace = GPU.CUDAObservables(CUDA.zeros(Float64,1,batch),CuArray(GPU.GPUOperator[]),
        CuArray(Float64[]),CuArray(GPU.GPUOperator[]),CuArray(GPU.GPUOperator[]),
        CuArray(GPU.GPUOperator[]),CuArray(GPU.GPUOperator[]),o.inverse,o.pfaffians,o.valid,
        CUDA.zeros(Float64,n,n,0),CUDA.zeros(Float64,n,n,0),
        CUDA.zeros(Float64,batch,n,2ns,nq),CUDA.zeros(Float64,0,batch),work,nothing)
    GPU.derivatives!(state,sr,workspace)
    @test Array(sr.derivatives) ≈ [Float64(ne) 0.0;4/3 0.0] atol=1e-12 rtol=1e-12
    @test Array(sr.derivatives)[1,:] ≈ Float64.(vec(Array(state.counts))) atol=0 rtol=0
    sensitivity = Array(sr.sensitivity)
    @test all(isfinite,sensitivity[:,1])
    @test all(isnan,sensitivity[:,2]) # all-valid walker avoided materialization
end
