using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
@testset "Exact cofactors of a singular individual QP sector" begin
    path = joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng = GPU.prepared_gpu_data(path;seed=79)
    cfg = GPU.initial_configurations(data,1,rng)
    s = gpu_state(data;configurations=cfg)
    ns,ne = s.nsite,s.nelec
    @test ne == 3
    sites = Int.(cfg[:,1]) .+ vcat(zeros(Int,ne),fill(ns,ne)) .+ 1
    slater = zeros(Float64,2ns,2ns,s.nqp)
    # QP1 has rank4: A15=A26=1; its only nonzero Pfaffian cofactor is
    # C34=-1. QP2 has A14=A25=A36=1, Pfaffian=-1, C14=C25=C36=-1.
    for (q,pairs) in ((1,((1,5),(2,6))),(2,((1,4),(2,5),(3,6))))
        for (i,j) in pairs
            slater[sites[i],sites[j],q] = -1
            slater[sites[j],sites[i],q] = 1
        end
    end
    copyto!(s.slater,slater)
    gpu_wavefunctions!(s)
    basis,weights = GPU.slater_basis(data,length(s.parameters),ns,ne,s.nqp,s.nproj)
    @test Array(s.overlaps) ≈ [-weights[2]] atol=1e-14 rtol=1e-14
    row(i,j,q) = sites[i]+(sites[j]-1)*2ns+(q-1)*4ns*ns
    expected = -(weights[1].*basis[row(3,4,1),:] .+
                weights[2].*(basis[row(1,4,2),:]+basis[row(2,5,2),:]+basis[row(3,6,2),:])) ./ weights[2]
    occ = zeros(Int,2ns)
    occ[sites] .= 1
    counts = zeros(Int,s.nproj)
    MVMCOptimizers.make_proj_cnt!(counts,occ,data)
    expected[1:s.nproj] .= counts
    sr = GPU.CUDASR(s,data)
    GPU.derivatives!(s,sr)
    @test vec(Array(sr.derivatives)) ≈ expected atol=1e-12 rtol=1e-12
    o = gpu_observables_workspace(s,data)
    gpu_observables!(s,o)
    @test !Array(o.valid)[1,1]
    GPU.derivatives!(s,sr,o)
    @test vec(Array(sr.derivatives)) ≈ expected atol=1e-12 rtol=1e-12
end
