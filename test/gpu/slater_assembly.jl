using Test, CUDA, MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
const P = MVMCOptimizers.MVMCExpertModeParsers
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
include("cpu_oracle.jl")
@testset "Resident direct Slater assembly matches CPU operation order" begin
    path = joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_ = GPU.prepared_gpu_data(path;seed=79)
    ns = data.modpara.nsite
    np = P.projection_layout(data).n_proj
    nq = CPU.get_n_qp_full(data)
    assembly = GPU.SlaterAssembly(data,np,ns,nq)
    state = (;parameters=CuArray(real.(CPU.pack_parameters(data))),
              slater=CUDA.zeros(Float64,2ns,2ns,nq))
    GPU.assemble_slater!(state,assembly)
    actual = Array(state.slater)
    oracle = reshape(real.(cpu_state(data).slater_matrix.slater_elm),2ns,2ns,nq)
    # Independent CPU assembly, same parameters and projection metadata. Do not
    # require computed floating-point bitwise equality under the parity policy.
    @test actual ≈ oracle atol=2eps(Float64) rtol=2eps(Float64)
    @test all(isfinite,actual)
    @test actual ≈ -permutedims(actual,(2,1,3)) atol=2eps(Float64) rtol=2eps(Float64)
    println("L128 direct Slater maximum CPU difference: ",maximum(abs.(actual.-oracle)))
end
@testset "C four-block assembly retains tiny raw parameters" begin
    # C slater.c uses raw Slater parameters without Julia's historical 1e-14
    # orbital-value filter. Two independent parameters exercise all four blocks.
    ns = 2
    pa = reshape(Int32[1,2,1,1],2,2,1)
    pb = reshape(Int32[1,1,2,1],2,2,1)
    signs = ones(Int32,2,2,1)
    assembly = GPU.SlaterAssembly(CuArray(pa),CuArray(pb),CuArray(signs),CuArray(signs),
                                 CuArray([0.3]),CuArray([0.7]),CuArray([0.2]),ns,1)
    state = (;parameters=CuArray([5e-15,-3e-15]),slater=CUDA.zeros(Float64,4,4,1))
    GPU.assemble_slater!(state,assembly)
    out = Array(state.slater)
    # For ri=1,rj=2: a=5e-15,b=-3e-15. These closed-form C outputs
    # would all vanish if either parameter were filtered during assembly.
    @test out[2,1,1] ≈ -2.4e-15 atol=1e-30 rtol=1e-15
    @test out[4,1,1] ≈ 2.9e-15 atol=1e-30 rtol=1e-15
    @test out[2,3,1] ≈ 1.1e-15 atol=1e-30 rtol=1e-15
    @test out[4,3,1] ≈ 2.4e-15 atol=1e-30 rtol=1e-15
end
@testset "Direct sparse Slater coefficients match CPU unit assembly" begin
    for model in ("hubbard_chain_real","L128")
        path = model == "L128" ? joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def") :
            joinpath(@__DIR__,"..","..","examples","inputs",model,"namelist.def")
        data,_ = GPU.prepared_gpu_data(path;seed=79)
        ns,ne = data.modpara.nsite,data.modpara.nelec
        np = CPU.count_total_parameters(data)
        nproj = P.projection_layout(data).n_proj
        nq = CPU.get_n_qp_full(data)
        basis = GPU.sparse_slater_basis(GPU.SlaterAssembly(data,nproj,ns,nq),np)
        ptr,rows,values = Array(basis.colPtr),Array(basis.rowVal),Array(basis.nzVal)
        selected = collect(1:np)
        oracledata = deepcopy(data)
        oracle = CPU.VMCOptimizationState(ns,ne,nproj,np,nq,1,false,false)
        unit = zeros(ComplexF64,np)
        for p in selected
            fill!(unit,0);unit[p]=1
            CPU.unpack_parameters!(oracledata,unit)
            CPU.update_slater_elm_fcmp!(oracledata,oracle)
            column = zeros(Float64,size(basis,1))
            for k in ptr[p]:ptr[p+1]-1
                column[rows[k]] = values[k]
            end
            @test column ≈ real.(oracle.slater_matrix.slater_elm) atol=eps(Float64) rtol=eps(Float64)
            @test issorted(rows[ptr[p]:ptr[p+1]-1])
        end
        @test length(values)*12+length(ptr)*4 < size(basis,1)*np*8
    end
end
