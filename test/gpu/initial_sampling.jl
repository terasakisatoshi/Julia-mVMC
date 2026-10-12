using Test,CUDA,MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const CPU=MVMCOptimizers
const P=MVMCOptimizers.MVMCExpertModeParsers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
function bounded_probe!(output,tape,bounds)
    b=CUDA.threadIdx().x
    @inbounds begin
        value,cursor,ok=GPU.initial_bounded_word(tape,b,1,bounds[b])
        output[1,b]=value;output[2,b]=cursor;output[3,b]=ok
    end
    return nothing
end
@testset "Unbiased initial bounded UInt32 selection" begin
    tape=CuArray(UInt32[0 0 0xffffffff;4 1 0xffffffff;9 2 0])
    bounds=CuArray(Int64[3,1,2147483649])
    out=CUDA.zeros(Int64,3,3)
    CUDA.@cuda threads=3 bounded_probe!(out,tape,bounds)
    @test Array(out)==Int64[1 0 2147483646;3 2 2;1 1 1]
    exhausted=CuArray(zeros(UInt32,2,1));out2=CUDA.zeros(Int64,3,1)
    CUDA.@cuda threads=1 bounded_probe!(out2,exhausted,CuArray(Int64[3]))
    @test Array(out2)==reshape(Int64[0,3,0],3,1)
end
@testset "Known permutation assigns constrained electron labels" begin
    cfg=CUDA.zeros(Int32,4,1);scratch=CUDA.zeros(Int32,8,1)
    localspins=CuArray(Int32[1,0,1,0]);pending=CUDA.fill(true,1);candidate=CUDA.zeros(Bool,1)
    CUDA.@cuda threads=1 GPU.initial_configuration_kernel!(cfg,scratch,localspins,
        CuArray(fill(UInt32(12),16,1)),pending,candidate,4,2,1)
    # All selected swap indices are zero. The label cycle is1,2,3,0;
    # the two nonlocal-site swaps reverse and then restore their order.
    @test vec(Array(cfg))==Int32[3,0,2,1]
    @test only(Array(candidate))
end
@testset "Device initial states preserve filling and repeatable RNG" begin
    for model in ("hubbard_chain_real","heisenberg_chain_real")
        path=joinpath(@__DIR__,"..","..","examples","inputs",model,"namelist.def")
        data,_=GPU.prepared_gpu_data(path;seed=79)
        a=GPU.initial_gpu_state(data,16;seed=123)
        cfg=Array(a.configurations)
        future=Array(CUDA.rand(UInt32,16))
        b=GPU.initial_gpu_state(data,16;seed=123)
        @test cfg==Array(b.configurations)
        @test future==Array(CUDA.rand(UInt32,16))
        @test all(isfinite,Array(a.overlaps)) && all(!=(0),Array(a.overlaps))
        loc=CPU.get_loc_spn_array(data)
        ns,ne=a.nsite,a.nelec
        for w in axes(cfg,2)
            @test length(unique(cfg[1:ne,w]))==ne
            @test length(unique(cfg[ne+1:2ne,w]))==ne
            @test all(x->0<=x<ns,cfg[:,w])
            @test all(site->loc[site+1]!=1 || count(==(site),cfg[:,w])==1,0:ns-1)
        end
        c=cpu_state(data)
        expected=[cpu_overlap(data,c,cfg[:,w]).ip for w in axes(cfg,2)]
        @test Array(a.overlaps) ≈ expected atol=1e-9 rtol=1e-9
    end
end
@testset "Device initializer bounded failure for zero amplitude" begin
    path=joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    CPU.unpack_parameters!(data,zeros(ComplexF64,CPU.count_total_parameters(data)))
    @test_throws ArgumentError GPU.initial_gpu_state(data,2;seed=123,max_attempts=2)
end
@testset "L128 device initialization independent CPU overlap" begin
    path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    s=GPU.initial_gpu_state(data,8;seed=123)
    cfg=Array(s.configurations)
    c=cpu_state(data)
    expected=[cpu_overlap(data,c,cfg[:,w]).ip for w in axes(cfg,2)]
    @test all(!=(0),expected)
    @test Array(s.overlaps) ≈ expected atol=1e-9 rtol=1e-9
    @test all(w->length(unique(cfg[1:s.nelec,w]))==s.nelec &&
                 length(unique(cfg[s.nelec+1:end,w]))==s.nelec,axes(cfg,2))
end
@testset "Initial state rejects singular QP despite nonzero projected sum" begin
    path=joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,_=GPU.prepared_gpu_data(path;seed=79)
    s=GPU.initial_gpu_state(data,1;seed=123)
    # Force only the first sector singular; all remaining sector factors are
    # nonsingular. Weighted sum stays nonzero and is independently checked.
    slater=Array(s.slater);slater[:,:,1].=0;copyto!(s.slater,slater)
    pending=CUDA.fill(true,1);candidate=CUDA.fill(true,1)
    CUDA.@cuda threads=128 blocks=1 GPU.initial_wavefunction_kernel!(GPU.initial_state_view(s),pending,candidate)
    @test only(Array(pending))
    @test isfinite(only(Array(s.overlaps))) && only(Array(s.overlaps)) != 0
end
@testset "Optional initial Pfaffians preserve RNG and independent CPU factors" begin
    for model in ("hubbard_chain_real","L128")
        path=model=="L128" ? joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def") :
            joinpath(@__DIR__,"..","..","examples","inputs",model,"namelist.def")
        data,_=GPU.prepared_gpu_data(path;seed=79)
        plain=GPU.initial_gpu_state(data,8;seed=123)
        future=Array(CUDA.rand(UInt32,16))
        s,pfs=GPU.initial_gpu_state(data,8;seed=123,return_pfaffians=true)
        @test Array(s.configurations)==Array(plain.configurations)
        @test Array(CUDA.rand(UInt32,16))==future
        @test size(pfs)==(s.nqp,8)
        cfg=Array(s.configurations)
        c=cpu_state(data)
        expected=zeros(Float64,s.nqp,8)
        for b in 1:8
            cpu_overlap(data,c,cfg[:,b])
            expected[:,b].=c.slater_matrix.pf_m_real
        end
        @test Array(pfs) ≈ expected atol=1e-9 rtol=1e-9
        @test all(isfinite,Array(pfs)) && all(!=(0),Array(pfs))
    end
end
