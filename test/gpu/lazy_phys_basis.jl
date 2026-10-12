using Test,CUDA,MVMCOptimizers,LinearAlgebra
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false);BLAS.set_num_threads(1)
@testset "PhysCal omits derivative basis and materializes it on SR request" begin
 path=joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def")
 data,rng=GPU.prepared_gpu_data(path;seed=79,physcal=true)
 baseline,seed_rng,pfs=GPU.prepared_runner_state(data,4,rng,79)
 @test baseline.basis!==nothing
 old=GPU.gpu_phys_cal_runner!(data;configurations=baseline.configurations,nsamples=2,warmup=128,interval=128,seed=79,state=baseline,seed_rng,initial_pfaffians=pfs)
 future_old=Array(CUDA.rand(UInt32,64))
 new=CPU.run_gpu_phys_cal_from_namelist(path;nsamples=2,walkers=4,device=parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")),seed=79,warmup=128,interval=128)
 future_new=Array(CUDA.rand(UInt32,64))
 @test new.state.basis===nothing
 @test new.status==old.status==0
 @test new.energy≈old.energy atol=1e-9 rtol=1e-9
 @test new.green_one≈old.green_one atol=1e-9 rtol=1e-9
 @test new.green_two≈old.green_two atol=1e-9 rtol=1e-9
 @test Array(new.state.configurations)==Array(old.state.configurations)
 @test Array(new.state.counts)==Array(old.state.counts)
 @test future_new==future_old
 cfg=Int.(Array(new.state.configurations))
 default=CPU.gpu_state(new.data;configurations=cfg,initialize_wavefunctions=false)
 @test default.basis!==nothing
 @test_throws MethodError CPU.gpu_state(new.data;configurations=cfg,needs_derivatives=false)
 lazy=GPU.CUDASR(new.state,new.data)
 regular=GPU.CUDASR(baseline,data)
 @test lazy.sparse_basis!==nothing
 @test new.state.basis===nothing # Workspace owns the materialized basis.
 @test Array(lazy.sparse_basis.colPtr)==Array(regular.sparse_basis.colPtr)
 @test Array(lazy.sparse_basis.rowVal)==Array(regular.sparse_basis.rowVal)
 @test Array(lazy.sparse_basis.nzVal)≈Array(regular.sparse_basis.nzVal) atol=1e-14 rtol=1e-14
 o=GPU.CUDAObservables(new.state,new.data;measure_greens=false)
 CPU.gpu_observables!(new.state,o)
 GPU.derivatives!(new.state,lazy,o)
 GPU.derivatives!(new.state,regular,o)
 @test Array(lazy.derivatives)≈Array(regular.derivatives) atol=1e-9 rtol=1e-9
end
