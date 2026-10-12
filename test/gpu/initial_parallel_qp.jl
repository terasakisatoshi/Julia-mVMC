using CUDA,Test,MVMCOptimizers,LinearAlgebra
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")));CUDA.allowscalar(false);BLAS.set_num_threads(1)
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
data,_=GPU.prepared_gpu_data(joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def");seed=79)
s=GPU.initial_gpu_state(data,8;seed=79)
work=GPU.parallel_initial_workspace(s)
@testset "Parallel QP initializer preserves ordered PF and state" begin
 for mode in (:ordinary,:singular,:masked)
  mode==:singular && (view(s.slater,:,:,1).=0.0)
  pending=CUDA.fill(true,8);candidate=CUDA.fill(true,8)
  if mode==:masked
   pending.=CuArray(Bool[true,true,false,true,true,true,true,true])
   candidate.=CuArray(Bool[true,false,true,true,true,true,true,true])
  end
  input_pending=copy(pending);input_candidate=copy(candidate)
  pf=CUDA.zeros(Float64,s.nqp,8)
  CUDA.@cuda threads=128 blocks=8 maxregs=64 fastmath=false GPU.initial_wavefunction_kernel!(GPU.initial_state_view(s),pending,candidate,pf)
  reference=(cfg=Array(s.configurations),counts=Array(s.counts),occ=Array(s.occupancy),
   logs=Array(s.logcorrelations),overlaps=Array(s.overlaps),pivots=Array(s.pivots),
   factors=Array(s.cooperative_factors),pf=Array(pf),pending=Array(pending))
  copyto!(pending,input_pending);copyto!(candidate,input_candidate);fill!(work.pfs,0.0)
  CUDA.seed!(123);future=Array(CUDA.rand(UInt32,32));CUDA.seed!(123)
  GPU.initial_wavefunctions_parallel!(s,pending,candidate,work)
  @test Array(s.configurations)==reference.cfg
  @test Array(s.counts)==reference.counts
  @test Array(s.occupancy)==reference.occ
  @test Array(s.pivots)==reference.pivots
  @test Array(pending)==reference.pending
  @test Array(s.logcorrelations)≈reference.logs atol=1e-12 rtol=1e-12
  @test Array(s.overlaps)≈reference.overlaps atol=1e-10 rtol=1e-12
  @test Array(s.cooperative_factors)≈reference.factors atol=1e-10 rtol=1e-12
  @test Array(work.pfs)≈reference.pf atol=1e-10 rtol=1e-12
  @test Array(CUDA.rand(UInt32,32))==future
 end
end
@testset "Private large Phys initializer preserves serial state and RNG" begin
 baseline,oldpf=GPU.initial_gpu_state(data,128;seed=123,return_pfaffians=true)
 future=Array(CUDA.rand(UInt32,32))
 candidate,newpf=GPU.initial_gpu_state(data,128;seed=123,return_pfaffians=true,needs_derivatives=false)
 @test candidate.basis===nothing
 @test Array(candidate.configurations)==Array(baseline.configurations)
 @test Array(candidate.counts)==Array(baseline.counts)
 @test Array(candidate.occupancy)==Array(baseline.occupancy)
 @test Array(candidate.pivots)==Array(baseline.pivots)
 @test Array(candidate.cooperative_factors)≈Array(baseline.cooperative_factors) atol=1e-10 rtol=1e-12
 @test Array(candidate.logcorrelations)≈Array(baseline.logcorrelations) atol=1e-12 rtol=1e-12
 @test Array(candidate.overlaps)≈Array(baseline.overlaps) atol=1e-10 rtol=1e-12
 @test Array(newpf)≈Array(oldpf) atol=1e-10 rtol=1e-12
 @test Array(CUDA.rand(UInt32,32))==future
end
