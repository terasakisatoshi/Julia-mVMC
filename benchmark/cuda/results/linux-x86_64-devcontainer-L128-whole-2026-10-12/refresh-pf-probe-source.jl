using CUDA,MVMCOptimizers,PfaPack,LinearAlgebra,Statistics,SHA,Test
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
const INPUT="/tmp/jl_emoPro/input/namelist.def"
# Constant packed strides only; factorization/pivot/RN instructions are reused.
struct StaticPFMatrix{N,T};data::T;end
Base.getindex(A::StaticPFMatrix{N},i,j,b) where N = @inbounds A.data[i+(j-1)*N+(b-1)*N*N]
Base.setindex!(A::StaticPFMatrix{N},x,i,j,b) where N = (@inbounds A.data[i+(j-1)*N+(b-1)*N*N]=x)
struct StaticPFPivots{N,T};data::T;end
Base.getindex(A::StaticPFPivots{N},i,b) where N = @inbounds A.data[i+(b-1)*N]
Base.setindex!(A::StaticPFPivots{N},x,i,b) where N = (@inbounds A.data[i+(b-1)*N]=x)
function static_pf_kernel!(values,A,pivots,::Val{N}) where N
    b=blockIdx().x
    pf=GPU.cooperative_pfaffian!(StaticPFMatrix{N,typeof(A)}(A),StaticPFPivots{N,typeof(pivots)}(pivots),N,b)
    if threadIdx().x==1
        @inbounds values[b]=pf
    end
    return nothing
end
function publish_only!(s,c)
 b=(blockIdx().x-1)*blockDim().x+threadIdx().x
 b>size(s.configurations,2)&&return nothing
 ip=0.0
 @inbounds for q=1:s.nqp
  ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],c.pfaffians[q,b]))
 end
 @inbounds s.overlaps[b]=ip
 @inbounds c.accepted_since_refactor[b]=0
 return nothing
end
function correlation_only!(s)
 b=(blockIdx().x-1)*blockDim().x+threadIdx().x
 b>size(s.configurations,2)&&return nothing
 @inbounds s.logcorrelations[b]=GPU.correlation!(s.counts,s.occupancy,s.configurations,s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
 return nothing
end
function record!(f,t,name)
 CUDA.synchronize();start=time_ns();f();CUDA.synchronize();wall=(time_ns()-start)/1e9;elapsed=wall
 t[name]=(wall_seconds=wall,)
 return nothing
end
println("scope=DevContainer GPU1 FP64 B320 n128 count2560; frozen production state after20SRsteps; synchronized wall timing; no device-event timings; no RNG in refresh/tuning")
flush(stdout)
result=CPU.run_gpu_para_opt_from_namelist(INPUT;nsteps=20,walkers=320,device=1,seed=79,warmup=128,interval=128)
result.status==0||error("runner failed")
s=result.state;n=2s.nelec;batch=320;count=batch*s.nqp;entries=n*n*count
cache=GPU.GPUSamplingCache(s);work=GPU.BatchedSamplingInverse(s)
sdev=GPU.cooperative_state(s);cdev=GPU.sampling_cache_state(cache)
handle=CUDA.cuBLAS.handle()
function stages!(s,cache,work,sdev,cdev,handle,n,batch,count,entries)
 t=Dict{Symbol,NamedTuple}()
 record!(t,:fill) do
  @cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.fill_batched_slater!(work.lu,work.factors,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
 end
 record!(t,:pf) do
  @cuda threads=128 blocks=count maxregs=64 fastmath=false GPU.cooperative_pfaffian_kernel!(work.pfaffians,work.factors,work.pf_pivots,n)
 end
 record!(t,:getrf) do
  CUDA.cuBLAS.cublasDgetrfBatched(handle,n,work.lu_ptrs,n,work.pivots,work.lu_info,count)
 end
 record!(t,:getri) do
  CUDA.cuBLAS.cublasDgetriBatched(handle,n,work.lu_ptrs,n,work.pivots,work.inverse_ptrs,n,work.inverse_info,count)
 end
 record!(t,:skew) do
  @cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.skew_inverse_kernel!(work.inverse,n,count)
 end
 record!(t,:publish_inverse) do
  @cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.publish_batched_inverse!(cache.inverse,cache.pfaffians,cache.valid,work.inverse,work.pfaffians,work.lu_info,work.inverse_info,n,batch,count)
 end
 record!(t,:publish_combined_correlation) do
  @cuda threads=32 blocks=cld(batch,32) fastmath=false GPU.publish_sampling_boundary!(sdev,cdev)
 end
 record!(t,:publish_only_diagnostic) do
  @cuda threads=32 blocks=cld(batch,32) fastmath=false publish_only!(sdev,cdev)
 end
 record!(t,:correlation_only_diagnostic) do
  @cuda threads=32 blocks=cld(batch,32) fastmath=false correlation_only!(sdev)
 end
 return t
end
stages!(s,cache,work,sdev,cdev,handle,n,batch,count,entries)
runs=[]
for rep=1:5
 t=stages!(s,cache,work,sdev,cdev,handle,n,batch,count,entries);push!(runs,t)
 for key in sort!(collect(keys(t));by=string)
  println("stage rep=",rep," name=",key," wall_seconds=",t[key].wall_seconds)
 end
 flush(stdout)
end
for key in sort!(collect(keys(runs[1]));by=string)
 println("stage_median name=",key," wall_seconds=",median([t[key].wall_seconds for t in runs]))
end
# Preserve the original full input and baseline factors/pivots before tuning.
@cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.fill_batched_slater!(work.lu,work.factors,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
original=copy(work.factors)
@cuda threads=128 blocks=count maxregs=64 fastmath=false GPU.cooperative_pfaffian_kernel!(work.pfaffians,work.factors,work.pf_pivots,n)
reference_values=Array(work.pfaffians);reference_pivots=Array(work.pf_pivots);reference_factors=copy(work.factors)
for b in (1,320,321,960,1280,2560)
 a=Array(view(original,:,:,b));p=zeros(Int32,n)
 @test CPU._ordinary_dsktf2_c_order!(a,p)==0
 @test isapprox(reference_values[b],PfaPack.utu2pfa(n,a,n,p);atol=1e-10,rtol=1e-12)
 @test reference_pivots[:,b]==p
 @test isapprox(Array(view(reference_factors,:,:,b)),a;atol=1e-10,rtol=1e-12)
end
println("independent_CPU_PF_checks=24pass")
flush(stdout)
for static in (false,true),threads in (32,64,128,256),cap in (32,48,64,128)
 copyto!(work.factors,original);CUDA.synchronize()
 k=if static
  @cuda launch=false maxregs=cap fastmath=false static_pf_kernel!(work.pfaffians,work.factors,work.pf_pivots,Val(128))
 else
  @cuda launch=false maxregs=cap fastmath=false GPU.cooperative_pfaffian_kernel!(work.pfaffians,work.factors,work.pf_pivots,n)
 end
 args=static ? (work.pfaffians,work.factors,work.pf_pivots,Val(128)) : (work.pfaffians,work.factors,work.pf_pivots,n)
 k(args...;threads,blocks=count);CUDA.synchronize()
 @test Array(work.pf_pivots)==reference_pivots
 @test isapprox(Array(work.pfaffians),reference_values;atol=1e-10,rtol=1e-12)
 # All factors are compared on device; only the scalar residual is read outside timing.
 error=maximum(abs.(work.factors.-reference_factors)./(1 .+abs.(reference_factors)))
 @test error<=1e-12
 samples=[]
 for _=1:5
  copyto!(work.factors,original);CUDA.synchronize()
  start=time_ns();k(args...;threads,blocks=count);CUDA.synchronize();elapsed=(time_ns()-start)/1e9
  push!(samples,elapsed)
 end
 println("PF_candidate static=",static," threads=",threads," maxregs=",cap," registers=",CUDA.registers(k)," memory=",CUDA.memory(k)," min_median_max=",(minimum(samples),median(samples),maximum(samples))," factor_normalized_error=",error," parity=PASS")
 flush(stdout)
end
println("all_candidates_finished=true; state_rng_untouched_by_refresh=true")
