using CUDA,MVMCOptimizers,LinearAlgebra,Statistics,SHA
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
const INPUT="/tmp/jl_emoPro/input/namelist.def"
const CHECKPOINT="/tmp/jl_emoPro/fixed/zqp_opt.dat"
function initialize(data,isphys)
 GPU.initial_gpu_state(data,320;seed=79,return_pfaffians=true,needs_derivatives=!isphys)
end
function setup(s,data,isphys)
 cache=GPU.GPUSamplingCache(s)
 work=GPU.BatchedSamplingInverse(s)
 o=GPU.CUDAObservables(s,data;measure_greens=isphys)
 return (cache,work,o)
end
function combined(data,isphys)
 s,pf=initialize(data,isphys)
 resources=setup(s,data,isphys)
 return (s,pf,resources)
end
function timed(f,reps=7)
 values=Float64[]
 for rep=1:reps
  GC.gc();CUDA.synchronize()
  start=time_ns();result=f();CUDA.synchronize()
  push!(values,(time_ns()-start)/1e9)
  result=nothing
 end
 return (minimum(values),median(values),maximum(values))
end
for isphys in (true,false)
 name=isphys ? :phys : :opt
 data,_=GPU.prepared_gpu_data(INPUT;seed=79,physcal=isphys,opt_para=isphys ? CHECKPOINT : nothing)
 s,pf=initialize(data,isphys);resources=setup(s,data,isphys);CUDA.synchronize()
 cfg=Array(s.configurations);future=Array(CUDA.rand(UInt32,64))
 println("setup_baseline kind=",name," configuration_sha256=",bytes2hex(sha256(reinterpret(UInt8,vec(cfg))))," future_rng_sha256=",bytes2hex(sha256(reinterpret(UInt8,future)))," counts_sum=",sum(Array(s.counts)))
 println("setup_workspace kind=",name," internal_rows_empty=",isempty(resources[3].batched_inverse.slater_rows)," internal_transform_empty=",isempty(resources[3].batched_inverse.transformed));flush(stdout)
 # Warm each measured path before recording allocation-inclusive results.
 initialize(data,isphys);setup(s,data,isphys);combined(data,isphys);CUDA.synchronize()
 println("allocation_setup kind=",name," phase=initialization min_median_max_wall=",timed(()->initialize(data,isphys)));flush(stdout)
 println("allocation_setup kind=",name," phase=workspaces min_median_max_wall=",timed(()->setup(s,data,isphys)));flush(stdout)
 println("allocation_setup kind=",name," phase=combined min_median_max_wall=",timed(()->combined(data,isphys)));flush(stdout)
 s=nothing;pf=nothing;resources=nothing;GC.gc();CUDA.synchronize()
end
