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
function cleanup()
 GC.gc(true);CUDA.synchronize();GC.gc(true);CUDA.reclaim();CUDA.synchronize()
end
function measure_one(f)
 start=time_ns();result=f();CUDA.synchronize()
 elapsed=(time_ns()-start)/1e9
 return elapsed
end
function warm_one(f)
 result=f();CUDA.synchronize()
 return nothing
end
function timed(f,reps=7)
 warm_one(f);cleanup()
 values=Float64[]
 for rep=1:reps
  cleanup();push!(values,measure_one(f))
 end
 cleanup()
 return (minimum(values),median(values),maximum(values))
end
for isphys in (true,false)
 name=isphys ? :phys : :opt
 data,_=GPU.prepared_gpu_data(INPUT;seed=79,physcal=isphys,opt_para=isphys ? CHECKPOINT : nothing)
 s,pf=initialize(data,isphys);resources=setup(s,data,isphys);CUDA.synchronize()
 cfg=Array(s.configurations);future=Array(CUDA.rand(UInt32,64))
 println("setup_baseline kind=",name," configuration_sha256=",bytes2hex(sha256(reinterpret(UInt8,vec(cfg))))," future_rng_sha256=",bytes2hex(sha256(reinterpret(UInt8,future)))," counts_sum=",sum(Array(s.counts)))
 println("setup_workspace kind=",name," internal_rows_empty=",isempty(resources[3].batched_inverse.slater_rows)," internal_transform_empty=",isempty(resources[3].batched_inverse.transformed));flush(stdout)
 resources=nothing;cleanup()
 println("allocation_setup kind=",name," phase=initialization min_median_max_wall=",timed(()->initialize(data,isphys)));flush(stdout)
 println("allocation_setup kind=",name," phase=workspaces min_median_max_wall=",timed(()->setup(s,data,isphys)));flush(stdout)
 println("allocation_setup kind=",name," phase=combined min_median_max_wall=",timed(()->combined(data,isphys)));flush(stdout)
 s=nothing;pf=nothing;resources=nothing;cleanup()
end
