using CUDA,MVMCOptimizers,LinearAlgebra,Statistics
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
result=CPU.run_gpu_para_opt_from_namelist("/tmp/jl_emoPro/input/namelist.def";nsteps=20,walkers=320,device=1,seed=79,warmup=128,interval=128)
s=result.state;data=result.data;n=2s.nelec;batch=320;count=batch*s.nqp
cache=GPU.GPUSamplingCache(s);inverse_work=GPU.BatchedSamplingInverse(s)
GPU.refresh_sampling_state!(s,cache,inverse_work)
function record!(f,t,name)
 CUDA.synchronize();start=time_ns();f();CUDA.synchronize()
 t[name]=(wall_seconds=(time_ns()-start)/1e9,)
end
function parts!(s,o,cache)
 n=2s.nelec;batch=size(s.configurations,2);count=batch*s.nqp;work=o.batched_inverse
 t=Dict{Symbol,NamedTuple}()
 record!(t,:publish_sampling_cache) do
  GPU.observable_sampling_cache!(s,o,cache)
 end
 record!(t,:target_map) do
  @cuda threads=128 blocks=cld(batch,128) GPU.target_sites_kernel!(work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
 end
 record!(t,:fill_slater_rows) do
  entries=length(work.slater_rows)
  @cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.fill_batched_slater_rows!(work.slater_rows,work.target_sites,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
 end
 record!(t,:transformed_gemm) do
  CUDA.cuBLAS.gemm_strided_batched!('N','N',work.alpha,work.inverse,work.slater_rows,work.beta,work.transformed)
 end
 record!(t,:publish_transform) do
  entries=n*2s.nsite*count
  @cuda threads=256 blocks=cld(entries,256) fastmath=false GPU.publish_batched_transform!(GPU.cache_view(o.transformed),work.transformed,work.target_map,n,s.nsite,batch,count)
 end
 record!(t,:parallel_operator) do
  terms=size(o.termvalues,1)
  @cuda threads=256 blocks=cld(terms*batch,256) fastmath=false GPU.parallel_operator_kernel!(o.termvalues,o.ops,o.one,o.two,o.ex1,o.ex2,s.configurations,s.occupancy,s.overlaps,s.logcorrelations,s.gutz,s.jastrow,s.parameters,s.slater,s.weights,GPU.factor_view(s.factors),GPU.pivot_view(s.pivots),s.pfaffians,GPU.cache_view(o.inverse),o.pfaffians,o.valid,GPU.cache_view(o.transformed),s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch)
 end
 record!(t,:ordered_reduce) do
  @cuda threads=128 blocks=cld(batch,128) fastmath=false GPU.ordered_observable_reduce_kernel!(o.values,o.termvalues,o.coeffs,o.valid,s.nqp,batch)
 end
 record!(t,:scalar_fallback) do
  @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false GPU.observable_kernel!(o.values,o.ops,o.coeffs,o.one,o.two,o.ex1,o.ex2,s.configurations,s.counts,s.occupancy,s.overlaps,s.logcorrelations,s.gutz,s.jastrow,s.parameters,s.slater,s.weights,GPU.factor_view(s.factors),GPU.pivot_view(s.pivots),s.pfaffians,GPU.cache_view(o.inverse),o.pfaffians,o.valid,GPU.cache_view(o.transformed),s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch,true)
 end
 return t
end
for greens in (false,true)
 o=GPU.CUDAObservables(s,data;measure_greens=greens)
 parts!(s,o,cache)
 runs=[]
 for rep=1:5
  t=parts!(s,o,cache);push!(runs,t)
  println("observable rep=",rep," greens=",greens," stages=",t);flush(stdout)
 end
 for key in sort!(collect(keys(runs[1]));by=string)
  println("observable_median greens=",greens," name=",key," wall_seconds=",median([t[key].wall_seconds for t in runs]))
 end
 println("term_count greens=",greens," terms=",size(o.termvalues,1)," values_rows=",size(o.values,1)," all_qp_valid=",all(Array(o.valid))," finite_values=",all(isfinite,Array(o.values)))
 flush(stdout)
end
