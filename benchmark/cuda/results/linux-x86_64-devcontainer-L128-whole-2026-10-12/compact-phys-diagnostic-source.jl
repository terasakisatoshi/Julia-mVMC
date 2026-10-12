using CUDA,MVMCOptimizers,LinearAlgebra,Statistics,SHA,Test
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
Base.include(GPU,"/tmp/cuda_compact_transform.jl")
Base.include(GPU,"/tmp/cuda_compact_phys_runner.jl")
path="/tmp/jl_AH2LF0/input/namelist.def"
checkpoint="/tmp/jl_AH2LF0/fixed/zqp_opt.dat"
function candidate()
    data,rng=GPU.prepared_gpu_data(path;seed=79,opt_para=checkpoint,physcal=true)
    s,seed_rng,initial_pfaffians=GPU.prepared_runner_state(data,320,rng,data.modpara.rnd_seed;needs_derivatives=false)
    GPU.compact_phys_cal_runner!(data;configurations=s.configurations,nsamples=1,warmup=128,interval=128,seed=79,state=s,seed_rng,initial_pfaffians)
end
reference()=CPU.run_gpu_phys_cal_from_namelist(path;opt_para=checkpoint,nsamples=1,walkers=320,device=1,seed=79,warmup=128,interval=128)
println("scope=diagnostic whole private compact runner versus unchanged public full-transform runner, commit694c224")
println("Julia=",VERSION," threads=",Threads.nthreads()," BLAS=",BLAS.get_config())
println("checkpoint_sha256=",bytes2hex(sha256(read(checkpoint))))
println("modpara_sha256=",bytes2hex(sha256(read("/tmp/jl_AH2LF0/input/modpara.def"))))
println("compact_source_sha256=",bytes2hex(sha256(read("/tmp/cuda_compact_transform.jl"))))
fulltimes=Float64[];compacttimes=Float64[]
for rep in 0:3
    results=[]
    for (label,f) in (("full",reference),("compact",candidate))
        GC.gc();CUDA.synchronize();start=time_ns();r=f();CUDA.synchronize()
        t=(time_ns()-start)/1e9
        rep>0 && push!(label=="full" ? fulltimes : compacttimes,t)
        println("rep=",rep," label=",label," seconds=",t," energy=",r.energy," accepted=",sum(Array(r.state.accepted)))
        push!(results,r)
    end
    a,b=results
    @test Array(a.state.configurations)==Array(b.state.configurations)
    @test a.energy≈b.energy atol=1e-9 rtol=1e-9
    @test a.green_one≈b.green_one atol=1e-9 rtol=1e-9
    @test a.green_two≈b.green_two atol=1e-9 rtol=1e-9
end
println("full_median=",median(fulltimes)," compact_median=",median(compacttimes))
