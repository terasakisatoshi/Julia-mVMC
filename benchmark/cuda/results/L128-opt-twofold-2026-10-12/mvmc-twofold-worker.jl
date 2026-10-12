using CUDA, MVMCOptimizers, LinearAlgebra, Serialization, SHA
CUDA.device!(1)
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
const INPUT = abspath(ARGS[1])
const CHECKPOINT = abspath(ARGS[2])
const OUTPUT = abspath(ARGS[3])
const VARIANT = ARGS[4]
mkpath(OUTPUT)
println("variant=",VARIANT," Julia=",VERSION," Julia_threads=",Threads.nthreads(),
        " BLAS_threads=",BLAS.get_num_threads()," BLAS=",BLAS.get_config(),
        " GPU=",CUDA.name(CUDA.device()))
println("optimizer_source=",pathof(MVMCOptimizers))
for path in (INPUT,CHECKPOINT,joinpath(dirname(INPUT),"modpara.def"))
    println("input_sha256 ",bytes2hex(sha256(read(path)))," ",path)
end
function run(rep)
    GC.gc()
    CUDA.synchronize()
    start=time_ns()
    result=run_gpu_para_opt_from_namelist(INPUT;nsteps=20,walkers=320,device=1,
                                         seed=79,warmup=128,interval=128)
    CUDA.synchronize()
    elapsed=(time_ns()-start)/1e9
    result.status==0 || error("public runner failed")
    all(isfinite,result.energies) && all(isfinite,result.parameters) || error("nonfinite result")
    receipt=(;seconds=elapsed,cfg=Array(result.state.configurations),
        occupancy=Array(result.state.occupancy),counts=Array(result.state.counts),
        accepted=Array(result.state.accepted),values=result.energies,
        parameters=result.parameters,future=Array(CUDA.rand(UInt32,64)))
    serialize(joinpath(OUTPUT,"$(VARIANT)-$(rep).bin"),receipt)
    println("rep=",rep," variant=",VARIANT," seconds=",elapsed,
            " energy_per_site=",result.energies[end]/128,
            " accepted_last_interval=",sum(receipt.accepted))
    flush(stdout)
end
println("READY");flush(stdout)
for line in eachline(stdin)
    line=="quit" && break
    rep=parse(Int,line)
    run(rep)
    # Release live arrays outside the clock. The other process stays idle;
    # each checkout keeps its own warmed CUDA allocation pool and JIT cache.
    GC.gc();CUDA.synchronize()
    println("DONE ",rep);flush(stdout)
end
