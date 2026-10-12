using CUDA,MVMCOptimizers,LinearAlgebra,Statistics,Test,SHA
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
Base.include(GPU,joinpath(@__DIR__,"baseline-ab-dispatch.jl"))
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
const INPUT=abspath(ARGS[1]);const CHECKPOINT=abspath(ARGS[2])
function run(kind,variant,rep)
    ENV["JULIA_MVMC_DELAYED_AB"]=variant
    GC.gc();CUDA.synchronize();start=time_ns()
    result=kind==:opt ? CPU.run_gpu_para_opt_from_namelist(INPUT;nsteps=20,walkers=320,device=1,seed=79,warmup=128,interval=128) : CPU.run_gpu_phys_cal_from_namelist(INPUT;opt_para=CHECKPOINT,nsamples=1,walkers=320,device=1,seed=79,warmup=128,interval=128)
    CUDA.synchronize();elapsed=(time_ns()-start)/1e9
    result.status==0 || error("public runner failed")
    values=kind==:opt ? result.energies : vcat(result.energy,result.green_one,result.green_two,result.green_two_ex)
    all(isfinite,values) || error("nonfinite results")
    receipt=(;cfg=Array(result.state.configurations),occupancy=Array(result.state.occupancy),counts=Array(result.state.counts),accepted=Array(result.state.accepted),values,parameters=kind==:opt ? result.parameters : Float64[],future=Array(CUDA.rand(UInt32,64)))
    println("rep=",rep," kind=",kind," variant=",variant," seconds=",elapsed," energy_per_site=",(kind==:opt ? values[end] : values[1])/128," accepted_last_interval=",sum(receipt.accepted));flush(stdout)
    return elapsed,receipt
end
println("scope=complete public Opt20; FP64 GPU1 RTX3060; Julia_threads=",Threads.nthreads()," BLAS_threads=",BLAS.get_num_threads()," warmup128 interval128 walkers320 seed79; launch/imports and GC outside clocks")
println("Julia=",VERSION," checkpoint_sha256=",bytes2hex(sha256(read(CHECKPOINT)))," namelist_sha256=",bytes2hex(sha256(read(INPUT))))
println("ab_kernel_source_sha256=",bytes2hex(sha256(read(joinpath(@__DIR__,"baseline-ab-dispatch.jl")))))
for name in ("cuda_sampling_update.jl","cuda_runners.jl","cuda_state.jl")
    println("production_source_sha256 ",name," ",bytes2hex(sha256(read(joinpath(dirname(pathof(CPU)),"..","ext",name)))))
end
flush(stdout)
for kind in (:opt,)
    for variant in ("baseline","candidate")
        run(kind,variant,0)
    end
    times=Dict("baseline"=>Float64[],"candidate"=>Float64[])
    @testset "whole $kind delayed combined repeated-control parity" begin
        reference=nothing
        for rep in 1:5
            order=isodd(rep) ? ("baseline","candidate") : ("candidate","baseline")
            for variant in order
                elapsed,receipt=run(kind,variant,rep);push!(times[variant],elapsed)
                if reference===nothing
                    reference=receipt
                else
                    equal_cfg=receipt.cfg==reference.cfg; @test equal_cfg
                    equal_occupancy=receipt.occupancy==reference.occupancy; @test equal_occupancy
                    equal_counts=receipt.counts==reference.counts; @test equal_counts
                    equal_accepted=receipt.accepted==reference.accepted; @test equal_accepted
                    equal_future=receipt.future==reference.future; @test equal_future
                    println("max_abs_energy_diff=",maximum(abs,receipt.values-reference.values)," max_abs_parameter_diff=",maximum(abs,receipt.parameters-reference.parameters));flush(stdout)
                    @test all(isapprox.(receipt.values,reference.values;atol=1e-9,rtol=1e-9))
                    @test all(isapprox.(receipt.parameters,reference.parameters;atol=1e-9,rtol=1e-9))
                end
            end
        end
    end
    println("summary kind=",kind," baseline_seconds=",times["baseline"]," candidate_seconds=",times["candidate"]," baseline_median=",median(times["baseline"])," candidate_median=",median(times["candidate"])," speedup=",median(times["baseline"])/median(times["candidate"]));flush(stdout)
end
