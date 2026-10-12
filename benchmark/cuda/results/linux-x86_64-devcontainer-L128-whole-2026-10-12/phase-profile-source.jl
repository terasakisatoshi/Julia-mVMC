using CUDA,MVMCOptimizers,LinearAlgebra,Statistics,SHA
const CPU=MVMCOptimizers
const GPU=Base.get_extension(CPU,:MVMCCUDAExt)
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
const INPUT="/tmp/jl_emoPro/input/namelist.def"
const CHECKPOINT="/tmp/jl_emoPro/fixed/zqp_opt.dat"
function phase!(f,t,key)
    start=time_ns();result=f();CUDA.synchronize()
    t[key]=get(t,key,0.0)+(time_ns()-start)/1e9
    return result
end
function profiled(kind)
    GC.gc();CUDA.synchronize();start=time_ns();t=Dict{Symbol,Float64}()
    isphys=kind==:phys
    data,rng=phase!(t,:parse_parameter_setup) do
        GPU.prepared_gpu_data(INPUT;seed=79,physcal=isphys,opt_para=isphys ? CHECKPOINT : nothing)
    end
    s,seed_rng,pf=phase!(t,:initial_walkers) do
        GPU.prepared_runner_state(data,320,rng,79)
    end
    phase!(t,:runner_validation) do
        GPU.operator_tables(data);GPU.validate_initial_walkers(s,s.configurations)
    end
    o=phase!(t,:observable_workspace) do
        GPU.CUDAObservables(s,data;measure_greens=isphys)
    end
    sampler,work,options,obs_cache=phase!(t,:sampling_workspace) do
        sampler=GPU.runner_sampling_cache(s)
        (sampler,GPU.BatchedSamplingInverse(s),GPU.runner_sampling_options(sampler),GPU.runner_observable_cache(sampler))
    end
    sr,solver,gauge=if isphys
        (nothing,nothing,nothing)
    else
        phase!(t,:sr_workspace) do
            (GPU.CUDASR(s,data),GPU.CUDASRSolver(length(s.parameters)),GPU.CUDAParameterGauge(data))
        end
    end
    phase!(t,:initial_inverse_refresh) do
        GPU.refresh_sampling_state!(s,sampler,work;initial_pfaffians=pf)
    end
    ex=GPU.exchange_probability(data)
    phase!(t,:initial_warmup) do
        CPU.gpu_sample!(s;steps=128,exchange_fraction=ex,cache=sampler,options...)
    end
    output=phase!(t,:history_or_accumulator) do
        CUDA.zeros(Float64,isphys ? size(o.values,1) : 20)
    end
    for step=1:(isphys ? 1 : 20)
        phase!(t,:sampling) do
            CPU.gpu_sample!(s;steps=128,exchange_fraction=ex,cache=sampler,options...)
        end
        phase!(t,:observables) do
            CPU.gpu_observables!(s,o;sampling_cache=obs_cache)
        end
        if isphys
            phase!(t,:phys_accumulate) do
                output .+= vec(sum(o.values;dims=2))./320
            end
        else
            phase!(t,:derivatives) do
                GPU.derivatives!(s,sr,o)
            end
            moments=phase!(t,:moments) do
                GPU.sr_moments(sr,copy(vec(view(o.values,1,:))))
            end
            phase!(t,:native_sr) do
                GPU.resident_sr_update!(s,sr,moments,data.modpara,solver)
            end
            phase!(t,:parameter_gauge) do
                GPU.sync_gpu_parameters!(s,gauge)
            end
            phase!(t,:slater_refresh) do
                GPU.refresh_slater!(s)
            end
            phase!(t,:inverse_refresh) do
                GPU.refresh_sampling_state!(s,sampler,work)
            end
            phase!(t,:energy_history_update) do
                view(output,step:step).=moments.meanE
            end
        end
    end
    summary=phase!(t,:final_summary) do
        if isphys
            means=Array(output)
            (energy=means[1],finite=all(isfinite,means))
        else
            GPU.check_sr_solver(solver)
            parameters=Array(s.parameters)
            CPU.unpack_parameters!(data,ComplexF64.(parameters))
            energies=Array(output)
            (energy=energies[end],finite=all(isfinite,parameters)&&all(isfinite,energies))
        end
    end
    summary.finite || error("nonfinite results")
    wall=(time_ns()-start)/1e9
    return (;wall,t,summary)
end
function whole(kind)
    GC.gc();CUDA.synchronize();start=time_ns()
    result=kind==:opt ? CPU.run_gpu_para_opt_from_namelist(INPUT;nsteps=20,walkers=320,device=1,seed=79,warmup=128,interval=128) : CPU.run_gpu_phys_cal_from_namelist(INPUT;opt_para=CHECKPOINT,nsamples=1,walkers=320,device=1,seed=79,warmup=128,interval=128)
    CUDA.synchronize();wall=(time_ns()-start)/1e9
    result.status==0||error("runner failed")
    return wall
end
println("scope=DevContainer_phase_profile B320 nsteps20 warmup128 interval128 GPU1 CPU_affinity0,18 FP64; per-phase synchronize adds instrumentation overhead")
println("checkpoint_sha256=",bytes2hex(sha256(read(CHECKPOINT))))
for filename in ("cuda_runners.jl","cuda_sr.jl","cuda_sampling_update.jl")
    println("source_sha256 ",filename," ",bytes2hex(sha256(read(joinpath(dirname(pathof(CPU)),"..","ext",filename)))))
end
flush(stdout)
for kind in (:opt,:phys)
    cold=profiled(kind)
    println("cold_instrumented kind=",kind," wall=",cold.wall," summary=",cold.summary);flush(stdout)
    runs=[];base=[]
    for rep=1:3
        run=profiled(kind);push!(runs,run)
        println("instrumented kind=",kind," rep=",rep," wall=",run.wall," phase_sum=",sum(values(run.t))," summary=",run.summary)
        for key in sort!(collect(keys(run.t));by=string)
            println("phase kind=",kind," rep=",rep," name=",key," seconds=",run.t[key])
        end
        flush(stdout)
        elapsed=whole(kind);push!(base,elapsed)
        println("uninstrumented kind=",kind," rep=",rep," wall=",elapsed);flush(stdout)
    end
    println("median kind=",kind," instrumented=",median(getproperty.(runs,:wall))," uninstrumented=",median(base)," overhead=",median(getproperty.(runs,:wall))-median(base))
    for key in sort!(collect(keys(runs[1].t));by=string)
        println("phase_median kind=",kind," name=",key," seconds=",median([r.t[key] for r in runs]))
    end
    flush(stdout)
end
