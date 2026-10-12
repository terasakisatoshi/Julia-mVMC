# Whole public-runner timings. This is an optional developer benchmark, not a
# deterministic CPU/GPU trajectory test: production RNG/proposals differ.
const BENCHMARK_SCRIPT_START_NS = time_ns()
using CUDA, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra, Statistics, SHA
const CPU = MVMCOptimizers
const GPU = Base.get_extension(CPU, :MVMCCUDAExt)

positive_env(name, fallback) = parse(Int, get(ENV, name, string(fallback)))
function override_modpara!(path, settings)
    text = read(path, String)
    for (key, value) in settings
        pattern = Regex("(?m)^\\s*" * key * "\\s+[^\\r\\n]*")
        line = key * "  " * string(value)
        text = occursin(pattern, text) ? replace(text, pattern => line) : text * "\n" * line * "\n"
    end
    write(path, text)
end

function fixed_parameter_checkpoint(namelist, directory, seed)
    data, _ = GPU.prepared_gpu_data(namelist; seed, physcal=true)
    # The public writer's two-point window produces the triple-column format
    # accepted by the fixed-parameter reader. Identical snapshots preserve the
    # initialized parameters and yield zero deviations.
    data.modpara.nsr_opt_itr_smp = 2
    state = CPU.VMCOptimizationState(data.modpara.nsite, data.modpara.nelec,
        MVMCExpertModeParsers.projection_layout(data).n_proj, CPU.count_total_parameters(data),
        CPU.get_n_qp_full(data), 1, false, false)
    push!(state.opt_data, CPU.OptDataPoint(0.0 + 0im, 0.0 + 0im, CPU.pack_parameters(data)))
    push!(state.opt_data, CPU.OptDataPoint(0.0 + 0im, 0.0 + 0im, CPU.pack_parameters(data)))
    CPU.output_opt_data!(data, state; output_dir=directory)
    head = isempty(data.modpara.c_para_file_head) ? "zqp" : data.modpara.c_para_file_head
    checkpoint = joinpath(directory, head * "_opt.dat")
    loaded = deepcopy(data)
    CPU.read_opt_para_file!(loaded, checkpoint) == CPU.count_total_parameters(data) ||
        error("fixed-parameter checkpoint is incomplete")
    isapprox(CPU.pack_parameters(loaded), CPU.pack_parameters(data); atol=1e-14, rtol=1e-14) ||
        error("fixed-parameter checkpoint changed initialized parameters")
    return checkpoint
end

function timed_runner(f; gpu=false)
    GC.gc()
    gpu && CUDA.synchronize()
    start = time_ns()
    result = f()
    gpu && CUDA.synchronize()
    seconds = (time_ns()-start)/1e9
    result.status == 0 || error("runner returned status $(result.status)")
    return seconds, result
end

function benchmark_end_to_end()
    original = isempty(ARGS) ? joinpath(@__DIR__, "..", "..", "benchmark", "cuda", "inputs",
                                     "hubbard_chain_L128", "namelist.def") : ARGS[1]
    original = abspath(original)
    steps = positive_env("JULIA_MVMC_E2E_STEPS", 2)
    samples = positive_env("JULIA_MVMC_E2E_SAMPLES", 32)
    warmup = positive_env("JULIA_MVMC_E2E_WARMUP", 1)
    interval = positive_env("JULIA_MVMC_E2E_INTERVAL", 1)
    repetitions = positive_env("JULIA_MVMC_E2E_REPS", 1)
    phys_runs = positive_env("JULIA_MVMC_E2E_PHYSCAL_RUNS", 1)
    device = positive_env("JULIA_MVMC_GPU_DEVICE", 1)
    seed = positive_env("JULIA_MVMC_E2E_SEED", 79)
    backend = get(ENV, "JULIA_MVMC_E2E_BACKEND", "both")
    backend in ("both", "cpu", "gpu") || error("backend must be both, cpu, or gpu")
    steps > 0 && samples > 0 && repetitions > 0 && phys_runs > 0 && warmup >= 0 && interval > 0 ||
        throw(ArgumentError("steps/samples/reps/PhysCal runs/interval must be positive; warmup nonnegative"))
    Threads.nthreads() == 1 || error("run with JULIA_NUM_THREADS=1")
    BLAS.set_num_threads(1)
    if backend != "cpu"
        CUDA.device!(device)
        CUDA.allowscalar(false)
    end

    root = mktempdir(; cleanup=false)
    input = joinpath(root, "input")
    cp(dirname(original), input)
    namelist = joinpath(input, basename(original))
    modpara_file = nothing
    for line in eachline(namelist)
        fields = split(line)
        if length(fields) >= 2 && lowercase(fields[1]) == "modpara"
            modpara_file = fields[2]
            break
        end
    end
    modpara_file === nothing && error("namelist has no ModPara entry")
    modpara_path = normpath(joinpath(input, modpara_file))
    relative_modpara = relpath(modpara_path, input)
    (isabspath(modpara_file) || relative_modpara == ".." || startswith(relative_modpara, "../")) &&
        error("benchmark requires a relative ModPara path inside the cloned input directory")
    islink(modpara_path) && error("benchmark refuses to overwrite a symlinked ModPara file")
    override_modpara!(modpara_path, [
        "NVMCSample"=>samples, "NVMCWarmUp"=>warmup, "NVMCInterval"=>interval,
        "NSROptItrStep"=>steps, "NSROptItrSmp"=>1, "NDataQtySmp"=>phys_runs,
        "NSRCG"=>0, "NSplitSize"=>1, "RndSeed"=>seed])
    parsed = MVMCExpertModeParsers.parse_expert_mode_files(namelist)
    ns = parsed.modpara.nsite
    checkpoint = fixed_parameter_checkpoint(namelist, joinpath(root,"fixed"), seed)
    warmup_proposals = warmup*interval*ns
    interval_proposals = interval*ns
    cpu_opt_attempts = ((warmup+samples)+(steps-1)*(samples+1))*interval*ns
    gpu_opt_attempts = samples*(warmup_proposals+steps*interval_proposals)
    cpu_phys_attempts = ((warmup+samples)+(phys_runs-1)*(samples+1))*interval*ns
    gpu_phys_attempts = samples*(warmup_proposals+phys_runs*interval_proposals)
    println("benchmark_scope=whole public runner, including parsing, initialization, sampling, numerical work, SR where applicable, and final returned summary")
    println("Julia=",VERSION," Julia_threads=",Threads.nthreads()," CPU_BLAS=",BLAS.get_config())
    if backend != "cpu"
        println("GPU_device=",device," GPU_name=",CUDA.name(CUDA.device()))
    else
        println("GPU_execution=disabled_for_CPU_only_baseline")
    end
    println("original_input=",original," retained_run_directory=",root)
    println("backend=$backend nsite=$ns optimization_steps=$steps samples_per_step=$samples PhysCal_runs=$phys_runs warmup_sweeps=$warmup interval_sweeps=$interval seed=$seed repetitions=$repetitions")
    println("fixed_parameter_sha256=",bytes2hex(sha256(read(checkpoint))))
    println("benchmark_source_sha256=",bytes2hex(sha256(read(@__FILE__))))
    println("namelist_sha256=",bytes2hex(sha256(read(namelist))),
            " modpara_sha256=",bytes2hex(sha256(read(modpara_path))))
    source_root = joinpath(@__DIR__,"..","..","MVMCOptimizers.jl","src")
    for source in ("vmc_sampling.jl","vmc_para_opt.jl","vmc_phys_cal.jl")
        println("CPU_source_sha256 ",source," ",bytes2hex(sha256(read(joinpath(source_root,source)))))
    end
    println("script_loading_compilation_and_untimed_benchmark_preparation_seconds=",
            (time_ns()-BENCHMARK_SCRIPT_START_NS)/1e9)
    println("CPU_opt_attempt_budget=$cpu_opt_attempts GPU_opt_attempt_budget=$gpu_opt_attempts CPU_PhysCal_attempt_budget=$cpu_phys_attempts GPU_PhysCal_attempt_budget=$gpu_phys_attempts")
    println("sampling_difference=CPU one sequential SFMT chain with draw-until-empty destinations and burn-state reuse; GPU $samples independent XORWOW chains with one initial warmup and $interval_proposals attempts per interval; see selected proposal-mode metadata below")
    println("GPU_sampling_mode=",get(ENV,"JULIA_MVMC_GPU_SAMPLING","update"),
            " GPU_initialization_mode=",get(ENV,"JULIA_MVMC_GPU_INITIALIZATION",ns >= 64 ? "gpu" : "cpu"))
    println("GPU_configured_proposals=",get(ENV,"JULIA_MVMC_GPU_PROPOSALS","empty"),
            " GPU_configured_correlation=",get(ENV,"JULIA_MVMC_GPU_CORRELATION","delta"),
            " GPU_configured_observables=",get(ENV,"JULIA_MVMC_GPU_OBSERVABLES","reuse"),
            " GPU_configured_inverse=",get(ENV,"JULIA_MVMC_GPU_INVERSE","batched"))
    println("GPU_proposal_mode_description=",get(ENV,"JULIA_MVMC_E2E_PROPOSAL_DESCRIPTION",
            "production default; inspect the recorded CUDA sampling source for proposal semantics"))
    gpu_source_root = joinpath(@__DIR__,"..","..","MVMCOptimizers.jl","ext")
    for source in ("cuda_runners.jl","cuda_state.jl","cuda_sampling_update.jl")
        println("GPU_source_sha256 ",source," ",bytes2hex(sha256(read(joinpath(gpu_source_root,source)))))
    end
    println("proposal_counts_are_not_valid_move_counts_or_effective_sample_sizes=true")
    println("CPU_valid_and_accepted_counts=not_exposed_by_public_runner GPU_valid_counts=not_exposed GPU_accepted_count=last_interval_only_after_timing")
    ne = parsed.modpara.nelec == 0 ?
        (parsed.modpara.nlocspin+parsed.modpara.ncond)÷2 : parsed.modpara.nelec
    if parsed.modpara.nex_update_path == 0 && 0 < ne < ns &&
       all(==(0),CPU.get_loc_spn_array(parsed)) && ns >= 64 &&
       get(ENV,"JULIA_MVMC_GPU_SAMPLING","update") == "update" &&
       get(ENV,"JULIA_MVMC_GPU_PROPOSALS","empty") == "empty"
        println("derived_valid_hopping_candidate_budgets=all configured attempts have a legal empty destination for this nonlocal-spin input; these are model/proposal deductions, not measured counters")
    end
    println("output_difference=CPU writes C-format per-step/final files; GPU returns numerical arrays and performs final summaries/transfers without those files")
    println("cold_first_execution_includes_compilation_and_real_work=true; reported separately, not used in warm timing ratio")
    flush(stdout)

    last_gpu_opt = nothing
    last_gpu_phys = nothing
    for phase in ("cold", "warm")
        count = phase == "cold" ? 1 : repetitions
        cpu_opt_times = Float64[]; gpu_opt_times = Float64[]
        cpu_phys_times = Float64[]; gpu_phys_times = Float64[]
        for rep in 1:count
            output = joinpath(root,phase*"-"*string(rep))
            if backend != "gpu"
                t, cpu_opt = timed_runner(() -> CPU.run_para_opt_from_namelist(namelist;
                    nsteps=steps, nsmp=1, mode=:real, seed,
                    output_dir=joinpath(output,"cpu-opt")))
                push!(cpu_opt_times,t)
                isfinite(cpu_opt.final_energy_per_site) || error("CPU optimization returned nonfinite energy")
                println("phase=$phase rep=$rep CPU_optimization_seconds=$t energy_per_site=",cpu_opt.final_energy_per_site)
                flush(stdout)
            end
            if backend != "cpu"
                t, last_gpu_opt = timed_runner(() -> CPU.run_gpu_para_opt_from_namelist(namelist;
                    nsteps=steps, walkers=samples, device, seed,
                    warmup=warmup_proposals, interval=interval_proposals);gpu=true)
                push!(gpu_opt_times,t)
                all(isfinite,last_gpu_opt.energies) && all(isfinite,last_gpu_opt.parameters) ||
                    error("GPU optimization returned nonfinite results")
                println("phase=$phase rep=$rep GPU_optimization_seconds=$t energy_per_site=",last(last_gpu_opt.energies)/ns,
                        " accepted_last_interval=",sum(Array(last_gpu_opt.state.accepted)))
                flush(stdout)
            end
            if backend != "gpu"
                t, cpu_phys = timed_runner(() -> CPU.run_phys_cal_from_namelist(namelist;
                    opt_para=checkpoint,mode=:real,seed,
                    output_dir=joinpath(output,"cpu-physcal")))
                push!(cpu_phys_times,t)
                head = isempty(parsed.modpara.c_data_file_head) ? "zvo" : parsed.modpara.c_data_file_head
                index = lpad(parsed.modpara.n_data_idx_start,3,'0')
                energy_file = joinpath(cpu_phys.output_dir,head*"_out_"*index*".dat")
                cpu_energy = parse(Float64,first(split(read(energy_file,String))))/ns
                isfinite(cpu_energy) || error("CPU PhysCal wrote nonfinite energy")
                println("phase=$phase rep=$rep CPU_PhysCal_seconds=$t energy_per_site=$cpu_energy output_dir=",cpu_phys.output_dir)
                flush(stdout)
            end
            if backend != "cpu"
                t, last_gpu_phys = timed_runner(() -> CPU.run_gpu_phys_cal_from_namelist(namelist;
                    opt_para=checkpoint,nsamples=phys_runs,walkers=samples,device,seed,
                    warmup=warmup_proposals,interval=interval_proposals);gpu=true)
                push!(gpu_phys_times,t)
                isfinite(last_gpu_phys.energy) && all(isfinite,last_gpu_phys.green_one) &&
                    all(isfinite,last_gpu_phys.green_two) && all(isfinite,last_gpu_phys.green_two_ex) ||
                    error("GPU PhysCal returned nonfinite results")
                println("phase=$phase rep=$rep GPU_PhysCal_seconds=$t energy_per_site=",last_gpu_phys.energy/ns,
                        " accepted_last_interval=",sum(Array(last_gpu_phys.state.accepted)))
                flush(stdout)
            end
        end
        println("phase=$phase CPU_opt_seconds=",cpu_opt_times," GPU_opt_seconds=",gpu_opt_times)
        println("phase=$phase CPU_PhysCal_seconds=",cpu_phys_times," GPU_PhysCal_seconds=",gpu_phys_times)
        if phase == "warm" && backend == "both"
            println("whole_runner_elapsed_ratio_opt_CPU_over_GPU=",median(cpu_opt_times)/median(gpu_opt_times))
            println("whole_runner_elapsed_ratio_PhysCal_CPU_over_GPU=",median(cpu_phys_times)/median(gpu_phys_times))
            println("ratios_are_observed_lifecycle_timings_with_the_sampling_and_output_differences_above_not_equivalent_trajectory_or_effective_sample_speedups=true")
        end
        flush(stdout)
    end
    return nothing
end
benchmark_end_to_end()
