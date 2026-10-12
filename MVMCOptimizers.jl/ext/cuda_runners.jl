# CPU setup follows existing Julia lifecycle; all Monte Carlo and per-sample
# numerical work after setup executes on the selected CUDA device.
function initial_configurations(data, walkers, rng)
    ns = data.modpara.nsite
    ne = data.modpara.nelec
    np = CPU.count_total_parameters(data)
    nproj = Parsers.projection_layout(data).n_proj
    nq = CPU.get_n_qp_full(data)
    cpu = CPU.VMCOptimizationState(ns, ne, nproj, np, nq, 1, false, false)
    CPU.update_slater_elm_fcmp!(data, cpu)
    cpu.slater_matrix.slater_elm_real .= real.(cpu.slater_matrix.slater_elm)
    cfg = Matrix{Int32}(undef, 2ne, walkers)
    for b in 1:walkers
        e = cpu.electron_config
        info = CPU.make_initial_sample!(e.tmp_ele_idx, e.tmp_ele_cfg, e.tmp_ele_num,
                                       e.tmp_ele_proj_cnt, data, cpu, rng)
        info == 0 || error("initial configuration failed (status $info)")
        cfg[:,b] .= e.tmp_ele_idx
    end
    return cfg
end

function prepared_gpu_data(namelist; seed=nothing, initial_def=:auto, opt_para=nothing, physcal=false)
    data = Parsers.parse_expert_mode_files(String(namelist))
    validate_cuda_data(data)
    operator_tables(data)
    physcal || data.modpara.nsrcg == 0 ||
        throw(ArgumentError("CUDA optimization requires NSRCG=0"))
    data.modpara.nmp_trans = max(1, abs(data.modpara.nmp_trans))
    if data.modpara.nelec == 0
        data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) ÷ 2
    end
    rng = CPU.SFMT19937RNG()
    actual_seed = CPU.resolve_rnd_seed(CPU.serial_context(), data.modpara.rnd_seed, seed)
    data.modpara.rnd_seed = actual_seed
    CPU.Random.seed!(rng, actual_seed)
    Parsers.init_parameter!(data; rng)
    initial = initial_def === :auto ? joinpath(dirname(namelist), "initial.def") : initial_def
    if initial !== nothing && initial !== :none && (initial_def !== :auto || isfile(initial))
        CPU.read_initial_def!(data, String(initial)) || error("could not read initial parameters")
    end
    Parsers.read_input_parameters!(data, String(namelist))
    if opt_para !== nothing
        CPU.read_opt_para_file!(data, String(opt_para)) == CPU.count_total_parameters(data) ||
            error("incomplete optimized parameter file")
    end
    CPU.sync_modified_parameter!(data; shift_correlations=!physcal)
    Parsers.init_qp_weight!(data)
    return data, rng
end

function exchange_probability(data)
    path = data.modpara.nex_update_path
    path == 0 && return 0.0
    path == 1 && return 0.5
    path == 2 && return 1.0
    throw(ArgumentError("CUDA requires NExUpdatePath in 0:2"))
end

function validate_initial_walkers(s::CUDAState, configurations)
    # A single setup-time scalar check, before seeding or sampling.
    all(x -> isfinite(x) && x != 0, s.overlaps) ||
        throw(ArgumentError("initial walkers must have finite nonzero overlap"))
    all(isfinite, s.logcorrelations) || throw(ArgumentError("initial correlation factors must be finite"))
    loc = Array(s.localspins)
    any(==(1),loc) || return nothing
    configurations = configurations isa CuArray ? Array(configurations) : configurations
    for b in axes(configurations,2), site in eachindex(loc)
        if loc[site] == 1
            count(==(site-1), view(configurations,:,b)) == 1 ||
                throw(ArgumentError("local-spin sites must be singly occupied"))
        end
    end
    return nothing
end

function runner_sampling_cache(s::CUDAState)
    mode = get(ENV,"JULIA_MVMC_GPU_SAMPLING","update")
    mode in ("update","refactor") || throw(ArgumentError("GPU sampling must be update or refactor"))
    return s.nsite >= 64 && mode == "update" ? GPUSamplingCache(s) : nothing
end

function runner_sampling_options(cache)
    proposal = get(ENV,"JULIA_MVMC_GPU_PROPOSALS","empty")
    correlation = get(ENV,"JULIA_MVMC_GPU_CORRELATION","delta")
    proposal in ("empty","self-loop") || throw(ArgumentError("GPU proposals must be empty or self-loop"))
    correlation in ("delta","full") || throw(ArgumentError("GPU correlation must be delta or full"))
    return (;empty_destinations=cache !== nothing && proposal == "empty",
        incremental_correlation=cache !== nothing && correlation == "delta")
end

function runner_observable_cache(cache)
    mode=get(ENV,"JULIA_MVMC_GPU_OBSERVABLES","reuse")
    mode in ("reuse","refactor") || throw(ArgumentError("GPU observables must be reuse or refactor"))
    return mode == "reuse" ? cache : nothing
end

function prepared_runner_state(data,walkers,rng,seed;needs_derivatives::Bool=true)
    mode = get(ENV,"JULIA_MVMC_GPU_INITIALIZATION",data.modpara.nsite >= 64 ? "gpu" : "cpu")
    mode in ("gpu","cpu") || throw(ArgumentError("GPU initialization must be gpu or cpu"))
    if mode == "gpu"
        s,pf=initial_gpu_state(data,walkers;seed,return_pfaffians=true,needs_derivatives)
        return s,false,pf
    end
    cfg = initial_configurations(data,walkers,rng)
    return cuda_state_for_runner(data;configurations=cfg,needs_derivatives),true,nothing
end

function runner_sampling_lengths(data,walkers,warmup,interval)
    m=data.modpara
    batch=walkers === nothing ? m.nvmc_sample : Int(walkers)
    spacing=interval === nothing ? m.nvmc_interval*m.nsite : Int(interval)
    thermal=warmup === nothing ? m.nvmc_warmup*m.nvmc_interval*m.nsite : Int(warmup)
    batch > 0 && spacing >= 0 && thermal >= 0 || throw(ArgumentError("invalid sampling lengths"))
    return batch,thermal,spacing
end

function gpu_phys_cal_runner!(data::ExpertModeData; configurations, nsamples::Integer=1,
                          warmup::Integer=10, interval::Integer=1, seed::Integer=79,
                          callback=nothing, state=nothing, seed_rng::Bool=true, initial_pfaffians=nothing)
    nsamples > 0 && warmup >= 0 && interval >= 0 || throw(ArgumentError("invalid sampling lengths"))
    # Validate operator tables before RNG use or launch.
    operator_tables(data)
    s = state === nothing ? cuda_state_for_runner(data; configurations,needs_derivatives=false) : state
    validate_initial_walkers(s, configurations)
    o = CUDAObservables(s, data)
    sampler = runner_sampling_cache(s)
    sampling_inverse = sampler === nothing ? nothing : BatchedSamplingInverse(s)
    sampling_options = runner_sampling_options(sampler)
    observable_cache = o.batched_inverse === nothing ? nothing : runner_observable_cache(sampler)
    seed_rng && CUDA.seed!(seed)
    ex = exchange_probability(data)
    sampler !== nothing && refresh_sampling_state!(s,sampler,sampling_inverse;initial_pfaffians)
    CPU.gpu_sample!(s; steps=warmup, exchange_fraction=ex, cache=sampler, sampling_options...)
    accumulator = CUDA.zeros(Float64, size(o.values,1))
    for sample in 1:nsamples
        CPU.gpu_sample!(s; steps=interval, exchange_fraction=ex, cache=sampler, sampling_options...)
        CPU.gpu_observables!(s, o;sampling_cache=observable_cache)
        accumulator .+= vec(sum(o.values; dims=2)) ./ size(o.values,2)
        callback !== nothing && callback(sample, s, o)
    end
    # Single final summary transfer; no saved configurations cross the bus.
    means = Array(accumulator) ./ nsamples
    return (; status=0, energy=means[1], green_one=means[2:1+length(o.one)],
            green_two=means[2+length(o.one):1+length(o.one)+length(o.two)],
            green_two_ex=means[2+length(o.one)+length(o.two):end], state=s)
end

function gpu_para_opt_runner!(data::ExpertModeData; configurations, nsteps::Integer,
                          warmup::Integer=10, interval::Integer=1, seed::Integer=79,
                          callback=nothing, state=nothing, seed_rng::Bool=true, initial_pfaffians=nothing)
    nsteps > 0 && warmup >= 0 && interval >= 0 || throw(ArgumentError("invalid optimization lengths"))
    data.modpara.nsrcg == 0 || throw(ArgumentError("CUDA currently uses direct SR (NSRCG=0)"))
    operator_tables(data)
    s = state === nothing ? CPU.gpu_state(data; configurations) : state
    validate_initial_walkers(s, configurations)
    o = CUDAObservables(s, data; measure_greens=false)
    sr = CUDASR(s, data)
    solver = CUDASRSolver(length(s.parameters))
    sampler = runner_sampling_cache(s)
    gauge = CUDAParameterGauge(data)
    sampling_inverse = sampler === nothing ? nothing : BatchedSamplingInverse(s)
    sampling_options = runner_sampling_options(sampler)
    observable_cache = o.batched_inverse === nothing ? nothing : runner_observable_cache(sampler)
    seed_rng && CUDA.seed!(seed)
    ex = exchange_probability(data)
    sampler !== nothing && refresh_sampling_state!(s,sampler,sampling_inverse;initial_pfaffians)
    CPU.gpu_sample!(s; steps=warmup, exchange_fraction=ex, cache=sampler, sampling_options...)
    energy_history = CUDA.zeros(Float64,nsteps)
    for step in 1:nsteps
        CPU.gpu_sample!(s; steps=interval, exchange_fraction=ex, cache=sampler, sampling_options...)
        CPU.gpu_observables!(s, o;sampling_cache=observable_cache)
        derivatives!(s, sr, o)
        e = copy(vec(view(o.values,1,:)))
        moments = sr_moments(sr, e)
        resident_sr_update!(s,sr,moments,data.modpara,solver)
        sync_gpu_parameters!(s,gauge)
        refresh_slater!(s)
        if sampler === nothing
            CPU.gpu_wavefunctions!(s)
        else
            refresh_sampling_state!(s,sampler,sampling_inverse)
        end
        view(energy_history,step:step) .= moments.meanE
        callback !== nothing && callback(step, s, moments)
    end
    check_sr_solver(solver)
    parameters = Array(s.parameters)
    CPU.unpack_parameters!(data,ComplexF64.(parameters))
    return (; status=0, energies=Array(energy_history), parameters, state=s)
end

# Prepared states are internal to the namelist runners. Public low-level calls
# always construct and validate a state against their supplied model.
function CPU.gpu_phys_cal!(data::ExpertModeData; configurations, nsamples::Integer=1,
        warmup::Integer=10, interval::Integer=1, seed::Integer=79, callback=nothing)
    return gpu_phys_cal_runner!(data; configurations, nsamples, warmup, interval, seed, callback)
end

function CPU.gpu_para_opt!(data::ExpertModeData; configurations, nsteps::Integer,
        warmup::Integer=10, interval::Integer=1, seed::Integer=79, callback=nothing)
    return gpu_para_opt_runner!(data; configurations, nsteps, warmup, interval, seed, callback)
end

function CPU.run_gpu_para_opt_from_namelist(namelist::AbstractString;
        nsteps::Integer, walkers::Union{Integer,Nothing}=nothing, device::Integer=0,
        seed::Union{Integer,Nothing}=nothing, initial_def=:auto,
        warmup::Union{Integer,Nothing}=nothing, interval::Union{Integer,Nothing}=nothing, callback=nothing)
    CUDA.device!(device)
    data, rng = prepared_gpu_data(namelist; seed, initial_def)
    walkers,warmup,interval=runner_sampling_lengths(data,walkers,warmup,interval)
    s,seed_rng,initial_pfaffians = prepared_runner_state(data,walkers,rng,data.modpara.rnd_seed)
    result = gpu_para_opt_runner!(data; configurations=s.configurations, nsteps, warmup, interval,
                               seed=data.modpara.rnd_seed, callback, state=s, seed_rng, initial_pfaffians)
    return merge(result, (; data))
end

function CPU.run_gpu_phys_cal_from_namelist(namelist::AbstractString;
        opt_para=nothing, nsamples::Union{Integer,Nothing}=nothing, walkers::Union{Integer,Nothing}=nothing,
        device::Integer=0, seed::Union{Integer,Nothing}=nothing, initial_def=:auto,
        warmup::Union{Integer,Nothing}=nothing, interval::Union{Integer,Nothing}=nothing,
        callback=nothing)
    CUDA.device!(device)
    data, rng = prepared_gpu_data(namelist; seed, initial_def, opt_para, physcal=true)
    walkers,warmup,interval=runner_sampling_lengths(data,walkers,warmup,interval)
    nsamples=nsamples === nothing ? data.modpara.n_data_qty_smp : Int(nsamples)
    s,seed_rng,initial_pfaffians = prepared_runner_state(data,walkers,rng,data.modpara.rnd_seed;needs_derivatives=false)
    result = gpu_phys_cal_runner!(data; configurations=s.configurations, nsamples, warmup, interval,
                               seed=data.modpara.rnd_seed, callback, state=s, seed_rng, initial_pfaffians)
    return merge(result, (; data))
end
