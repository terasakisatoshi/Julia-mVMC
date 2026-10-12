function baseline_opt_runner!(data::ExpertModeData; configurations, nsteps::Integer,
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

function paired_opt_runner!(data::ExpertModeData; configurations, nsteps::Integer,
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
    sampling_inverse = sampler === nothing ? nothing :
        (2s.nelec >= 128 ? PairedSamplingInverse(s) : BatchedSamplingInverse(s))
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

function gpu_para_opt_runner!(data::ExpertModeData;kwargs...)
    get(ENV,"JULIA_MVMC_PAIRED_AB","candidate")=="baseline" ? baseline_opt_runner!(data;kwargs...) : paired_opt_runner!(data;kwargs...)
end
