
function compact_phys_cal_runner!(data::ExpertModeData; configurations, nsamples::Integer=1,
                          warmup::Integer=10, interval::Integer=1, seed::Integer=79,
                          callback=nothing, state=nothing, seed_rng::Bool=true, initial_pfaffians=nothing)
    nsamples > 0 && warmup >= 0 && interval >= 0 || throw(ArgumentError("invalid sampling lengths"))
    # Validate operator tables before RNG use or launch.
    operator_tables(data)
    s = state === nothing ? cuda_state_for_runner(data; configurations,needs_derivatives=false) : state
    validate_initial_walkers(s, configurations)
    o = CUDAObservables(s, data)
    compact = CompactTransform(s,data)
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
        compact_observables!(s,o,compact;sampling_cache=observable_cache)
        accumulator .+= vec(sum(o.values; dims=2)) ./ size(o.values,2)
        callback !== nothing && callback(sample, s, o)
    end
    # Single final summary transfer; no saved configurations cross the bus.
    means = Array(accumulator) ./ nsamples
    return (; status=0, energy=means[1], green_one=means[2:1+length(o.one)],
            green_two=means[2+length(o.one):1+length(o.one)+length(o.two)],
            green_two_ex=means[2+length(o.one)+length(o.two):end], state=s)
end
