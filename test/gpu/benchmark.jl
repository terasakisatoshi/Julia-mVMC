using Random, LinearAlgebra, CUDA, MVMCOptimizers, MVMCExpertModeParsers
const CPU = MVMCOptimizers
const P = MVMCExpertModeParsers
const GPU = Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
const DEVICE = parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","1"))
CUDA.device!(DEVICE)
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
CUDA.versioninfo()
println("Julia=",VERSION," CPU_BLAS=",BLAS.get_config()," device=",DEVICE," GPU_RNG=cuRAND/XORWOW")
flush(stdout)
path = isempty(ARGS) ? joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def") : ARGS[1]
walkers = parse(Int,get(ENV,"JULIA_MVMC_GPU_WALKERS","256"))
steps = parse(Int,get(ENV,"JULIA_MVMC_GPU_BENCH_STEPS","20"))
data,rng = GPU.prepared_gpu_data(path;seed=79)
cfg = GPU.initial_configurations(data,walkers,rng)
threads = parse(Int,get(ENV,"JULIA_MVMC_GPU_THREADS","32"))
s = gpu_state(data;configurations=cfg,threads)
o = GPU.CUDAObservables(s,data)
sr = GPU.CUDASR(s,data)
c = cpu_state(data)
println("setup_complete nsite=",s.nsite," walkers=",walkers)
flush(stdout)
CUDA.seed!(79)
# Warm up each compilation/library path before measuring.
for _ in 1:2
    gpu_sample!(s;steps,exchange_fraction=GPU.exchange_probability(data))
    gpu_observables!(s,o)
    GPU.derivatives!(s,sr,o)
    GPU.sr_moments(sr.derivatives,copy(vec(view(o.values,1,:))))
    CUDA.synchronize()
end
println("warmup_complete")
flush(stdout)
function times(f; device_time=false)
    values = Float64[]
    for _ in 1:5
        CUDA.synchronize()
        t = if device_time
            CUDA.@elapsed f()
        else
            @elapsed begin f(); CUDA.synchronize() end
        end
        push!(values,t)
    end
    return minimum(values),sort(values)[3],maximum(values)
end
println("input=",path," nsite=",s.nsite," nqp=",s.nqp," walkers=",walkers," threads=",s.threads)
for (name,f) in (
    ("sampling_$(steps)_proposals_per_walker",()->gpu_sample!(s;steps,exchange_fraction=GPU.exchange_probability(data))),
    ("energy_and_requested_greens",()->gpu_observables!(s,o)),
    ("analytic_log_derivatives_cached",()->GPU.derivatives!(s,sr,o)),
    ("SR_moments",()->GPU.sr_moments(sr.derivatives,copy(vec(view(o.values,1,:)))))
)
    println(name," gpu_wall_min_median_max_s=",times(f)," gpu_event_min_median_max_s=",times(f;device_time=true))
    flush(stdout)
end
# Compare identical fixed configurations and observables, not unlike MC paths.
fixed = Int.(Array(s.configurations))
cpu_observables(data,c,fixed[:,1])
println("CPU_energy_greens_derivatives_single_thread_min_median_max_s=",
        times(()->[cpu_observables(data,c,fixed[:,b]) for b in 1:walkers]))
reference = [cpu_observables(data,c,fixed[:,b]) for b in 1:walkers]
expected_values = hcat(first.(reference)...)
expected_derivatives = hcat(last.(reference)...)
println("measured_configuration_CPU_GPU_max_observable_error=",maximum(abs,Array(o.values)-expected_values),
        " max_derivative_error=",maximum(abs,Array(sr.derivatives)-expected_derivatives))
flush(stdout)
all(isfinite,expected_values) && all(isfinite,expected_derivatives) ||
    error("CPU reference failed on a measured configuration")
all(isapprox.(Array(o.values),expected_values;atol=1e-9,rtol=1e-9)) ||
    error("measured GPU observables diverge from CPU reference")
all(isapprox.(Array(sr.derivatives),expected_derivatives;atol=1e-9,rtol=1e-9)) ||
    error("measured GPU derivatives diverge from CPU reference")
println("GPU_energy_greens_derivatives_min_median_max_s=",
        times(()->begin gpu_observables!(s,o); GPU.derivatives!(s,sr,o) end))
if get(ENV,"JULIA_MVMC_GPU_PROFILE","0") == "1"
    println("PROFILE: warmed sampling + observables + derivatives; no host transfers expected")
    profile = CUDA.@profile begin
        gpu_sample!(s;steps,exchange_fraction=GPU.exchange_probability(data))
        gpu_observables!(s,o)
        GPU.derivatives!(s,sr,o)
        CUDA.synchronize()
    end
    show(stdout, profile)
    println()
    # CUDA's profiler allocates/uploads a dummy Int64 before its first context
    # synchronization. Use the same application-range boundaries as its display.
    first_sync = findfirst(==("cuCtxSynchronize"), profile.host.name)
    last_sync = findlast(==("cuCtxSynchronize"), profile.host.name)
    first_id = profile.host.id[first_sync+1]
    last_id = profile.host.id[last_sync-1]
    transfers = [name for (id,name) in zip(profile.device.id,profile.device.name)
                 if first_id <= id <= last_id && startswith(name,"[copy ")]
    println("inner_loop_memory_copy_records=", length(transfers))
    isempty(transfers) || error("unexpected inner-loop memory copies: $transfers")
end
