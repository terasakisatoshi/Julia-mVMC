# Two devices evaluate disjoint halves of the same fixed configuration batch.
# Run with julia --threads=2; initialization and final result copies are excluded.
using Random, LinearAlgebra, CUDA, MVMCOptimizers, MVMCExpertModeParsers, Serialization
const CPU = MVMCOptimizers
const P = MVMCExpertModeParsers
const GPU = Base.get_extension(CPU,:MVMCCUDAExt)
include("cpu_oracle.jl")
Threads.nthreads() >= 2 || error("use Julia --threads=2 for two-device execution")
BLAS.set_num_threads(1)
path = isempty(ARGS) ? joinpath(@__DIR__,"..","..","benchmark","cuda","inputs","hubbard_chain_L128","namelist.def") : ARGS[1]
walkers = parse(Int,get(ENV,"JULIA_MVMC_GPU_WALKERS","256"))
iseven(walkers) && walkers > 0 || error("walkers must be positive and even")
data,rng = GPU.prepared_gpu_data(path;seed=79)
fixed = GPU.initial_configurations(data,walkers,rng)
commands = [Channel{Symbol}(1) for _ in 1:2]
responses = [Channel{Any}(1) for _ in 1:2]
workers = map(0:1) do device
    Threads.@spawn begin
        try
            CUDA.device!(device)
            CUDA.allowscalar(false)
            cols = device*div(walkers,2)+1:(device+1)*div(walkers,2)
            s = gpu_state(data;configurations=fixed[:,cols])
            o = gpu_observables_workspace(s,data)
            sr = GPU.CUDASR(s,data)
            evaluate() = (gpu_observables!(s,o); GPU.derivatives!(s,sr,o); CUDA.synchronize())
            evaluate(); evaluate()
            put!(responses[device+1],(:ready,nothing))
            while true
                command = take!(commands[device+1])
                command == :stop && break
                if command == :run
                    put!(responses[device+1],(:time,@elapsed evaluate()))
                elseif command == :results
                    put!(responses[device+1],(:results,(Array(o.values),Array(sr.derivatives))))
                end
            end
        catch error
            put!(responses[device+1],(:error,(error,catch_backtrace())))
            rethrow()
        end
    end
end
function receive(device)
    tag,value = take!(responses[device])
    tag == :error && throw(value[1])
    return value
end
for i in 1:2
    receive(i)
end
println("setup_complete nsite=",data.modpara.nsite," walkers=",walkers," devices=0,1 CPU_BLAS=",BLAS.get_config()," Julia=",VERSION)
flush(stdout)
values = Float64[]
for _ in 1:5
    elapsed = @elapsed begin
        foreach(c->put!(c,:run),commands)
        foreach(receive,1:2)
    end
    push!(values,elapsed)
end
foreach(c->put!(c,:results),commands)
results = [receive(i) for i in 1:2]
actual = hcat(first.(results)...)
actualO = hcat(last.(results)...)
c = cpu_state(data)
cpu_observables(data,c,Int.(fixed[:,1]))
cpu_times = [@elapsed [cpu_observables(data,c,Int.(fixed[:,b])) for b in 1:walkers] for _ in 1:5]
reference = [cpu_observables(data,c,Int.(fixed[:,b])) for b in 1:walkers]
expected = hcat(first.(reference)...)
expectedO = hcat(last.(reference)...)
println("GPU_two_devices_wall_min_median_max_s=",(minimum(values),sort(values)[3],maximum(values)))
println("CPU_single_thread_min_median_max_s=",(minimum(cpu_times),sort(cpu_times)[3],maximum(cpu_times)))
println("max_observable_error=",maximum(abs,actual-expected)," max_derivative_error=",maximum(abs,actualO-expectedO))
bad = findall(.!isapprox.(actual,expected;atol=1e-9,rtol=1e-9))
badO = findall(.!isapprox.(actualO,expectedO;atol=1e-9,rtol=1e-9))
println("mismatched_observables=",length(bad)," mismatched_derivatives=",length(badO))
for index in bad[1:min(10,length(bad))]
    println("mismatch=",index," GPU=",actual[index]," CPU=",expected[index])
end
dump = get(ENV,"JULIA_MVMC_GPU_DEBUG_DUMP","")
if !isempty(dump)
    serialize(dump,(;fixed,observed=actual,derivatives=actualO,expected,expectedO))
end
flush(stdout)
try
    all(isapprox.(actual,expected;atol=1e-9,rtol=1e-9)) || error("two-device observables differ from CPU")
    all(isapprox.(actualO,expectedO;atol=1e-9,rtol=1e-9)) || error("two-device derivatives differ from CPU")
finally
    foreach(c->put!(c,:stop),commands)
    foreach(fetch,workers)
end
