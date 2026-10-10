using MPI, MVMCOptimizers, LinearAlgebra, Profile
BLAS.set_num_threads(1)
MPI.Init_thread(MPI.THREAD_FUNNELED)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
input, output, steps_s = ARGS
steps = parse(Int, steps_s)
mkpath(output)
function production(label)
    MVMCOptimizers.run_para_opt_from_namelist(input; nsteps=steps, nsmp=steps,
        mode=:real, output_dir=joinpath(output, label))
end
try
    production("warmup")
    Profile.init(n=10^7, delay=0.001)
    Profile.clear()
    measured = @timed Profile.@profile production("profile")
    @assert measured.value.status == 0
    open(joinpath(output, "timed-rank-$rank.txt"), "w") do io
        println(io, "seconds=$(measured.time) allocated_bytes=$(measured.bytes) gc_seconds=$(measured.gctime)")
        println(io, measured.gcstats)
    end
    open(joinpath(output, "flat-rank-$rank.txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (10000, 250)), format=:flat,
            sortedby=:count, C=true, mincount=5)
    end
    open(joinpath(output, "tree-rank-$rank.txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (10000, 250)), C=true, mincount=10)
    end
    ENV["MVMC_C_TIMER"] = "1"
    production("timer-warmup")
    measured = @timed production("timer")
    open(joinpath(output, "timer-timed-rank-$rank.txt"), "w") do io
        println(io, "seconds=$(measured.time) allocated_bytes=$(measured.bytes) gc_seconds=$(measured.gctime)")
    end
finally
    MPI.Finalize()
end
