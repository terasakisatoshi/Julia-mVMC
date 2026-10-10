# Optional developer check, independent of ordinary Rust tests.
using MPI, MVMCOptimizers, SFMT
using LinearAlgebra
using Random
BLAS.set_num_threads(1)
MPI.Init_thread(MPI.THREAD_FUNNELED)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
input_root, output_root, ranks_s = ARGS
ranks = parse(Int, ranks_s)
@assert MPI.Comm_size(MPI.COMM_WORLD) == ranks
try
    for size in (16, 24, 32)
        output = joinpath(output_root, "L$size")
        # Dump the next full SFMT block before and after the fixed prefix.
        # sfmt_dump_rand32 restores the state and does not consume draws.
        mkpath(output)
        Random.seed!(SFMT19937RNG(), 1 + rank)
        initial = SFMT.sfmt_dump_rand32(624)
        open(joinpath(output, "rng-initial-rank-$rank.txt"), "w") do io
            foreach(word -> println(io, word), initial)
        end
        result = MVMCOptimizers.run_para_opt_from_namelist(
            joinpath(input_root, "L$size-ranks$ranks-threads$(Threads.nthreads())", "inputs", "namelist.def");
            nsteps=20, nsmp=20, mode=:real, output_dir=output)
        @assert result.status == 0
        block = SFMT.sfmt_dump_rand32(624)
        open(joinpath(output, "rng-next-rank-$rank.txt"), "w") do io
            foreach(word -> println(io, word), block)
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
catch err
    showerror(stderr, err, catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD, 1)
end
MPI.Finalize()
