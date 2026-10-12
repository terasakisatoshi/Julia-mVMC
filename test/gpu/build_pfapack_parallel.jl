# Optional Linux-only scheduling of the three independent canonical native builds.
# Keep numerical sources, compiler flags and link order unchanged.
using SHA
Sys.islinux() || error("this helper reproduces the Linux build commands")
length(ARGS) == 1 || error("usage: build_pfapack_parallel.jl PfaPack.jl/deps")
deps = abspath(only(ARGS))
expected = "ac71bd3d0ca72284aec10db36e201bee2b90015aaa41d35d26ae281dc6a9cfb6"
actual = bytes2hex(sha256(read(joinpath(deps, "build.jl"))))
actual == expected || error("canonical PfaPack build.jl changed (SHA-256 $actual); use Pkg.build or update this helper after reviewing the new canonical commands")
budget = haskey(ENV, "MVMC_BUILD_JOBS") ? parse(Int, ENV["MVMC_BUILD_JOBS"]) :
    parse(Int, readchomp(`env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc`))
budget > 0 || error("MVMC_BUILD_JOBS must be a positive integer")
workers = min(3, budget)
println("canonical_build_sha256=", actual)
println("build_job_budget=", budget, " parallel_independent_native_builds=", workers)
commands = [
    `gfortran -shared -fPIC -O3 -o $(joinpath(deps,"libzsktf2.so")) $(joinpath(deps,"zskr2.f")) $(joinpath(deps,"zsktf2.f")) -lblas -llapack`,
    `gfortran -shared -fPIC -O3 -o $(joinpath(deps,"libdsktf2.so")) $(joinpath(deps,"dskr2.f")) $(joinpath(deps,"dsktf2.f")) -lblas -llapack`,
    `g++ -shared -fPIC -std=c++11 -O3 -D_CC_IMPL -DBLAS_EXTERNAL -I$deps -o $(joinpath(deps,"libltl2inv.so")) $(joinpath(deps,"ltl2inv.cc")) $(joinpath(deps,"ilaenv_lauum.cc")) -lblas -llapack`,
]
foreach(println, commands)
flush(stdout)
errors = Exception[]
for first in 1:workers:length(commands)
    tasks = map(commands[first:min(first + workers - 1, length(commands))]) do command
        @async run(command)
    end
    for task in tasks
        try
            fetch(task)
        catch error
            push!(errors, error)
        end
    end
end
isempty(errors) || throw(CompositeException(errors))
println("native_builds_completed=", length(commands))
flush(stdout)
