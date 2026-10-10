using MPI, MVMCOptimizers, LinearAlgebra, Profile
BLAS.set_num_threads(1)
MPI.Init_thread(MPI.THREAD_FUNNELED)
input, output = ARGS
function production(label)
    MVMCOptimizers.run_para_opt_from_namelist(input; nsteps=20, nsmp=20,
        mode=:real, output_dir=joinpath(output,label))
end
try
    production("warmup")
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=0.001 production("alloc")
    data=Profile.Allocs.fetch()
    counts=Dict{String,Tuple{Int,Int}}()
    for alloc in data.allocs
        frame=findfirst(f -> occursin("MVMCOptimizers",string(f.file)),alloc.stacktrace)
        key=frame===nothing ? "other" : "$(alloc.type) $(alloc.stacktrace[frame].file):$(alloc.stacktrace[frame].line) $(alloc.stacktrace[frame].func)"
        count,bytes=get(counts,key,(0,0))
        counts[key]=(count+1,bytes+alloc.size)
    end
    open(joinpath(output,"allocation-sites.txt"),"w") do io
        for (site,(count,bytes)) in sort(collect(counts),by=p -> -last(p)[2])
            println(io,"$count $bytes $site")
        end
    end
finally
    MPI.Finalize()
end
