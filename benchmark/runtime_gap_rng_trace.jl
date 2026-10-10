# Optional developer audit: forward the actual SFMT C primitives and record
# integer words, primitive kind, seed calls and draw count. No RNG replacement.
using MPI, MVMCOptimizers, SFMT, LinearAlgebra, Random, SHA
const AUDIT_STREAM = IOBuffer()
const AUDIT_U32 = Ref(0)
const AUDIT_REAL2 = Ref(0)
const AUDIT_SEEDS = UInt32[]
function SFMT.C_API.gen_rand32()
    word = ccall((:gen_rand32,SFMT.C_API.libsfmt),UInt32,())
    write(AUDIT_STREAM,UInt8(1),word)
    AUDIT_U32[] += 1
    return word
end
function SFMT.C_API.genrand_real2()
    # The dump restores the original C state, including its index.
    word = only(SFMT.sfmt_dump_rand32(1))
    value = ccall((:genrand_real2,SFMT.C_API.libsfmt),Cdouble,())
    expected = ccall((:to_real2,SFMT.C_API.libsfmt),Cdouble,(UInt32,),word)
    @assert value == expected # exact RNG conversion contract, not numerical output
    write(AUDIT_STREAM,UInt8(2),word)
    AUDIT_REAL2[] += 1
    return value
end
function SFMT.C_API.init_gen_rand(seed)
    value=UInt32(seed)
    push!(AUDIT_SEEDS,value)
    write(AUDIT_STREAM,UInt8(3),value)
    ccall((:init_gen_rand,SFMT.C_API.libsfmt),Cvoid,(UInt32,),value)
end
BLAS.set_num_threads(1)
MPI.Init_thread(MPI.THREAD_FUNNELED)
rank=MPI.Comm_rank(MPI.COMM_WORLD)
input_root,output_root,ranks_s=ARGS
ranks=parse(Int,ranks_s)
@assert MPI.Comm_size(MPI.COMM_WORLD)==ranks
try
    for size in (16,24,32)
        output=joinpath(output_root,"L$size")
        mkpath(output)
        truncate(AUDIT_STREAM,0); seekstart(AUDIT_STREAM)
        AUDIT_U32[]=0; AUDIT_REAL2[]=0; empty!(AUDIT_SEEDS)
        result=MVMCOptimizers.run_para_opt_from_namelist(
            joinpath(input_root,"L$size-ranks$ranks-threads$(Threads.nthreads())","inputs","namelist.def");
            nsteps=20,nsmp=20,mode=:real,output_dir=output)
        @assert result.status==0
        @assert AUDIT_U32[]>0 && AUDIT_REAL2[]>0
        @assert AUDIT_SEEDS==UInt32[1+rank]
        bytes=take!(AUDIT_STREAM)
        @assert length(bytes)==5*(AUDIT_U32[]+AUDIT_REAL2[]+length(AUDIT_SEEDS))
        open(joinpath(output,"rng-trace-rank-$rank.txt"),"w") do io
            println(io,"u32=$(AUDIT_U32[]) real2=$(AUDIT_REAL2[]) words=$(AUDIT_U32[]+AUDIT_REAL2[]) seed=$(only(AUDIT_SEEDS))")
            println(io,"sha256=$(bytes2hex(sha256(bytes)))")
        end
        MPI.Barrier(MPI.COMM_WORLD)
    end
catch err
    showerror(stderr,err,catch_backtrace())
    MPI.Abort(MPI.COMM_WORLD,1)
end
MPI.Finalize()
