using MVMCOptimizers
const MPI=MVMCOptimizers.MPI
length(ARGS)>=2 || error("usage: launch_mpi.jl NP DRIVER [driver arguments]")
np=parse(Int,ARGS[1]);np>0 || error("NP must be positive")
driver=abspath(ARGS[2])
project=dirname(Base.active_project())
driverargs=vcat(ARGS[3:end],["--expect-ranks",string(np)])
cmd=`$(MPI.mpiexec()) -n $np $(Base.julia_cmd()) --threads=1 --project=$project $driver $driverargs`
# Keep MPI.jl's launcher library environment, then add only benchmark controls.
binding=get(ENV,"HYDRA_BINDING","core")
cmd=addenv(cmd,"JULIA_MVMC_MPI"=>"1","JULIA_NUM_THREADS"=>"1","OPENBLAS_NUM_THREADS"=>"1","OMP_NUM_THREADS"=>"1","HYDRA_BINDING"=>binding)
println("launch_ranks=",np," launcher=",first(MPI.mpiexec().exec)," project=",project);flush(stdout)
println("rank_binding=",binding);flush(stdout)
run(cmd)
