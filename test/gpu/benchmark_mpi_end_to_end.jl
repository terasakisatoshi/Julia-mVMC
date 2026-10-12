# Optional whole-run strong-scaling measurement. No CUDA is loaded by this driver.
using MVMCOptimizers, LinearAlgebra, Statistics, SHA
const CPU=MVMCOptimizers
const Parsers=CPU.MVMCExpertModeParsers
const MPI=CPU.MPI
function arguments(args)
    d=Dict{String,String}()
    i=1
    while i<=length(args)
        startswith(args[i],"--") || error("expected --key value")
        key=args[i][3:end]
        if occursin('=',key)
            k,v=split(key,'=';limit=2);d[k]=v;i+=1
        else
            i<length(args) || error("missing value for --$key")
            d[key]=args[i+1];i+=2
        end
    end
    return d
end
function override_modpara!(path,settings)
    text=read(path,String)
    for (key,value) in settings
        pattern=Regex("(?m)^\\s*"*key*"\\s+[^\\r\\n]*")
        line=key*"  "*string(value)
        text=occursin(pattern,text) ? replace(text,pattern=>line) : text*"\n"*line*"\n"
    end
    write(path,text)
end
function fixed_checkpoint(namelist,directory,seed)
    data=Parsers.parse_expert_mode_files(namelist)
    data.modpara.nmp_trans=max(1,abs(data.modpara.nmp_trans))
    if data.modpara.nelec==0
        data.modpara.nelec=(data.modpara.nlocspin+data.modpara.ncond)÷2
    end
    rng=CPU.SFMT19937RNG();CPU.Random.seed!(rng,seed)
    Parsers.init_parameter!(data;rng)
    initial=joinpath(dirname(namelist),"initial.def")
    if isfile(initial)
        CPU.read_initial_def!(data,initial) || error("initial.def load failed")
    end
    Parsers.read_input_parameters!(data,namelist)
    CPU.sync_modified_parameter!(data;shift_correlations=false)
    Parsers.init_qp_weight!(data)
    data.modpara.nsr_opt_itr_smp=2
    state=CPU.VMCOptimizationState(data.modpara.nsite,data.modpara.nelec,
        Parsers.projection_layout(data).n_proj,CPU.count_total_parameters(data),CPU.get_n_qp_full(data),1,false,false)
    for _ in 1:2
        push!(state.opt_data,CPU.OptDataPoint(0.0+0im,0.0+0im,CPU.pack_parameters(data)))
    end
    CPU.output_opt_data!(data,state;output_dir=directory)
    head=isempty(data.modpara.c_para_file_head) ? "zqp" : data.modpara.c_para_file_head
    path=joinpath(directory,head*"_opt.dat")
    loaded=deepcopy(data)
    CPU.read_opt_para_file!(loaded,path)==CPU.count_total_parameters(data) || error("incomplete checkpoint")
    isapprox(CPU.pack_parameters(loaded),CPU.pack_parameters(data);atol=1e-14,rtol=1e-14) || error("checkpoint changed parameters")
    return path
end
function main()
    opts=arguments(ARGS)
    input=abspath(opts["input"])
    mode=get(opts,"mode","qp")
    phase=get(opts,"phase","both")
    mode in ("qp","groups") && phase in ("opt","phys","both") || error("invalid mode/phase")
    total=parse(Int,get(opts,"samples","300"));steps=parse(Int,get(opts,"steps","20"))
    reps=parse(Int,get(opts,"reps","3"));seed=parse(Int,get(opts,"seed","79"))
    warmup=parse(Int,get(opts,"warmup","1"));interval=parse(Int,get(opts,"interval","1"))
    physruns=parse(Int,get(opts,"phys-runs","1"))
    total>0 && steps>0 && reps>0 && warmup>=0 && interval>0 && physruns>0 || error("invalid lengths")
    Threads.nthreads()==1 || error("use Julia --threads=1")
    BLAS.set_num_threads(1)
    ENV["JULIA_MVMC_MPI"]="1"
    MPI.Initialized() || MPI.Init()
    comm=MPI.COMM_WORLD;rank=MPI.Comm_rank(comm);size=MPI.Comm_size(comm)
    size==parse(Int,get(opts,"expect-ranks",string(size))) || error("unexpected MPI world size")
    groups=mode=="groups" ? size : 1
    total%groups==0 || error("total samples must divide equally across groups")
    pergroup=div(total,groups);nsplit=mode=="qp" ? size : 1
    root=rank==0 ? abspath(get(opts,"output",mktempdir(;cleanup=false))) : ""
    root=MPI.bcast(root,comm;root=0)
    namelist=joinpath(root,"input",basename(input))
    checkpoint=joinpath(root,"fixed","zqp_opt.dat")
    if rank==0
        mkpath(root)
        cloned=joinpath(root,"input")
        ispath(cloned) && error("output input directory already exists: $cloned")
        cp(dirname(input),cloned)
        entry=only(filter(line->length(split(line))>=2 && lowercase(first(split(line)))=="modpara",readlines(namelist)))
        modfile=split(entry)[2]
        modpath=normpath(joinpath(cloned,modfile))
        relative=relpath(modpath,cloned)
        (!isabspath(modfile) && relative!=".." && !startswith(relative,"../") && !islink(modpath)) || error("unsafe ModPara path")
        override_modpara!(modpath,["NVMCSample"=>pergroup,"NVMCWarmUp"=>warmup,"NVMCInterval"=>interval,
            "NSROptItrStep"=>steps,"NSROptItrSmp"=>1,"NDataQtySmp"=>physruns,
            "NSRCG"=>0,"NSplitSize"=>nsplit,"RndSeed"=>seed])
        supplied=get(opts,"checkpoint","auto")
        checkpoint=supplied=="auto" ? fixed_checkpoint(namelist,joinpath(root,"fixed"),seed) : abspath(supplied)
        isfile(checkpoint) || error("checkpoint missing")
    end
    checkpoint=MPI.bcast(checkpoint,comm;root=0)
    MPI.Barrier(comm)
    data=Parsers.parse_expert_mode_files(namelist)
    ns=data.modpara.nsite
    allowed=only(filter(line->startswith(line,"Cpus_allowed_list:"),readlines("/proc/self/status")))
    affinities=MPI.gather((rank,allowed),comm;root=0)
    if rank==0
        println("scope=whole_public_runner; MPI_MAX_wall; cold_compilation_separate; samples_global=$total samples_per_group=$pergroup groups=$groups NSplitSize=$nsplit world=$size mode=$mode steps=$steps warmup=$warmup interval=$interval")
        println("Julia=",VERSION," BLAS=",BLAS.get_config()," MPI=",MPI.identify_implementation()," threads=",Threads.nthreads())
        println("input=",namelist," checkpoint=",checkpoint," checkpoint_sha256=",bytes2hex(sha256(read(checkpoint))))
        println("optimization_sampling_proposal_attempts_all_groups=",groups*((warmup+pergroup)+(steps-1)*(pergroup+1))*interval*ns)
        println("group_seeds=base+group1 base=",seed,"; QPsplit shares one SFMT trajectory; groups mode uses independent C group seeds")
        println("rank_cpu_affinities=",affinities)
        flush(stdout)
    end
    opttimes=Float64[];phystimes=Float64[]
    for temperature in ("cold","warm")
        for rep in 1:(temperature=="cold" ? 1 : reps)
            for kind in (phase=="both" ? ("opt","phys") : (phase,))
                output=joinpath(root,"$temperature-$rep-$kind")
                GC.gc();MPI.Barrier(comm)
                start=time_ns()
                result=kind=="opt" ? CPU.run_para_opt_from_namelist(namelist;nsteps=steps,nsmp=1,mode=:real,seed,output_dir=output) :
                    CPU.run_phys_cal_from_namelist(namelist;opt_para=checkpoint,mode=:real,seed,output_dir=output)
                elapsed=(time_ns()-start)/1e9
                elapsed=MPI.Allreduce(elapsed,MPI.MAX,comm)
                status=MPI.Allreduce(Int(result.status != 0),MPI.MAX,comm)
                status==0 || error("MPI public runner status $status")
                if rank==0
                    if kind=="opt"
                        isfinite(result.final_energy_per_site) || error("nonfinite optimization energy")
                        println("phase=$temperature rep=$rep kind=opt MPI_MAX_seconds=$elapsed energy_per_site=",result.final_energy_per_site)
                    else
                        head=isempty(data.modpara.c_data_file_head) ? "zvo" : data.modpara.c_data_file_head
                        idx=lpad(data.modpara.n_data_idx_start,3,'0')
                        energy=parse(Float64,first(split(read(joinpath(output,head*"_out_"*idx*".dat"),String))))/ns
                        isfinite(energy) || error("nonfinite PhysCal energy")
                        println("phase=$temperature rep=$rep kind=phys MPI_MAX_seconds=$elapsed energy_per_site=$energy")
                    end
                    flush(stdout)
                end
                temperature=="warm" && push!(kind=="opt" ? opttimes : phystimes,elapsed)
            end
        end
    end
    if rank==0
        !isempty(opttimes) && println("MPI_opt_warm_min_median_max_s=",(minimum(opttimes),median(opttimes),maximum(opttimes)))
        !isempty(phystimes) && println("MPI_phys_warm_min_median_max_s=",(minimum(phystimes),median(phystimes),maximum(phystimes)))
        println("artifacts=",root);flush(stdout)
    end
end
try
    main()
catch error
    showerror(stderr,error,catch_backtrace());println(stderr);flush(stderr)
    MPI.Initialized() && !MPI.Finalized() && MPI.Abort(MPI.COMM_WORLD,1)
    rethrow()
end
