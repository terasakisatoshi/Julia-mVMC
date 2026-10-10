using MVMCOptimizers, SFMT, LinearAlgebra
const captured=Dict{String,Dict{String,Any}}()
const label=Ref("")
@eval SFMT.C_API begin
    const probe_count=Ref(0)
    function init_gen_rand(seed)
        probe_count[]=0
        ccall((:init_gen_rand,libsfmt),Cvoid,(UInt32,),seed)
    end
    function gen_rand32()
        probe_count[]+=1
        ccall((:gen_rand32,libsfmt),UInt32,())
    end
    function genrand_real2()
        probe_count[]+=1
        ccall((:genrand_real2,libsfmt),Cdouble,())
    end
end
function record(state,stage)
    snapshot=Dict{String,Any}("draws"=>SFMT.C_API.probe_count[])
    for field in (:ele_idx,:ele_cfg,:ele_num,:ele_proj_cnt,:counter)
        snapshot["config.$field"]=copy(getproperty(state.electron_config,field))
    end
    for field in (:sr_opt_oo_real,:sr_opt_ho_real,:sr_opt_o_real,:sr_opt_o_store_real)
        snapshot["sr.$field"]=copy(getproperty(state.sr_opt,field))
    end
    for field in (:inv_m_real,:pf_m_real)
        snapshot["slater.$field"]=copy(getproperty(state.slater_matrix,field))
    end
    captured[label[]*":"*stage]=snapshot
end
function install_observer!()
    source=read(joinpath(dirname(pathof(MVMCOptimizers)),"vmc_para_opt.jl"),String)
    start=first(findfirst("function vmc_para_opt!(",source)); stop=last(findnext("\nend",source,start))
    method=source[start:stop]
    for timer in (4,21)
        needle="ctimer_stop!(timer, $timer)"
        @assert count(needle,method)==1
        method=replace(method,needle=>needle*"; Main.record(state, \"stage$timer\")")
    end
    Core.eval(MVMCOptimizers,Meta.parse(method))
end
function run(label_value)
    label[]=label_value
    install_observer!()
    root=normpath(joinpath(@__DIR__,"..","reference","heisenberg_chain_real_nsrcg"))
    output=mktempdir()
    result=MVMCOptimizers.run_para_opt_from_namelist(joinpath(root,"inputs","namelist.def");nsteps=1,nsmp=1,mode=:real,output_dir=output)
    @assert result.status==0
    captured[label_value*":parameters"]=Dict("values"=>parse.(Float64,split(read(joinpath(output,"zqp_opt.dat"),String))))
end
run("candidate")
source=read(joinpath(dirname(pathof(MVMCOptimizers)),"ordinary_real_inverse.jl"),String)
start=first(findfirst("function _ordinary_utu2inv_real_c_order!(",source)); stop=last(findnext("\nend",source,start))
method=source[start:stop]
@assert count("if n <= 64",method)==1
Core.eval(MVMCOptimizers,Meta.parse(replace(method,"if n <= 64"=>"if false")))
run("candidate_lapack")
for file in ("ordinary_real_factor.jl","ordinary_real_inverse.jl","calculate_m_all.jl","vmc_sampling.jl","vmc_main_cal.jl")
    original=read(`git show 3254752e1acf7cfe6f5069522889814a8cace90d:MVMCOptimizers.jl/src/$file`,String)
    original=replace(original,r"(?m)^include\([^\n]+\)$"=>"nothing")
    Base.include_string(MVMCOptimizers,original,"baseline-"*file)
end
run("baseline")
for stage in ("stage4","stage21","parameters"), comparison in ("candidate","candidate_lapack")
    baseline=captured["baseline:"*stage]; values=captured[comparison*":"*stage]
    for field in sort!(collect(keys(baseline)))
        a=values[field]; b=baseline[field]
        if a isa AbstractArray
            maxdelta=isempty(a) ? 0.0 : maximum(abs,a-b)
            first=findfirst(!iszero,a-b)
            detail=first===nothing ? "" : " candidate=$(a[first]) baseline=$(b[first])"
            println("DIAG label=$comparison stage=$stage field=$field maxabs=$maxdelta first=$first$detail")
        else
            println("DIAG label=$comparison stage=$stage field=$field candidate=$a baseline=$b")
        end
    end
end
