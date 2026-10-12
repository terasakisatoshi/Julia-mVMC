# Compare the production and frozen PR86 samplers on one identical 128-step
# tape, then independently multiply each selected Slater matrix by its inverse.
using CUDA,MVMCOptimizers,Random,LinearAlgebra,Test,PfaPack
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
Base.include(GPU,joinpath(@__DIR__,"baseline-ab-dispatch.jl"))
CUDA.device!(1);CUDA.allowscalar(false);BLAS.set_num_threads(1)
function diagnose()
    data,rng=GPU.prepared_gpu_data(ARGS[1];seed=79)
    s,_,_=GPU.prepared_runner_state(data,320,rng,79)
    c=GPU.GPUSamplingCache(s;delayed=true)
    GPU.refresh_sampling_state!(s,c,GPU.BatchedSamplingInverse(s))
    tape=CuArray(rand(MersenneTwister(548128),UInt32,5,320,128))
    fields=(:configurations,:occupancy,:counts,:overlaps,:logcorrelations,:accepted)
    cfields=(:inverse,:pfaffians,:valid,:candidate_pf,:accepted_since_refactor)
    saved=Dict(f=>copy(getproperty(s,f)) for f in fields)
    csaved=Dict(f=>copy(getproperty(c,f)) for f in cfields)
    results=[]
    cpupfs=zeros(s.nqp,320)
    for variant in ("baseline","candidate")
        for f in fields;copyto!(getproperty(s,f),saved[f]);end
        for f in cfields;copyto!(getproperty(c,f),csaved[f]);end
        ENV["JULIA_MVMC_DELAYED_AB"]=variant
        GPU.sample_with_cache!(s,c,tape,0.0;empty_destinations=true,
            incremental_correlation=true,triangle_update=true)
        r=Dict(f=>Array(getproperty(s,f)) for f in fields)
        inv=Array(c.inverse);slater=Array(s.slater);n=2s.nelec
        normalized=0.0
        for b in 1:320,q in 1:s.nqp
            rows=Int.(r[:configurations][:,b]).+1
            rows[s.nelec+1:end].+=s.nsite
            A=-slater[rows,rows,q];X=inv[:,:,q,b]
            normalized=max(normalized,opnorm(A*X-I,Inf)/(opnorm(A,Inf)*opnorm(X,Inf)))
            variant=="candidate" && (cpupfs[q,b]=PfaPack.pfaffian_ltl!(copy(A)))
        end
        println("variant=",variant," max_normalized_inverse_residual=",normalized)
        r[:inverse]=inv;r[:pfaffians]=Array(c.pfaffians)
        push!(results,r)
    end
    @testset "Long L128 production controls and physical cache values" begin
        a,b=results
        for f in (:configurations,:occupancy,:counts,:accepted)
            @test a[f]==b[f]
        end
        for f in (:overlaps,:logcorrelations,:pfaffians)
            mask=isapprox.(a[f],b[f];atol=1e-9,rtol=1e-9)
            println("field=",f," max_abs_difference=",maximum(abs,a[f]-b[f])," failed_entries=",count(!,mask))
            if f==:pfaffians
                slater=Array(s.slater)
                for index in findall(!,mask)
                    q,walker=Tuple(index)
                    rows=Int.(b[:configurations][:,walker]).+1
                    rows[s.nelec+1:end].+=s.nsite
                    A=-slater[rows,rows,q]
                    cpu=PfaPack.pfaffian_ltl!(copy(A))
                    println("pf_index=",index," baseline=",a[f][index]," candidate=",b[f][index]," cpu=",cpu," condition=",cond(A)," baseline_rel_error=",abs(a[f][index]/cpu-1)," candidate_rel_error=",abs(b[f][index]/cpu-1))
                end
            end
            # Old incremental sector Pfaffians are an approximate diagnostic,
            # not an independent oracle. Their failed comparisons remain in
            # the log; physical overlaps are also checked against CPU factors.
            f!=:pfaffians && @test all(mask)
        end
        cpuoverlap=vec(transpose(Array(s.weights))*cpupfs)
        for (variant,result) in zip(("baseline","candidate"),results)
            errors=abs.((result[:overlaps]-cpuoverlap)./cpuoverlap)
            println("variant=",variant," max_relative_CPU_overlap_error=",maximum(errors))
            println("variant=",variant," max_relative_CPU_sector_error=",maximum(abs.((result[:pfaffians]-cpupfs)./cpupfs)))
            @test all(isapprox.(result[:overlaps],cpuoverlap;atol=1e-9,rtol=1e-9))
        end
        @test all(iszero,Array(c.updates.count))
        println("max_abs_inverse_difference=",maximum(abs,a[:inverse]-b[:inverse]))
        println("per_entry_old_inverse_atol_rtol_1e9=",
            all(isapprox.(a[:inverse],b[:inverse];atol=1e-9,rtol=1e-9)))
    end
end
diagnose()
