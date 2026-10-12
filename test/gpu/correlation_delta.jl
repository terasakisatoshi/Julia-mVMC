using CUDA, MVMCOptimizers, Random, Test, LinearAlgebra
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers, :MVMCCUDAExt)

function delta_check_kernel!(out, unchanged, ops, cfg, moved, counts, occ, gutz,
                             jastrow, parameters, ns, ne, ng, nj, batch)
    b = (blockIdx().x-1)*blockDim().x+threadIdx().x
    b > batch && return nothing
    before = GPU.correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
    delta = GPU.operator_correlation_delta(ops[b], occ, gutz, jastrow, parameters, ns, ng, nj, b)
    # The helper must leave the base occupation/counts intact for the next op.
    ok = true
    @inbounds for r in 1:2ns
        spin = r > ns ? 1 : 0
        site = (r-1)%ns
        expected = Int32(0)
        for k in spin*ne+1:(spin+1)*ne
            expected |= cfg[k,b] == site
        end
        ok &= occ[r,b] == expected
    end
    @inbounds unchanged[b] = ok
    after = GPU.correlation!(counts, occ, moved, gutz, jastrow, parameters, ns, ne, ng, nj, b)
    @inbounds out[1,b] = before
    @inbounds out[2,b] = delta
    @inbounds out[3,b] = after
    return nothing
end

function apply_op(base, op, ne)
    cfg = copy(base)
    for (i,j,spin) in (op[7] == 2 ? ((op[4],op[5],op[6]),(op[1],op[2],op[3])) : ((op[1],op[2],op[3]),))
        range = spin*ne+1:(spin+1)*ne
        electron = findfirst(k -> cfg[k] == j, range)
        electron === nothing && return nothing
        i != j && any(k -> cfg[k] == i, range) && return nothing
        cfg[range[electron]] = i
    end
    return cfg
end

function reference_counts(cfg, gutz, jastrow, ns, ne, ng, nj)
    occ = zeros(Int,2ns)
    for k in eachindex(cfg)
        occ[cfg[k]+(k > ne ? ns : 0)+1] = 1
    end
    counts = zeros(Int32,ng+nj)
    for i in 1:ns
        0 <= gutz[i] < ng && (counts[gutz[i]+1] += occ[i]*occ[ns+i])
        for j in i+1:ns
            if 0 <= jastrow[i,j] < nj
                counts[ng+jastrow[i,j]+1] += (occ[i]+occ[ns+i]-1)*(occ[j]+occ[ns+j]-1)
            end
        end
    end
    return counts
end

@testset "Incremental ordinary-operator correlation logs" begin
    rng = MersenneTwister(731)
    for ns in (6,128)
        ne=ns÷2; ng=3; nj=5
        base=Int32.(vcat(0:2:ns-1,1:2:ns-1))
        candidates=GPU.GPUOperator[]
        sites = ns == 6 ? (0:ns-1) : (0:7)
        for i in sites,j in sites,s in 0:1
            push!(candidates,GPU.op1(i,j,s))
            for k in sites,l in sites,t in 0:1
                push!(candidates,GPU.op2(i,j,s,k,l,t))
            end
        end
        if ns == 128
            for _ in 1:1000
                i,j,k,l=rand(rng,0:ns-1,4)
                push!(candidates,GPU.op2(i,j,rand(rng,0:1),k,l,rand(rng,0:1)))
            end
        end
        ops=GPU.GPUOperator[]; moved=Vector{Int32}[]
        for op in candidates
            cfg=apply_op(base,op,ne)
            cfg === nothing && continue
            push!(ops,op); push!(moved,cfg)
        end
        batch=length(ops)
        gutz=Int32.([i%5-1 for i in 1:ns])
        jastrow=Int32.([((i+3j)%7)-1 for i in 1:ns,j in 1:ns])
        parameters=[0.19,-0.11,0.07,0.13,-0.17,0.031,0.23,-0.09]
        out=CUDA.zeros(Float64,3,batch); counts=CUDA.zeros(Int32,ng+nj,batch)
        occ=CUDA.zeros(Int32,2ns,batch); unchanged=CUDA.zeros(Bool,batch)
        @cuda threads=128 blocks=cld(batch,128) delta_check_kernel!(out,unchanged,CuArray(ops),
            CuArray(repeat(base,1,batch)),CuArray(hcat(moved...)),counts,occ,
            CuArray(gutz),CuArray(jastrow),CuArray(parameters),ns,ne,ng,nj,batch)
        actual=Array(out)
        expected=hcat([reference_counts(cfg,gutz,jastrow,ns,ne,ng,nj) for cfg in moved]...)
        @test Array(counts) == expected
        @test all(Array(unchanged))
        @test all(isapprox.(actual[1,:]+actual[2,:],actual[3,:];atol=2e-12,rtol=2e-12))
        @test all(isapprox.(actual[3,:],vec(parameters'*expected);atol=2e-12,rtol=2e-12))
        println("correlation delta nsite=$ns valid_operators=$batch")
    end
end
