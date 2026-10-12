# Read-only electron-label overlay. Independent operators never mutate shared
# configurations, so every thread sees the same reference walker.
struct OperatorConfiguration{C} <: AbstractMatrix{Int32}
    parent::C
    first::Int
    second::Int
    first_site::Int32
    second_site::Int32
    walker::Int
end
Base.size(c::OperatorConfiguration) = size(c.parent)
@inline function Base.getindex(c::OperatorConfiguration, i::Int, b::Int)
    if b == c.walker
        i == c.first && return c.first_site
        i == c.second && return c.second_site
    end
    @inbounds return c.parent[i,b]
end

@inline function operator_electron(cfg, destination, source, spin, ne, b)
    electron = 0
    @inbounds for k in spin*ne+1:(spin+1)*ne
        if cfg[k,b] == source
            electron = k
        end
        if destination != source && cfg[k,b] == destination
            return 0
        end
    end
    return electron
end

@inline function parallel_green_value(op, cfg, occ, overlaps, logs, gutz, jastrow,
                                     parameters, slater, weights, factors, pivots, pfs,
                                     inv, pf, valid, transformed, ns, ne, ng, nj, nq, batch, b)
    k = 0
    if op[7] == 2
        k = operator_electron(cfg, op[4], op[5], op[6], ne, b)
        k == 0 && return 0.0
    end
    right = OperatorConfiguration(cfg, 0, k, Int32(0), op[4], b)
    a = operator_electron(right, op[1], op[2], op[3], ne, b)
    a == 0 && return 0.0
    if op[1] == op[2] && (op[7] == 1 || op[4] == op[5])
        return 1.0
    end
    moved = OperatorConfiguration(cfg, a, k, op[1], op[4], b)
    @inbounds newlog = logs[b] + operator_correlation_delta(
        op, occ, gutz, jastrow, parameters, ns, ng, nj, b)
    newip = updated_overlap!(slater, weights, moved, factors, pivots, pfs,
        inv, pf, valid, transformed, a, k, op, ns, ne, nq, batch, b)
    @inbounds return exp(newlog-logs[b])*newip/overlaps[b]
end

function parallel_operator_kernel!(termvalues, ops, one, two, ex1, ex2,
                                   cfg, occ, overlaps, logs, gutz, jastrow,
                                   parameters, slater, weights, factors, pivots, pfs,
                                   inv, pf, valid, transformed, ns, ne, ng, nj, nq, batch)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    count = size(termvalues,1)
    idx > count*batch && return nothing
    t = (idx-1)%count+1
    b = (idx-1)÷count+1
    # Invalid individual projection matrices require the original scalar
    # recomputation path; they cannot share a mutable factor scratch here.
    @inbounds for q in 1:nq
        !valid[q,b] && return nothing
    end
    nops, none, ntwo = length(ops), length(one), length(two)
    if t <= nops
        @inbounds op = ops[t]
    elseif t <= nops+none
        @inbounds op = one[t-nops]
    elseif t <= nops+none+ntwo
        @inbounds op = two[t-nops-none]
    else
        e = t-nops-none-ntwo
        @inbounds op = ex1[e]
        g1 = parallel_green_value(op,cfg,occ,overlaps,logs,gutz,jastrow,
            parameters,slater,weights,factors,pivots,pfs,inv,pf,valid,
            transformed,ns,ne,ng,nj,nq,batch,b)
        @inbounds op2 = ex2[e]
        g2 = parallel_green_value(op2,cfg,occ,overlaps,logs,gutz,jastrow,
            parameters,slater,weights,factors,pivots,pfs,inv,pf,valid,
            transformed,ns,ne,ng,nj,nq,batch,b)
        @inbounds termvalues[t,b] = g1*g2
        return nothing
    end
    value = parallel_green_value(op,cfg,occ,overlaps,logs,gutz,jastrow,
        parameters,slater,weights,factors,pivots,pfs,inv,pf,valid,
        transformed,ns,ne,ng,nj,nq,batch,b)
    @inbounds termvalues[t,b] = value
    return nothing
end

function ordered_observable_reduce_kernel!(values, termvalues, coeffs, valid, nq, batch)
    b = (blockIdx().x-1)*blockDim().x+threadIdx().x
    b > batch && return nothing
    @inbounds for q in 1:nq
        !valid[q,b] && return nothing
    end
    # Preserve the Hamiltonian term addition order, rather than a tree sum.
    energy = 0.0
    @inbounds for t in eachindex(coeffs)
        energy += coeffs[t]*termvalues[t,b]
    end
    @inbounds values[1,b] = energy
    @inbounds for row in 2:size(values,1)
        values[row,b] = termvalues[length(coeffs)+row-1,b]
    end
    return nothing
end

function parallel_observables!(s, o)
    batch = size(s.configurations,2)
    count = size(o.termvalues,1)
    if count != 0
        @cuda threads=256 blocks=cld(count*batch,256) fastmath=false parallel_operator_kernel!(
            o.termvalues,o.ops,o.one,o.two,o.ex1,o.ex2,s.configurations,
            s.occupancy,s.overlaps,s.logcorrelations,s.gutz,s.jastrow,s.parameters,
            s.slater,s.weights,factor_view(s.factors),pivot_view(s.pivots),s.pfaffians,
            cache_view(o.inverse),o.pfaffians,o.valid,cache_view(o.transformed),
            s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch)
    end
    @cuda threads=128 blocks=cld(batch,128) fastmath=false ordered_observable_reduce_kernel!(
        o.values,o.termvalues,o.coeffs,o.valid,s.nqp,batch)
    return nothing
end
