# Normal/sz-conserved operator lists retain input term order.
# Format: (creation1, annihilation1, spin1, creation2, annihilation2, spin2, pairs).
const GPUOperator = NTuple{7,Int32}
spinint(s::Symbol) = s === :up ? 0 : s === :down ? 1 : throw(ArgumentError("invalid spin $s"))
op1(i,j,s) = GPUOperator((i,j,s,0,0,0,1))
op2(i,j,s,k,l,t) = GPUOperator((i,j,s,k,l,t,2))

function operator_tables(data)
    ops = GPUOperator[]
    coeffs = Float64[]
    add(op, coefficient) = (push!(ops, op); push!(coeffs, coefficient))
    for t in data.coulomb_intra_terms
        add(op2(t.site,t.site,0,t.site,t.site,1), t.value)
    end
    for t in data.coulomb_inter_terms, s in 0:1, u in 0:1
        add(op2(t.site1,t.site1,s,t.site2,t.site2,u), t.value)
    end
    for t in data.hund_terms, s in 0:1
        add(op2(t.site1,t.site1,s,t.site2,t.site2,s), -t.value)
    end
    for t in data.transfer_terms
        isreal(t.value) || throw(ArgumentError("CUDA requires real Transfer coefficients"))
        t.spin1 == t.spin2 || throw(ArgumentError("CUDA requires spin-conserving Transfer"))
        if t.spin == :both
            for s in 0:1
                add(op1(t.site1,t.site2,s), -real(t.value))
            end
        else
            add(op1(t.site1,t.site2,t.spin1), -real(t.value))
        end
    end
    for t in data.pair_hop_terms
        add(op2(t.site1,t.site2,0,t.site1,t.site2,1), t.value)
    end
    for t in data.exchange_terms, s in 0:1
        add(op2(t.site1,t.site2,s,t.site2,t.site1,1-s), t.value)
    end
    for t in data.inter_all_terms
        t.spin0 == t.spin1 && t.spin2 == t.spin3 && isreal(t.value) ||
            throw(ArgumentError("CUDA requires real spin-conserving InterAll"))
        add(op2(t.site0,t.site1,t.spin0,t.site2,t.site3,t.spin2), real(t.value))
    end
    one = GPUOperator[]
    two = GPUOperator[]
    ex1 = GPUOperator[]
    ex2 = GPUOperator[]
    for t in data.green_one_terms
        t.spin1 == t.spin2 || throw(ArgumentError("CUDA normal Green requires spin-conserving pairs"))
        push!(one, op1(t.site1,t.site2,spinint(t.spin1)))
    end
    for t in data.green_two_terms
        t.spin1 == t.spin2 && t.spin3 == t.spin4 ||
            throw(ArgumentError("CUDA normal Green requires spin-conserving pairs"))
        push!(two, op2(t.site1,t.site2,spinint(t.spin1),t.site3,t.site4,spinint(t.spin3)))
    end
    for t in data.green_two_ex_terms
        t.spin_i1 == t.spin_j1 && t.spin_i2 == t.spin_j2 ||
            throw(ArgumentError("CUDA normal factored Green requires spin-conserving pairs"))
        push!(ex1, op1(t.site_i1,t.site_j1,t.spin_i1))
        push!(ex2, op1(t.site_i2,t.site_j2,t.spin_i2))
    end
    return ops, coeffs, one, two, ex1, ex2
end

@inline function move_operator!(cfg, i, j, spin, ne, b)
    electron = 0
    @inbounds for k in spin*ne+1:(spin+1)*ne
        if cfg[k,b] == j
            electron = k
        end
        if i != j && cfg[k,b] == i
            return 0
        end
    end
    electron == 0 && return 0
    @inbounds cfg[electron,b] = i
    return electron
end

@inline function green_value!(op, cfg, counts, occ, overlaps, logs, gutz, jastrow,
                              parameters, slater, weights, factors, pivots, pfs,
                              cached_inverse, cached_pf, cached_valid,
                              ns, ne, ng, nj, nq, batch, b)
    # Apply the right-hand pair first. Keeping electron labels makes Pfaffian
    # antisymmetry carry the fermionic signs, as in CPU green_func1/green_func2.
    k = 0
    if op[7] == 2
        k = move_operator!(cfg, op[4], op[5], op[6], ne, b)
        k == 0 && return 0.0
    end
    i = move_operator!(cfg, op[1], op[2], op[3], ne, b)
    if i == 0
        k != 0 && (@inbounds cfg[k,b] = op[5])
        return 0.0
    end
    if op[1] == op[2] && (op[7] == 1 || op[4] == op[5])
        return 1.0
    end
    newlog = correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
    newip = updated_overlap!(slater, weights, cfg, factors, pivots, pfs,
                              cached_inverse, cached_pf, cached_valid,
                              i, k, ns, ne, nq, batch, b)
    @inbounds result = exp(newlog - logs[b]) * newip / overlaps[b]
    @inbounds cfg[i,b] = op[2]
    k != 0 && (@inbounds cfg[k,b] = op[5])
    return result
end

function observable_kernel!(values, ops, coeffs, one, two, ex1, ex2,
                            cfg, counts, occ, overlaps, logs, gutz, jastrow,
                            parameters, slater, weights, factors, pivots, pfs,
                            cached_inverse, cached_pf, cached_valid,
                            ns, ne, ng, nj, nq, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    energy = 0.0
    @inbounds for t in eachindex(ops)
        energy += coeffs[t] * green_value!(ops[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds values[1,b] = energy
    row = 1
    @inbounds for t in eachindex(one)
        row += 1
        values[row,b] = green_value!(one[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds for t in eachindex(two)
        row += 1
        values[row,b] = green_value!(two[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds for t in eachindex(ex1)
        row += 1
        g1 = green_value!(ex1[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, ns, ne, ng, nj, nq, batch, b)
        g2 = green_value!(ex2[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, ns, ne, ng, nj, nq, batch, b)
        values[row,b] = g1 * g2
    end
    correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
    return nothing
end

struct CUDAObservables
    values::CuMatrix{Float64}
    ops::CuVector{GPUOperator}
    coeffs::CuVector{Float64}
    one::CuVector{GPUOperator}
    two::CuVector{GPUOperator}
    ex1::CuVector{GPUOperator}
    ex2::CuVector{GPUOperator}
    inverse::CuArray{Float64,4}
    pfaffians::CuMatrix{Float64}
    valid::CuMatrix{Bool}
end

function CUDAObservables(s::CUDAState, data; measure_greens=true)
    ops, coeffs, one, two, ex1, ex2 = operator_tables(data)
    if !measure_greens
        empty!(one); empty!(two); empty!(ex1); empty!(ex2)
    end
    rows = 1 + length(one) + length(two) + length(ex1)
    return CUDAObservables(CUDA.zeros(Float64, rows, size(s.configurations, 2)),
        CuArray(ops), CuArray(coeffs), CuArray(one), CuArray(two), CuArray(ex1), CuArray(ex2),
        CUDA.zeros(Float64, size(s.configurations,2), 2s.nelec, 2s.nelec, s.nqp),
        CUDA.zeros(Float64, s.nqp, size(s.configurations,2)),
        CUDA.zeros(Bool, s.nqp, size(s.configurations,2)))
end

function CPU.gpu_observables!(s::CUDAState, o::CUDAObservables)
    batch = size(s.configurations, 2)
    @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false inverse_cache_kernel!(
        cache_view(o.inverse), o.pfaffians, o.valid, s.configurations, s.slater, factor_view(s.factors),
        pivot_view(s.pivots), s.pfaffians, s.nsite, s.nelec, s.nqp, batch)
    @cuda threads=s.threads blocks=cld(batch, s.threads) fastmath=false observable_kernel!(
        o.values, o.ops, o.coeffs, o.one, o.two, o.ex1, o.ex2,
        s.configurations, s.counts, s.occupancy, s.overlaps, s.logcorrelations,
        s.gutz, s.jastrow, s.parameters, s.slater, s.weights, factor_view(s.factors), pivot_view(s.pivots),
        s.pfaffians, cache_view(o.inverse), o.pfaffians, o.valid,
        s.nsite, s.nelec, s.ngutz, s.njastrow, s.nqp, batch)
    return o.values
end

CPU.gpu_observables_workspace(s::CUDAState, data::ExpertModeData) = CUDAObservables(s, data)

function inverse_cache_kernel!(cache, pf, valid, cfg, slater, factors, pivots, pfs,
                               ns, ne, nq, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    n = 2ne
    @inbounds for q in 1:nq
        for i in 1:n, j in 1:n
            ri = cfg[i,b] + (i > ne ? ns : 0) + 1
            rj = cfg[j,b] + (j > ne ? ns : 0) + 1
            factors[i,j,b] = -slater[ri,rj,q]
        end
        pfaffian_kernel!(pfs, factors, pivots, n, batch)
        pf[q,b] = pfs[b]
        for i in 1:n, j in 1:n
            ri = cfg[i,b] + (i > ne ? ns : 0) + 1
            rj = cfg[j,b] + (j > ne ? ns : 0) + 1
            factors[i,j,b] = -slater[ri,rj,q]
        end
        valid[q,b] = pf[q,b] != 0 && inverse!(factors, view(cache,:,:,q,:), n, b)
    end
    return nothing
end

# CalculateNewPfM2/CalculateNewPfMTwo2_real, using cached A^-1 in mathematical
# row/column indexing (CPU stores that same logical inverse in a flat C layout).
@inline function updated_overlap!(slater, weights, cfg, factors, pivots, pfs,
                                  inv, pf, valid, a, c, ns, ne, nq, batch, b)
    ip = 0.0
    n = 2ne
    @inbounds for q in 1:nq
        if !valid[q,b]
            for i in 1:n, j in 1:n
                ri = cfg[i,b] + (i > ne ? ns : 0) + 1
                rj = cfg[j,b] + (j > ne ? ns : 0) + 1
                factors[i,j,b] = -slater[ri,rj,q]
            end
            pfaffian_kernel!(pfs,factors,pivots,n,batch)
            ip += weights[q]*pfs[b]
            continue
        end
        ra = cfg[a,b] + (a > ne ? ns : 0) + 1
        ratio = 0.0
        if c == 0 || c == a
            for j in 1:n
                rj = cfg[j,b] + (j > ne ? ns : 0) + 1
                ratio -= inv[a,j,q,b]*slater[rj,ra,q]
            end
        else
            rc = cfg[c,b] + (c > ne ? ns : 0) + 1
            pa = pc = qa = qc = 0.0
            for j in 1:n
                rj = cfg[j,b] + (j > ne ? ns : 0) + 1
                va = slater[rj,ra,q]
                vc = slater[rj,rc,q]
                pa += inv[a,j,q,b]*va
                pc += inv[c,j,q,b]*va
                qa += inv[a,j,q,b]*vc
                qc += inv[c,j,q,b]*vc
            end
            cMa = 0.0
            for i in 1:n
                ri = cfg[i,b] + (i > ne ? ns : 0) + 1
                tmp = 0.0
                for j in 1:n
                    rj = cfg[j,b] + (j > ne ? ns : 0) + 1
                    tmp += inv[i,j,q,b]*slater[rj,ra,q]
                end
                cMa += slater[ri,rc,q]*tmp
            end
            ratio = inv[a,c,q,b]*slater[ra,rc,q] + inv[a,c,q,b]*cMa + pa*qc - pc*qa
        end
        ip += weights[q]*ratio*pf[q,b]
    end
    return ip
end
