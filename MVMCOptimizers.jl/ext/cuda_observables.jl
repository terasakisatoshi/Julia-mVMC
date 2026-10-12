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
                              cached_inverse, cached_pf, cached_valid, cached_transform,
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
    newlog = logs[b] + operator_correlation_delta(op, occ, gutz, jastrow, parameters, ns, ng, nj, b)
    newip = updated_overlap!(slater, weights, cfg, factors, pivots, pfs,
                              cached_inverse, cached_pf, cached_valid, cached_transform,
                              i, k, op, ns, ne, nq, batch, b)
    @inbounds result = exp(newlog - logs[b]) * newip / overlaps[b]
    @inbounds cfg[i,b] = op[2]
    k != 0 && (@inbounds cfg[k,b] = op[5])
    return result
end

function observable_kernel!(values, ops, coeffs, one, two, ex1, ex2,
                            cfg, counts, occ, overlaps, logs, gutz, jastrow,
                            parameters, slater, weights, factors, pivots, pfs,
                            cached_inverse, cached_pf, cached_valid, cached_transform,
                            ns, ne, ng, nj, nq, batch, fallback_only=false)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    if fallback_only
        allvalid = true
        @inbounds for q in 1:nq
            allvalid &= cached_valid[q,b]
        end
        allvalid && return nothing
    end
    energy = 0.0
    @inbounds for t in eachindex(ops)
        energy += coeffs[t] * green_value!(ops[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, cached_transform, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds values[1,b] = energy
    row = 1
    @inbounds for t in eachindex(one)
        row += 1
        values[row,b] = green_value!(one[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, cached_transform, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds for t in eachindex(two)
        row += 1
        values[row,b] = green_value!(two[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, cached_transform, ns, ne, ng, nj, nq, batch, b)
    end
    @inbounds for t in eachindex(ex1)
        row += 1
        g1 = green_value!(ex1[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, cached_transform, ns, ne, ng, nj, nq, batch, b)
        g2 = green_value!(ex2[t], cfg, counts, occ, overlaps, logs,
            gutz, jastrow, parameters, slater, weights, factors, pivots, pfs,
            cached_inverse, cached_pf, cached_valid, cached_transform, ns, ne, ng, nj, nq, batch, b)
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
    packed_factors::CuArray{Float64,3}
    packed_inverse::CuArray{Float64,3}
    transformed::CuArray{Float64,4}
    termvalues::CuMatrix{Float64}
    batched_inverse::Union{Nothing,BatchedInverse}
    compact::Union{Nothing,CompactTransform}
    warp_hamiltonian::Bool
end

# One-body Hamiltonian ratios already have an ordered inverse-row dot path.
# Diagonal two-body terms require only occupations. Requested Green functions
# keep the full transformed matrix, as do Hamiltonians with two moving pairs.
requires_transformed_slater(ops; measure_greens=true) = measure_greens ||
    any(op -> op[7] != 1 && (op[1] != op[2] || op[4] != op[5]), ops)

function CUDAObservables(s::CUDAState, data; measure_greens=true, compact_transform=true)
    ops, coeffs, one, two, ex1, ex2 = operator_tables(data)
    inverse_mode = get(ENV, "JULIA_MVMC_GPU_INVERSE", "batched")
    inverse_mode in ("batched", "cooperative") ||
        throw(ArgumentError("JULIA_MVMC_GPU_INVERSE must be batched or cooperative"))
    use_batched = s.nelec >= 32 && inverse_mode == "batched"
    compact_sites = use_batched && measure_greens && compact_transform ?
                    compact_transform_sites(data) : Int32[]
    use_compact = compact_transform_eligible(s,compact_sites)
    packed_batch = s.nelec >= 32 && !use_batched ? size(s.configurations,2) : 0
    transform_batch = s.nelec >= 32 && !use_compact && requires_transformed_slater(ops; measure_greens) ?
                      size(s.configurations,2) : 0
    if !measure_greens
        empty!(one); empty!(two); empty!(ex1); empty!(ex2)
    end
    warp_hamiltonian = use_batched && !measure_greens && !requires_transformed_slater(ops;measure_greens=false)
    rows = 1 + length(one) + length(two) + length(ex1)
    return CUDAObservables(CUDA.zeros(Float64, rows, size(s.configurations, 2)),
        CuArray(ops), CuArray(coeffs), CuArray(one), CuArray(two), CuArray(ex1), CuArray(ex2),
        CUDA.zeros(Float64, size(s.configurations,2), 2s.nelec, 2s.nelec, s.nqp),
        CUDA.zeros(Float64, s.nqp, size(s.configurations,2)),
        CUDA.zeros(Bool, s.nqp, size(s.configurations,2)),
        CUDA.zeros(Float64, 2s.nelec, 2s.nelec, packed_batch),
        CUDA.zeros(Float64, 2s.nelec, 2s.nelec, packed_batch),
        CUDA.zeros(Float64, transform_batch,
                   2s.nelec, 2s.nsite, s.nqp),
        CUDA.zeros(Float64, length(ops)+length(one)+length(two)+length(ex1),
                   s.nelec >= 32 ? size(s.configurations,2) : 0),
        use_batched ? BatchedInverse(s;allocate_transform=!use_compact && transform_batch>0) : nothing,
        use_compact ? CompactTransform(s,compact_sites) : nothing,warp_hamiltonian)
end

function CPU.gpu_observables!(s::CUDAState, o::CUDAObservables;sampling_cache=nothing)
    if o.compact !== nothing
        return compact_observables!(s,o,o.compact;sampling_cache)
    end
    batch = size(s.configurations, 2)
    if sampling_cache !== nothing
        sampling_cache isa GPUSamplingCache || throw(ArgumentError("invalid sampling cache"))
        observable_sampling_cache!(s,o,sampling_cache)
    elseif o.batched_inverse !== nothing
        batched_inverse_cache!(s,o,o.batched_inverse)
    elseif s.nelec >= 32
        @cuda threads=256 blocks=batch shmem=16s.nelec fastmath=false cooperative_inverse_cache_kernel!(
            cache_view(o.inverse), o.pfaffians, o.valid, s.configurations, s.slater,
            o.packed_factors, o.packed_inverse, pivot_view(s.pivots), s.nsite, s.nelec, s.nqp, batch)
    else
        @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false inverse_cache_kernel!(
            cache_view(o.inverse), o.pfaffians, o.valid, s.configurations, s.slater, factor_view(s.factors),
            pivot_view(s.pivots), s.pfaffians, s.nsite, s.nelec, s.nqp, batch)
    end
    if !isempty(o.transformed) && o.batched_inverse !== nothing
        batched_transform_cache!(s,o,o.batched_inverse)
    elseif !isempty(o.transformed) && s.nelec >= 32
        entries = batch * 2s.nelec * 2s.nsite * s.nqp
        @cuda threads=256 blocks=cld(entries,256) fastmath=false transformed_slater_kernel!(
            cache_view(o.transformed), cache_view(o.inverse), o.valid,
            s.configurations, s.slater, s.nsite, s.nelec, s.nqp, batch)
    end
    if s.nelec >= 32
        parallel_observables!(s,o)
    end
    @cuda threads=s.threads blocks=cld(batch, s.threads) fastmath=false observable_kernel!(
        o.values, o.ops, o.coeffs, o.one, o.two, o.ex1, o.ex2,
        s.configurations, s.counts, s.occupancy, s.overlaps, s.logcorrelations,
        s.gutz, s.jastrow, s.parameters, s.slater, s.weights, factor_view(s.factors), pivot_view(s.pivots),
        s.pfaffians, cache_view(o.inverse), o.pfaffians, o.valid, cache_view(o.transformed),
        s.nsite, s.nelec, s.ngutz, s.njastrow, s.nqp, batch, s.nelec >= 32)
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
                                  inv, pf, valid, transformed, a, c, op, ns, ne, nq, batch, b)
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
            ip = CUDA.add_rn(ip,CUDA.mul_rn(weights[q],pfs[b]))
            continue
        end
        ra = cfg[a,b] + (a > ne ? ns : 0) + 1
        ratio = 0.0
        if size(transformed,4) != 0
            olda = (c == a ? op[5] : op[2]) + (a > ne ? ns : 0) + 1
            if c == 0 || c == a
                ratio = -transformed_entry(transformed, inv, slater, a, ra, a, c, ra, 0,
                                           olda, 0, q, b)
            else
                rc = cfg[c,b] + (c > ne ? ns : 0) + 1
                oldc = op[5] + (c > ne ? ns : 0) + 1
                pa = transformed_entry(transformed, inv, slater, a, ra, a, c, ra, rc, olda, oldc, q, b)
                pc = transformed_entry(transformed, inv, slater, c, ra, a, c, ra, rc, olda, oldc, q, b)
                qa = transformed_entry(transformed, inv, slater, a, rc, a, c, ra, rc, olda, oldc, q, b)
                qc = transformed_entry(transformed, inv, slater, c, rc, a, c, ra, rc, olda, oldc, q, b)
                cMa = 0.0
                for i in 1:n
                    ri = cfg[i,b] + (i > ne ? ns : 0) + 1
                    tmp = transformed_entry(transformed, inv, slater, i, ra, a, c, ra, rc, olda, oldc, q, b)
                    cMa += slater[ri,rc,q] * tmp
                end
                ratio = inv[a,c,q,b]*slater[ra,rc,q] + inv[a,c,q,b]*cMa + pa*qc - pc*qa
            end
        elseif c == 0 || c == a
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
        newpf = CUDA.mul_rn(ratio,pf[q,b])
        ip = CUDA.add_rn(ip,CUDA.mul_rn(weights[q],newpf))
    end
    return ip
end

# Large matrices use a block per walker and packed scratch, so neighboring
# lanes update neighboring matrix entries rather than serializing O(n^3) work.
function cooperative_inverse_cache_kernel!(cache, pf, valid, cfg, slater, a, inv, pivots,
                                           ns, ne, nq, batch)
    b = blockIdx().x
    lane = threadIdx().x
    lanes = blockDim().x
    n = 2ne
    control = CuStaticSharedArray(Float64, 4)
    column = CuDynamicSharedArray(Float64, n)
    @inbounds for q in 1:nq
        for idx in lane:lanes:n*n
            i = (idx-1)%n+1
            j = (idx-1)÷n+1
            ri = cfg[i,b]+(i > ne ? ns : 0)+1
            rj = cfg[j,b]+(j > ne ? ns : 0)+1
            a[i,j,b] = -slater[ri,rj,q]
        end
        result = cooperative_pfaffian!(a, pivots, n, b)
        if lane == 1
            pf[q,b] = result
            control[4] = result != 0 ? 1.0 : 0.0
        end
        sync_threads()
        if control[4] != 0
            for idx in lane:lanes:n*n
                i = (idx-1)%n+1
                j = (idx-1)÷n+1
                ri = cfg[i,b]+(i > ne ? ns : 0)+1
                rj = cfg[j,b]+(j > ne ? ns : 0)+1
                a[i,j,b] = -slater[ri,rj,q]
                inv[i,j,b] = i == j ? 1.0 : 0.0
            end
            sync_threads()
            for k in 1:n
                if lane == 1
                    pivot = k
                    best = abs(a[k,k,b])
                    for i in k+1:n
                        x = abs(a[i,k,b])
                        if x > best
                            best = x
                            pivot = i
                        end
                    end
                    control[2] = pivot
                    control[3] = best
                    if best == 0
                        control[4] = 0.0
                    end
                end
                sync_threads()
                control[4] == 0 && break
                pivot = Int(control[2])
                if pivot != k
                    for j in lane:lanes:n
                        a[k,j,b],a[pivot,j,b] = a[pivot,j,b],a[k,j,b]
                        inv[k,j,b],inv[pivot,j,b] = inv[pivot,j,b],inv[k,j,b]
                    end
                end
                sync_threads()
                if lane == 1
                    control[3] = 1.0/a[k,k,b]
                end
                sync_threads()
                scale = control[3]
                for j in lane:lanes:n
                    a[k,j,b] *= scale
                    inv[k,j,b] *= scale
                end
                sync_threads()
                for i in lane:lanes:n
                    column[i] = a[i,k,b]
                end
                sync_threads()
                for idx in lane:lanes:n*n
                    i = (idx-1)%n+1
                    j = (idx-1)÷n+1
                    if i != k
                        x = column[i]
                        a[i,j,b] -= x*a[k,j,b]
                        inv[i,j,b] -= x*inv[k,j,b]
                    end
                end
                sync_threads()
            end
        end
        if lane == 1
            valid[q,b] = control[4] != 0
        end
        if control[4] != 0
            for idx in lane:lanes:n*n
                i = (idx-1)%n+1
                j = (idx-1)÷n+1
                cache[i,j,q,b] = inv[i,j,b]
            end
        end
        sync_threads()
    end
    return nothing
end

# Resident A^-1 times occupied Slater rows. Each dot keeps the original j order.
function transformed_slater_kernel!(transformed, inv, valid, cfg, slater, ns, ne, nq, batch)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    n = 2ne
    idx > batch*n*2ns*nq && return nothing
    offset = idx-1
    b = offset%batch+1
    offset ÷= batch
    i = offset%n+1
    offset ÷= n
    r = offset%(2ns)+1
    q = offset÷(2ns)+1
    result = 0.0
    @inbounds if valid[q,b]
        for j in 1:n
            rj = cfg[j,b]+(j > ne ? ns : 0)+1
            result += inv[i,j,q,b]*slater[rj,r,q]
        end
    end
    @inbounds transformed[i,r,q,b] = result
    return nothing
end

@inline function transformed_entry(transformed, inv, slater, i, r, a, c,
                                   ra, rc, olda, oldc, q, b)
    @inbounds result = transformed[i,r,q,b] + inv[i,a,q,b]*(slater[ra,r,q]-slater[olda,r,q])
    if c != 0 && c != a
        @inbounds result += inv[i,c,q,b]*(slater[rc,r,q]-slater[oldc,r,q])
    end
    return result
end
