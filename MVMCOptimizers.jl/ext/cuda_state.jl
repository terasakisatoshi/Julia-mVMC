# Host ownership of device buffers. Only setup/reporting copies are allowed.
struct CUDAState
    nsite::Int
    nelec::Int
    nproj::Int
    nqp::Int
    threads::Int
    parameters::CuVector{Float64}
    basis::Union{CuMatrix{Float64},CUDA.cuSPARSE.CuSparseMatrixCSC{Float64,Int32}}
    slater::CuArray{Float64,3}
    weights::CuVector{Float64}
    configurations::CuMatrix{Int32} # zero-based sites, up electrons followed by down
    occupancy::CuMatrix{Int32}
    counts::CuMatrix{Int32}
    gutz::CuVector{Int32}
    jastrow::CuMatrix{Int32}
    ngutz::Int
    njastrow::Int
    factors::CuArray{Float64,3}
    pivots::CuMatrix{Int32}
    pfaffians::CuVector{Float64}
    overlaps::CuVector{Float64}
    logcorrelations::CuVector{Float64}
    localspins::CuVector{Int32}
    accepted::CuVector{Int32}
    cooperative_factors::CuArray{Float64,3}
    slater_assembly::Union{Nothing,SlaterAssembly}
end

# Walker is the leading physical dimension: adjacent lanes access adjacent
# doubles. Logical views retain the C row/column indexing in scalar kernels.
factor_view(a::CuArray{Float64,3}) = PermutedDimsArray(a, (2,3,1))
pivot_view(a::CuMatrix{Int32}) = PermutedDimsArray(a, (2,1))
cache_view(a::CuArray{Float64,4}) = PermutedDimsArray(a, (2,3,4,1))

function validate_cuda_data(data)
    CPU.validate_supported_modpara(data.modpara)
    data.i_flg_orbital_general == 0 || throw(ArgumentError("CUDA requires sz-conserved orbitals"))
    !CPU.get_all_complex_flag(data) || throw(ArgumentError("CUDA runner requires real parameters"))
    !CPU.has_rbm_terms(data) || throw(ArgumentError("CUDA RBM is not supported"))
    isempty(data.doublon_holon_2site_params) && isempty(data.doublon_holon_4site_params) ||
        throw(ArgumentError("CUDA doublon-holon projection is not supported"))
    isempty(data.opt_trans) || throw(ArgumentError("CUDA variational OptTrans is not supported"))
    data.modpara.lanczos_mode == 0 || throw(ArgumentError("CUDA Lanczos is not supported"))
    data.modpara.nsplit_size == 1 || throw(ArgumentError("CUDA runner is serial; MPI split is not supported"))
    data.modpara.two_sz == 0 || throw(ArgumentError("CUDA requires 2Sz=0"))
    data.modpara.nex_update_path in 0:2 || throw(ArgumentError("CUDA requires NExUpdatePath in 0:2"))
    isempty(data.doublon_holon_2site_indices) && isempty(data.doublon_holon_4site_indices) ||
        throw(ArgumentError("CUDA doublon-holon indices are not supported"))
    return nothing
end

# Small lattices retain the independent CPU unit-parameter derivative basis.
# Large lattices build CSC coefficients directly from C four-block geometry.
# Neither setup path evaluates samples or performs Monte Carlo work on the host.
function slater_basis(data, np, ns, ne, nq, nproj)
    model = deepcopy(data)
    model.modpara.nmp_trans = max(1, abs(model.modpara.nmp_trans))
    Parsers.init_qp_weight!(model)
    state = CPU.VMCOptimizationState(ns, ne, nproj, np, nq, 1, false, false)
    basis = zeros(Float64, 4 * ns * ns * nq, np)
    unit = zeros(ComplexF64, np)
    for p in nproj+1:np
        fill!(unit, 0)
        unit[p] = 1
        CPU.unpack_parameters!(model, unit)
        CPU.update_slater_elm_fcmp!(model, state)
        basis[:, p] .= real.(state.slater_matrix.slater_elm)
    end
    return basis, real.(model.qp_weights.qp_full_weight)
end

function CPU.gpu_state(data::ExpertModeData; configurations::AbstractMatrix{<:Integer}, threads::Integer=32, initialize_wavefunctions::Bool=true)
    CUDA.functional() || throw(ArgumentError("A functional CUDA device is required"))
    validate_cuda_data(data)
    threads in (32,64,128,256) || throw(ArgumentError("walker block size must be 32, 64, 128 or 256"))
    ns = data.modpara.nsite
    ne = data.modpara.nelec == 0 ? (data.modpara.nlocspin + data.modpara.ncond) ÷ 2 : data.modpara.nelec
    0 < ne <= ns || throw(ArgumentError("CUDA requires 0 < NElec <= Nsite"))
    size(configurations, 1) == 2ne || throw(DimensionMismatch("configuration electron count"))
    batch = size(configurations, 2)
    batch > 0 || throw(ArgumentError("at least one walker is required"))
    all(x -> 0 <= x < ns, configurations) || throw(ArgumentError("configuration site bounds"))
    for b in 1:batch, spin in 0:1
        length(unique(configurations[spin*ne+1:(spin+1)*ne, b])) == ne ||
            throw(ArgumentError("duplicate occupied sites within a spin sector"))
    end
    np = CPU.count_total_parameters(data)
    nproj = Parsers.projection_layout(data).n_proj
    nq = max(1, data.modpara.nsp_gauss_leg) * max(1, abs(data.modpara.nmp_trans)) * max(1, data.n_qp_opt_trans)
    assembly = SlaterAssembly(data,nproj,ns,nq)
    if ns >= 64
        basis = sparse_slater_basis(assembly,np)
        weights = real.(data.qp_weights.qp_full_weight)
    else
        dense, weights = slater_basis(data,np,ns,ne,nq,nproj)
        basis = CuArray(dense)
    end
    params = CPU.pack_parameters(data)
    all(isreal, params) || throw(ArgumentError("CUDA parameters must be real"))
    gutz = data.n_gutzwiller_idx == 0 ? fill(Int32(-1), ns) : Int32.(data.gutzwiller_idx)
    jastrow = data.n_jastrow_idx == 0 ? fill(Int32(-1), ns, ns) : Int32.(data.jastrow_idx)
    size(jastrow) == (ns, ns) && length(gutz) == ns || throw(DimensionMismatch("projection indices"))
    s = CUDAState(ns, ne, nproj, nq, Int(threads), CuArray(real.(params)), basis,
        CUDA.zeros(Float64, 2ns, 2ns, nq), CuArray(weights), CuArray(Int32.(configurations)),
        CUDA.zeros(Int32, 2ns, batch), CUDA.zeros(Int32, nproj, batch), CuArray(gutz),
        CuArray(jastrow), data.n_gutzwiller_idx, data.n_jastrow_idx,
        CUDA.zeros(Float64, batch, 2ne, 2ne), CUDA.zeros(Int32, batch, 2ne),
        CUDA.zeros(Float64, batch), CUDA.zeros(Float64, batch), CUDA.zeros(Float64, batch),
        CuArray(Int32.(CPU.get_loc_spn_array(data))), CUDA.zeros(Int32, batch),
        CUDA.zeros(Float64, 2ne, 2ne, ns >= 64 ? batch : 0),
        assembly)
    refresh_slater!(s)
    initialize_wavefunctions && CPU.gpu_wavefunctions!(s)
    return s
end

function refresh_slater!(s::CUDAState)
    if s.slater_assembly === nothing
        mul!(vec(s.slater),s.basis,s.parameters)
    else
        assemble_slater!(s,s.slater_assembly)
    end
    return s.slater
end

@inline function correlation!(counts, occ, cfg, gutz, jastrow, parameters,
                             ns, ne, ng, nj, b)
    @inbounds for i in 1:2ns
        occ[i, b] = 0
    end
    @inbounds for i in 1:2ne
        r = cfg[i, b] + (i > ne ? ns : 0) + 1
        occ[r, b] = 1
    end
    @inbounds for p in 1:ng+nj
        counts[p, b] = 0
    end
    @inbounds for i in 1:ns
        p = gutz[i] + 1
        if 1 <= p <= ng
            counts[p, b] += occ[i, b] * occ[ns+i, b]
        end
        xi = occ[i, b] + occ[ns+i, b] - 1
        for j in i+1:ns
            p = jastrow[i, j] + 1
            if 1 <= p <= nj
                xj = occ[j, b] + occ[ns+j, b] - 1
                counts[ng+p, b] += xi * xj
            end
        end
    end
    logcorr = 0.0
    @inbounds for p in 1:ng+nj
        logcorr += parameters[p] * counts[p, b]
    end
    return logcorr
end

@inline function projected_overlap!(slater, weights, cfg, factors, pivots, pfs,
                                    ns, ne, nq, batch, b)
    result = 0.0
    @inbounds for q in 1:nq
        for i in 1:2ne, j in 1:2ne
            ri = cfg[i, b] + (i > ne ? ns : 0) + 1
            rj = cfg[j, b] + (j > ne ? ns : 0) + 1
            factors[i, j, b] = -slater[ri, rj, q]
        end
        pfaffian_kernel!(pfs, factors, pivots, 2ne, batch)
        result = CUDA.add_rn(result,CUDA.mul_rn(weights[q],pfs[b]))
    end
    return result
end

function wavefunction_kernel!(overlaps, logs, cfg, counts, occ, gutz, jastrow,
                              parameters, slater, weights, factors, pivots, pfs,
                              ns, ne, ng, nj, nq, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    @inbounds logs[b] = correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
    @inbounds overlaps[b] = projected_overlap!(slater, weights, cfg, factors, pivots, pfs,
                                               ns, ne, nq, batch, b)
    return nothing
end

function CPU.gpu_wavefunctions!(s::CUDAState)
    batch = size(s.configurations, 2)
    if s.nsite >= 64
        @cuda threads=128 blocks=batch fastmath=false cooperative_wavefunction_kernel!(cooperative_state(s))
        return s
    end
    @cuda threads=s.threads blocks=cld(batch, s.threads) fastmath=false wavefunction_kernel!(
        s.overlaps, s.logcorrelations, s.configurations, s.counts, s.occupancy,
        s.gutz, s.jastrow, s.parameters, s.slater, s.weights, factor_view(s.factors),
        pivot_view(s.pivots), s.pfaffians, s.nsite, s.nelec, s.ngutz, s.njastrow, s.nqp, batch)
    return s
end

# A fixed-draw, symmetric Metropolis proposal. Its rejection self-loops differ
# from C's draw-until-empty proposal. Shared tapes verify numerics/decisions;
# CPU SFMT trajectories remain a separate unchanged contract.
function sampling_kernel!(cfg, counts, occ, overlaps, logs, accepted, localspins,
                          gutz, jastrow, parameters, slater, weights, factors,
                          pivots, pfs, tape, exchange_fraction, ns, ne, ng, nj, nq, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    oldip = @inbounds overlaps[b]
    oldlog = @inbounds logs[b]
    naccepted = Int32(0)
    @inbounds for step in 1:size(tape, 3)
        i = Int(tape[1, b, step] % UInt32(ne)) + 1
        spin = tape[2, b, step] < UInt32(0x80000000) ? 0 : 1
        i += spin * ne
        ri = cfg[i, b]
        exchange = Float64(tape[4, b, step]) / 4294967296.0 < exchange_fraction
        j = 0
        rj = 0
        valid = true
        if exchange
            j = Int(tape[3, b, step] % UInt32(ne)) + 1 + (1-spin)*ne
            rj = cfg[j, b]
            valid = ri != rj
            for k in 1:ne
                if cfg[k+spin*ne, b] == rj || cfg[k+(1-spin)*ne, b] == ri
                    valid = false
                end
            end
        else
            rj = Int(tape[3, b, step] % UInt32(ns))
            valid = localspins[ri+1] == 0 && localspins[rj+1] == 0
            for k in 1:ne
                if cfg[k+spin*ne, b] == rj
                    valid = false
                end
            end
        end
        valid || continue
        cfg[i, b] = rj
        exchange && (cfg[j, b] = ri)
        newlog = correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
        newip = projected_overlap!(slater, weights, cfg, factors, pivots, pfs, ns, ne, nq, batch, b)
        logratio = 2 * (newlog - oldlog + log(abs(newip) + 1e-100) - log(abs(oldip) + 1e-100))
        u = Float64(tape[5, b, step]) / 4294967296.0
        if newip != 0 && log(u) < logratio
            oldip = newip
            oldlog = newlog
            naccepted += Int32(1)
        else
            cfg[i, b] = ri
            exchange && (cfg[j, b] = rj)
        end
    end
    @inbounds begin
        overlaps[b] = oldip
        logs[b] = correlation!(counts, occ, cfg, gutz, jastrow, parameters, ns, ne, ng, nj, b)
        accepted[b] = naccepted
    end
    return nothing
end

function CPU.gpu_sample!(s::CUDAState; steps::Integer, exchange_fraction::Real=0,
                         tape=nothing, cache=nothing, empty_destinations::Bool=false,
                         incremental_correlation::Bool=false)
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))
    0 <= exchange_fraction <= 1 || throw(ArgumentError("exchange_fraction must be in [0, 1]"))
    if cache===nothing
        empty_destinations || incremental_correlation ?
            throw(ArgumentError("cached sampler options require a sampling cache")) : nothing
    else
        cache isa GPUSamplingCache && cache.owner===s ||
            throw(ArgumentError("sampling cache belongs to a different state"))
    end
    batch = size(s.configurations, 2)
    draws = tape === nothing ? CUDA.rand(UInt32, 5, batch, steps) : tape
    draws isa CuArray{UInt32,3} || throw(ArgumentError("tape must be a device UInt32 array"))
    size(draws) == (5, batch, steps) || throw(DimensionMismatch("random tape dimensions"))
    cache !== nothing && return sample_with_cache!(s,cache,draws,exchange_fraction;
        empty_destinations,incremental_correlation)
    if s.nsite >= 64
        @cuda threads=128 blocks=batch fastmath=false cooperative_sampling_kernel!(
            cooperative_state(s), draws, Float64(exchange_fraction))
        return s
    end
    @cuda threads=s.threads blocks=cld(batch, s.threads) fastmath=false sampling_kernel!(
        s.configurations, s.counts, s.occupancy, s.overlaps, s.logcorrelations,
        s.accepted, s.localspins, s.gutz, s.jastrow, s.parameters, s.slater,
        s.weights, factor_view(s.factors), pivot_view(s.pivots), s.pfaffians, draws, Float64(exchange_fraction),
        s.nsite, s.nelec, s.ngutz, s.njastrow, s.nqp, batch)
    return s
end
