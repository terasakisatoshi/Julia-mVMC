# Analytic Pfaffian log derivatives and device SR moments.
# Gauss-Jordan supplies A^-1, independently of the CPU LTL inverse algorithm.
@inline function inverse!(a, inv, n, b)
    @inbounds for j in 1:n, i in 1:n
        inv[i,j,b] = i == j ? 1.0 : 0.0
    end
    @inbounds for k in 1:n
        pivot = k
        best = abs(a[k,k,b])
        for i in k+1:n
            x = abs(a[i,k,b])
            if x > best
                pivot = i
                best = x
            end
        end
        best == 0 && return false
        if pivot != k
            for j in 1:n
                a[k,j,b], a[pivot,j,b] = a[pivot,j,b], a[k,j,b]
                inv[k,j,b], inv[pivot,j,b] = inv[pivot,j,b], inv[k,j,b]
            end
        end
        scale = 1.0 / a[k,k,b]
        for j in 1:n
            a[k,j,b] *= scale
            inv[k,j,b] *= scale
        end
        for i in 1:n
            if i != k
                x = a[i,k,b]
                for j in 1:n
                    a[i,j,b] -= x * a[k,j,b]
                    inv[i,j,b] -= x * inv[k,j,b]
                end
            end
        end
    end
    return true
end

function derivative_kernel!(sensitivity, inv, cfg, counts, overlaps, slater, weights,
                            factors, pivots, pfs, ns, ne, nproj, nq, np, batch)
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
        pf = pfs[b]
        for i in 1:n, j in 1:n
            ri = cfg[i,b] + (i > ne ? ns : 0) + 1
            rj = cfg[j,b] + (j > ne ? ns : 0) + 1
            factors[i,j,b] = -slater[ri,rj,q]
        end
        nonsingular = pf != 0 && inverse!(factors, inv, n, b)
        for j in 2:n, i in 1:j-1
            ri = cfg[i,b] + (i > ne ? ns : 0) + 1
            rj = cfg[j,b] + (j > ne ? ns : 0) + 1
            cofactor = 0.0
            if nonsingular
                cofactor = pf * inv[j,i,b]
            elseif n == 2
                cofactor = 1.0
            else
                # A singular individual QP sector may still have nonzero
                # derivatives inside a nonzero projected sum. Use exact minors.
                ii = 0
                for k in 1:n
                    if k != i && k != j
                        ii += 1
                        rki = cfg[k,b] + (k > ne ? ns : 0) + 1
                        jj = 0
                        for l in 1:n
                            if l != i && l != j
                                jj += 1
                                rlj = cfg[l,b] + (l > ne ? ns : 0) + 1
                                factors[ii,jj,b] = -slater[rki,rlj,q]
                            end
                        end
                    end
                end
                pfaffian_kernel!(pfs, factors, pivots, n-2, batch)
                cofactor = iseven(i+j+1) ? pfs[b] : -pfs[b]
            end
            row = ri + (rj-1)*2ns + (q-1)*4ns*ns
            sensitivity[row,b] = -weights[q] * cofactor / overlaps[b]
        end
    end
    return nothing
end

struct CUDASR
    derivatives::CuMatrix{Float64}
    inverse::CuArray{Float64,3}
    flags::CuVector{Bool}
    sensitivity::CuMatrix{Float64}
    alpha::CuVector{Float64}
    beta::CuVector{Float64}
end
function CUDASR(s::CUDAState, data)
    np = length(s.parameters)
    flags = [CPU.get_opt_flag_for_parameter(data, 2(p-1)) != 0 for p in 1:np]
    return CUDASR(CUDA.zeros(Float64, np, size(s.configurations,2)),
                  similar(s.factors), CuArray(flags),
                  CUDA.zeros(Float64, size(s.basis,1), size(s.configurations,2)),
                  CuArray([1.0]), CuArray([0.0]))
end
function derivatives!(s::CUDAState, sr::CUDASR)
    batch = size(s.configurations,2)
    fill!(sr.sensitivity, 0)
    @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false derivative_kernel!(
        sr.sensitivity, factor_view(sr.inverse), s.configurations, s.counts, s.overlaps,
        s.slater, s.weights, factor_view(s.factors), pivot_view(s.pivots), s.pfaffians,
        s.nsite, s.nelec, s.nproj, s.nqp, length(s.parameters), batch)
    return finish_derivatives!(s, sr)
end

function finish_derivatives!(s::CUDAState, sr::CUDASR)
    batch = size(s.configurations,2)
    # gemm! accepts resident alpha/beta pointers. Generic mul!/gemmEx! in CUDA
    # 6.4 uploads two scalar CuRefs on each call, visible in CUPTI traces.
    CUDA.cuBLAS.gemm!('T', 'N', sr.alpha, s.basis, sr.sensitivity, sr.beta, sr.derivatives)
    if s.nproj > 0
        entries = s.nproj * batch
        @cuda threads=256 blocks=cld(entries,256) projection_derivative_kernel!(
            sr.derivatives, s.counts, s.nproj, batch)
    end
    return sr.derivatives
end

function cached_derivative_kernel!(sensitivity, inv, pf, valid, cfg, overlaps,
                                   slater, weights, factors, pivots, pfs,
                                   ns, ne, nq, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    n = 2ne
    @inbounds for q in 1:nq, j in 2:n, i in 1:j-1
        cofactor = 0.0
        if valid[q,b]
            cofactor = pf[q,b]*inv[j,i,q,b]
        elseif n == 2
            cofactor = 1.0
        else
            ii = 0
            for k in 1:n
                if k != i && k != j
                    ii += 1
                    rk = cfg[k,b] + (k > ne ? ns : 0) + 1
                    jj = 0
                    for l in 1:n
                        if l != i && l != j
                            jj += 1
                            rl = cfg[l,b] + (l > ne ? ns : 0) + 1
                            factors[ii,jj,b] = -slater[rk,rl,q]
                        end
                    end
                end
            end
            pfaffian_kernel!(pfs,factors,pivots,n-2,batch)
            cofactor = iseven(i+j+1) ? pfs[b] : -pfs[b]
        end
        ri = cfg[i,b] + (i > ne ? ns : 0) + 1
        rj = cfg[j,b] + (j > ne ? ns : 0) + 1
        row = ri + (rj-1)*2ns + (q-1)*4ns*ns
        sensitivity[row,b] = -weights[q]*cofactor/overlaps[b]
    end
    return nothing
end

function derivatives!(s::CUDAState, sr::CUDASR, o::CUDAObservables)
    batch = size(s.configurations,2)
    fill!(sr.sensitivity,0)
    @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false cached_derivative_kernel!(
        sr.sensitivity, cache_view(o.inverse), o.pfaffians, o.valid, s.configurations,
        s.overlaps, s.slater, s.weights, factor_view(s.factors), pivot_view(s.pivots),
        s.pfaffians, s.nsite, s.nelec, s.nqp, batch)
    return finish_derivatives!(s,sr)
end

function projection_derivative_kernel!(O, counts, nproj, batch)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index > nproj * batch && return nothing
    p = (index - 1) % nproj + 1
    b = (index - 1) ÷ nproj + 1
    @inbounds O[p,b] = Float64(counts[p,b])
    return nothing
end

function sr_moments(O, energies)
    batch = size(O,2)
    meanO = vec(sum(O; dims=2)) ./ batch
    meanE = sum(energies; dims=1) ./ batch
    OO = (O * transpose(O)) ./ batch
    HO = (O * energies) ./ batch
    covariance = OO - meanO * transpose(meanO)
    return (; meanO, meanE, OO, HO, covariance)
end

function sr_system_kernel!(S, rhs, variance, flags, meanO, meanE, HO, cutoff, delta, dt, np)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i > np && return nothing
    @inbounds begin
        active_i = flags[i] && variance[i] > cutoff
        rhs[i] = active_i ? -2dt * (HO[i] - meanE[1]*meanO[i]) : 0.0
        for j in 1:np
            active_j = flags[j] && variance[j] > cutoff
            if !active_i || !active_j
                S[i,j] = i == j ? 1.0 : 0.0
            elseif i == j
                S[i,j] *= 1.0 + delta
            end
        end
    end
    return nothing
end

function sr_update!(s::CUDAState, sr::CUDASR, moments, modpara)
    S = copy(moments.covariance)
    variance = diag(S)
    # One scalar at an optimization boundary. Neither samples nor matrices
    # leave the device; cuSOLVER factorization and parameter updates stay there.
    cutoff = maximum(abs, variance) * modpara.dsr_opt_red_cut
    rhs = similar(s.parameters)
    np = length(rhs)
    @cuda threads=128 blocks=cld(np,128) fastmath=false sr_system_kernel!(S, rhs,
        variance, sr.flags, moments.meanO, moments.meanE, moments.HO, cutoff,
        modpara.dsr_opt_sta_del, modpara.dsr_opt_step_dt, np)
    update = cholesky!(Hermitian(S, :U)) \ rhs
    s.parameters .+= update
    return update
end
