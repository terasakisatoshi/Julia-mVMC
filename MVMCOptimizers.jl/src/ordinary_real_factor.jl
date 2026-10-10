# Ordinary real upper/normal DSKTF2, translated from Michael Wimmer's
# PFAPACK (LAPACK-style modified BSD-3); redistribution follows the
# original LAPACK copyright notices and full license reproduced with the
# source-adjacent ordinary_real_factor.LICENSE. See dsktf2.f:208-260 and
# dskr2.f:162-178. This private port does not modify PfaPack methods.
function _ordinary_dsktf2_c_order!(A::AbstractMatrix{Float64}, ipiv::Vector{<:Integer})
    Base.require_one_based_indexing(A, ipiv)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("Matrix must be square"))
    length(ipiv) >= n || throw(DimensionMismatch("pivot workspace is too short"))
    # Shape and pivot capacity are checked once. Every matrix index below is
    # bounded by n; preserve the scalar C update order without per-element
    # SubArray bounds checks in the cubic rank-2 loop.
    @inbounds for i in 1:n
        ipiv[i] = i
    end
    info = 0
    @inbounds for k in n:-1:2
        kp = 1
        colmax = abs(A[1, k])
        for j in 2:k-1
            candidate = abs(A[j, k])
            if candidate > colmax
                kp = j
                colmax = candidate
            end
        end
        kk = k - 1
        if colmax == 0.0
            info == 0 && (info = kk)
            kp = kk
        end
        if kp != kk
            for j in 1:kp-1
                A[j, kk], A[j, kp] = A[j, kp], A[j, kk]
            end
            for j in kp+1:kk-1
                A[j, kk], A[kp, j] = A[kp, j], A[j, kk]
            end
            for j in k:n
                A[kk, j], A[kp, j] = A[kp, j], A[kk, j]
            end
            for j in kp:kk-1
                A[j, kk] = -A[j, kk]
            end
            for j in kp+1:kk-1
                A[kp, j] = -A[kp, j]
            end
        end
        if colmax != 0.0
            alpha = 1.0 / A[kk, k]
            for j in 1:k-2
                # DSKR2 skips the products when both vector entries are zero.
                if A[j, k] != 0.0 || A[j, kk] != 0.0
                    temp1 = alpha * A[j, kk]
                    temp2 = alpha * A[j, k]
                    @simd for i in 1:j-1
                        # Deliberately no @fastmath/@turbo or fused muladd.
                        A[i, j] = (A[i, j] + A[i, k] * temp1) - A[i, kk] * temp2
                    end
                end
                A[j, j] = 0.0
            end
            for j in 1:k-2
                A[j, k] *= alpha
            end
        end
        ipiv[kk] = kp
    end
    return info
end
