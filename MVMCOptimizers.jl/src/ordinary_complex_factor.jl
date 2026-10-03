# Private upper/normal ZSKTF2 translation from Michael Wimmer's PFAPACK.
# See ordinary_complex_factor.NOTICE and the source-adjacent LAPACK license.
# No dependency method replacement, @fastmath, @turbo, or fused updates.
function _ordinary_zsktf2_c_order!(A::AbstractMatrix{ComplexF64}, ipiv::Vector{<:Integer})
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("Matrix must be square"))
    length(ipiv) >= n || throw(DimensionMismatch("pivot workspace is too short"))
    for i in 1:n
        ipiv[i] = i
    end
    info = 0
    for k in n:-1:2
        # IZAMAX uses abs(real)+abs(imag), with the first maximum retained.
        kp = 1
        pivot_score = abs(real(A[1,k])) + abs(imag(A[1,k]))
        for i in 2:k-1
            score = abs(real(A[i,k])) + abs(imag(A[i,k]))
            if score > pivot_score
                kp = i
                pivot_score = score
            end
        end
        colmax = abs(A[kp,k])
        kk = k - 1
        if colmax == 0.0
            info == 0 && (info = kk)
            kp = kk
        end
        if kp != kk
            for i in 1:kp-1
                A[i,kk], A[i,kp] = A[i,kp], A[i,kk]
            end
            for i in kp+1:kk-1
                A[i,kk], A[kp,i] = A[kp,i], A[i,kk]
            end
            for j in k:n
                A[kk,j], A[kp,j] = A[kp,j], A[kk,j]
            end
            for i in kp:kk-1
                A[i,kk] = (-1.0+0.0im) * A[i,kk]
            end
            for j in kp+1:kk-1
                A[kp,j] = (-1.0+0.0im) * A[kp,j]
            end
        end
        if colmax != 0.0
            # The received ZSKR2 alpha is GNU/LLVM C division, not inv(pivot).
            alpha = _ordinary_complex_divide(1.0+0.0im, A[kk,k])
            for j in 1:k-2
                if A[j,k] != 0.0+0.0im || A[j,kk] != 0.0+0.0im
                    temp1 = alpha * A[j,kk]
                    temp2 = alpha * A[j,k]
                    for i in 1:j-1
                        A[i,j] = (A[i,j] + A[i,k]*temp1) - A[i,kk]*temp2
                    end
                end
                A[j,j] = 0.0+0.0im
            end
            # Preserve ZSCAL alpha-before-vector complex multiplication.
            for i in 1:k-2
                A[i,k] = alpha * A[i,k]
            end
        end
        ipiv[kk] = kp
    end
    return info
end
