# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Private real translation of RuQing Xu's invert.tcc, via PfaPack Utu2.
# See source-adjacent ordinary_real_inverse.NOTICE and .LICENSE.
# Unit-upper DTRTI2 column algorithm, independently expressed in Julia.
# LAPACK DTRTI2 calls DTRMV(U,N,U) then DSCAL(-1) for each column.
# The diagonal and lower triangle must remain untouched: they store other
# parts of the skew factorization. Only independent i entries are vectorized.
function _ordinary_unit_upper_trtri_real!(A::AbstractMatrix{Float64})
    Base.require_one_based_indexing(A)
    n = size(A, 1)
    size(A, 2) == n || throw(DimensionMismatch("triangular inverse must be square"))
    @inbounds for j in 2:n
        for k in 1:j-1
            temp = A[k,j]
            if temp != 0.0
                @simd ivdep for i in 1:k-1
                    A[i,j] += temp * A[i,k]
                end
            end
        end
        @simd ivdep for i in 1:j-1
            A[i,j] = -A[i,j]
        end
    end
    return A
end

function _ordinary_sktdsmx_real_c_order!(n::Int, vt::Vector{Float64},
                                        B::AbstractMatrix{Float64}, C::AbstractMatrix{Float64})
    Base.require_one_based_indexing(vt, B, C)
    n >= 2 && iseven(n) || throw(ArgumentError("solve requires a positive even dimension"))
    size(B) == (n,n) && size(C) == (n,n) || throw(DimensionMismatch("solve workspace dimensions"))
    length(vt) >= n-1 || throw(DimensionMismatch("solve vector workspace"))
    @inbounds for j in 1:n
        C[2,j] = B[1,j] / -vt[1]
    end
    @inbounds for i in 2:2:n-2
        for j in 1:n
            C[i+2,j] = (B[i+1,j] - C[i,j] * vt[i]) / -vt[i+1]
        end
    end
    @inbounds for j in 1:n
        C[n-1,j] = B[n,j] / vt[n-1]
    end
    @inbounds for i in n-3:-2:1
        for j in 1:n
            C[i,j] = (B[i+1,j] + C[i+2,j] * vt[i+1]) / vt[i]
        end
    end
    return nothing
end

function _ordinary_utu2inv_real_c_order!(n::Int, A::AbstractMatrix{Float64}, ldA::Int,
                                        ipiv::Vector{<:Integer}, vt::Vector{Float64},
                                        M::AbstractMatrix{Float64}, ldM::Int)
    Base.require_one_based_indexing(A, ipiv, vt, M)
    n >= 2 && iseven(n) || throw(ArgumentError("inverse requires a positive even dimension"))
    size(A) == (n,n) && size(M) == (n,n) || throw(DimensionMismatch("inverse workspace dimensions"))
    ldA >= n && ldM >= n || throw(DimensionMismatch("inverse leading dimensions"))
    length(ipiv) >= n && length(vt) >= n-1 || throw(DimensionMismatch("inverse vector workspace"))
    all(i -> 1 <= ipiv[i] <= n, 1:n) || throw(ArgumentError("inverse pivot outside matrix"))
    # Same existing Julia/C full pipeline; only the scalar solver differs.
    fill!(M, 0.0)
    @inbounds for i in 1:n
        M[i,i] = 1.0
    end
    # Optimize the measured matrix sizes. Tiny SR-CG systems amplify provider
    # roundoff: on macOS Julia 1.11, a 2.8e-17 inverse difference at n=6 changed
    # the first parameter update. Retain their established LAPACK provider.
    if 32 <= n <= 64
        _ordinary_unit_upper_trtri_real!(@view(A[1:n-1,2:n]))
    else
        LinearAlgebra.LAPACK.trtri!('U', 'U', @view(A[1:n-1,2:n]))
    end
    @inbounds for j in 1:n-2, i in 1:j
        M[i,j+1] = A[i,j+2]
    end
    @inbounds for i in 1:n-1
        vt[i] = -A[i,i+1]
    end
    _ordinary_sktdsmx_real_c_order!(n, vt, M, A)
    @inbounds for j in 1:n
        target = ipiv[j]
        if target != j
            for i in 1:n
                A[i,j], A[i,target] = A[i,target], A[i,j]
            end
        end
    end
    LinearAlgebra.BLAS.trmm!('L', 'U', 'T', 'U', 1.0, M, A)
    @inbounds for i in 1:n
        target = ipiv[i]
        if target != i
            for j in 1:n
                A[i,j], A[target,j] = A[target,j], A[i,j]
            end
        end
    end
    return nothing
end
