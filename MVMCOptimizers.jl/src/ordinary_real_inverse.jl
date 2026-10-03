# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Private real translation of RuQing Xu's invert.tcc, via PfaPack Utu2.
# See source-adjacent ordinary_real_inverse.NOTICE and .LICENSE.
function _ordinary_sktdsmx_real_c_order!(n::Int, vt::Vector{Float64},
                                        B::AbstractMatrix{Float64}, C::AbstractMatrix{Float64})
    for j in 1:n
        C[2,j] = B[1,j] / -vt[1]
    end
    for i in 2:2:n-2
        for j in 1:n
            C[i+2,j] = (B[i+1,j] - C[i,j] * vt[i]) / -vt[i+1]
        end
    end
    for j in 1:n
        C[n-1,j] = B[n,j] / vt[n-1]
    end
    for i in n-3:-2:1
        for j in 1:n
            C[i,j] = (B[i+1,j] + C[i+2,j] * vt[i+1]) / vt[i]
        end
    end
    return nothing
end

function _ordinary_utu2inv_real_c_order!(n::Int, A::AbstractMatrix{Float64}, ldA::Int,
                                        ipiv::Vector{<:Integer}, vt::Vector{Float64},
                                        M::AbstractMatrix{Float64}, ldM::Int)
    n >= 2 && iseven(n) || throw(ArgumentError("inverse requires a positive even dimension"))
    size(A) == (n,n) && size(M) == (n,n) || throw(DimensionMismatch("inverse workspace dimensions"))
    ldA >= n && ldM >= n || throw(DimensionMismatch("inverse leading dimensions"))
    length(ipiv) >= n && length(vt) >= n-1 || throw(DimensionMismatch("inverse vector workspace"))
    all(i -> 1 <= ipiv[i] <= n, 1:n) || throw(ArgumentError("inverse pivot outside matrix"))
    # Same existing Julia/C full pipeline; only the scalar solver differs.
    fill!(M, 0.0)
    for i in 1:n
        M[i,i] = 1.0
    end
    LinearAlgebra.LAPACK.trtri!('U', 'U', @view(A[1:n-1,2:n]))
    for j in 1:n-2, i in 1:j
        M[i,j+1] = A[i,j+2]
    end
    for i in 1:n-1
        vt[i] = -A[i,i+1]
    end
    _ordinary_sktdsmx_real_c_order!(n, vt, M, A)
    for j in 1:n
        target = ipiv[j]
        if target != j
            for i in 1:n
                A[i,j], A[i,target] = A[i,target], A[i,j]
            end
        end
    end
    LinearAlgebra.BLAS.trmm!('L', 'U', 'T', 'U', 1.0, M, A)
    for i in 1:n
        target = ipiv[i]
        if target != i
            for j in 1:n
                A[i,j], A[target,j] = A[target,j], A[i,j]
            end
        end
    end
    return nothing
end
