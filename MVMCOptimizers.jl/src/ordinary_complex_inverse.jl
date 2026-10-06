# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Private complex upper/full translation of xrq-phys/Pfaffine invert.tcc,
# via PfaPack's Julia Utu2 port. See ordinary_complex_inverse.NOTICE/LICENSE.
include("ordinary_complex_quotient.jl")

function _ordinary_sktdsmx_complex_c_order!(n::Int, vt::Vector{ComplexF64},
                                           B::AbstractMatrix{ComplexF64}, C::AbstractMatrix{ComplexF64})
    for j in 1:n
        C[2,j] = _ordinary_complex_divide(B[1,j],-vt[1])
    end
    for i in 2:2:n-2, j in 1:n
        C[i+2,j] = _ordinary_complex_divide(B[i+1,j]-C[i,j]*vt[i],-vt[i+1])
    end
    for j in 1:n
        C[n-1,j] = _ordinary_complex_divide(B[n,j],vt[n-1])
    end
    for i in n-3:-2:1, j in 1:n
        C[i,j] = _ordinary_complex_divide(B[i+1,j]+C[i+2,j]*vt[i+1],vt[i])
    end
    return nothing
end

function _ordinary_utu2inv_complex_c_order!(n::Int, A::AbstractMatrix{ComplexF64}, ldA::Int,
                                           ipiv::Vector{<:Integer}, vt::Vector{ComplexF64},
                                           M::AbstractMatrix{ComplexF64}, ldM::Int)
    n >= 2 && iseven(n) || throw(ArgumentError("inverse requires a positive even dimension"))
    size(A) == (n,n) && size(M) == (n,n) || throw(DimensionMismatch("inverse workspace dimensions"))
    ldA >= n && ldM >= n || throw(DimensionMismatch("inverse leading dimensions"))
    length(ipiv) >= n && length(vt) >= n-1 || throw(DimensionMismatch("inverse vector workspace"))
    all(i -> 1 <= ipiv[i] <= n, 1:n) || throw(ArgumentError("inverse pivot outside matrix"))
    fill!(M,0.0+0.0im)
    for i in 1:n
        M[i,i] = 1.0+0.0im
    end
    LinearAlgebra.LAPACK.trtri!('U','U',@view(A[1:n-1,2:n]))
    for j in 1:n-2, i in 1:j
        M[i,j+1] = A[i,j+2]
    end
    for i in 1:n-1
        vt[i] = -A[i,i+1]
    end
    _ordinary_sktdsmx_complex_c_order!(n,vt,M,A)
    for j in 1:n
        target=ipiv[j]
        if target != j
            for i in 1:n
                A[i,j],A[i,target] = A[i,target],A[i,j]
            end
        end
    end
    LinearAlgebra.BLAS.trmm!('L','U','T','U',1.0+0.0im,M,A)
    for i in 1:n
        target=ipiv[i]
        if target != i
            for j in 1:n
                A[i,j],A[target,j] = A[target,j],A[i,j]
            end
        end
    end
    return nothing
end
