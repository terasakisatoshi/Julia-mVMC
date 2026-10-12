module MVMCCUDAExt

using CUDA
using MVMCOptimizers
using LinearAlgebra
using MVMCOptimizers.MVMCExpertModeParsers: ExpertModeData
const Parsers = MVMCOptimizers.MVMCExpertModeParsers
const CPU = MVMCOptimizers

@inline pivot_score(x::Float64) = abs(x)
@inline pivot_score(x::ComplexF64) = abs(real(x)) + abs(imag(x))

# Upper/normal DSKTF2 scalar operation order, from ordinary_real_factor.jl.
# That file's PFAPACK BSD license and provenance also apply to this translation.
# Each lane owns one matrix: no per-pivot launches or host decisions.
function pfaffian_kernel!(values, A, pivots, n, batch)
    b = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b > batch && return nothing
    T = eltype(A)
    sign = one(T)
    @inbounds for i in 1:n
        pivots[i, b] = i
    end
    @inbounds for k in n:-1:2
        kp = 1
        colmax = pivot_score(A[1, k, b])
        for j in 2:k-1
            candidate = pivot_score(A[j, k, b])
            if candidate > colmax
                kp = j
                colmax = candidate
            end
        end
        kk = k - 1
        if colmax == 0
            if iseven(k)
                values[b] = zero(T)
                return nothing
            end
            # Only even tridiagonal pivots enter the Pfaffian product. A zero
            # odd column is valid for nonsingular block-skew matrices.
            pivots[kk,b] = kk
            continue
        end
        if kp != kk
            for j in 1:kp-1
                A[j, kk, b], A[j, kp, b] = A[j, kp, b], A[j, kk, b]
            end
            for j in kp+1:kk-1
                A[j, kk, b], A[kp, j, b] = A[kp, j, b], A[j, kk, b]
            end
            for j in k:n
                A[kk, j, b], A[kp, j, b] = A[kp, j, b], A[kk, j, b]
            end
            for j in kp:kk-1
                A[j, kk, b] = -A[j, kk, b]
            end
            for j in kp+1:kk-1
                A[kp, j, b] = -A[kp, j, b]
            end
            sign = -sign
        end
        alpha = one(T) / A[kk, k, b]
        for j in 1:k-2
            if A[j, k, b] != 0 || A[j, kk, b] != 0
                temp1 = alpha * A[j, kk, b]
                temp2 = alpha * A[j, k, b]
                for i in 1:j-1
                    A[i, j, b] = (A[i, j, b] + A[i, k, b] * temp1) -
                                  A[i, kk, b] * temp2
                end
            end
            A[j, j, b] = zero(T)
        end
        for j in 1:k-2
            A[j, k, b] *= alpha
        end
        pivots[kk, b] = kp
    end
    pf = one(T)
    @inbounds for i in 1:2:n-1
        pf *= A[i, i+1, b]
    end
    @inbounds values[b] = sign * pf
    return nothing
end

function MVMCOptimizers.gpu_pfaffians!(values::CuVector{T}, A::CuArray{T,3},
                                      pivots::CuMatrix{Int32}) where {T<:Union{Float64,ComplexF64}}
    n, m, batch = size(A)
    n == m && n > 0 && iseven(n) ||
        throw(ArgumentError("Pfaffian requires a positive even square matrix"))
    length(values) == batch || throw(DimensionMismatch("Pfaffian batch length"))
    size(pivots) == (n, batch) || throw(DimensionMismatch("pivot batch dimensions"))
    batch == 0 && return values
    @cuda threads=128 blocks=cld(batch, 128) fastmath=false pfaffian_kernel!(values, A, pivots, n, batch)
    return values
end

include("cuda_state.jl")
include("cuda_observables.jl")
include("cuda_sr.jl")
include("cuda_runners.jl")

end
