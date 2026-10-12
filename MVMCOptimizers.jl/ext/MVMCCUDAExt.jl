module MVMCCUDAExt

using CUDA
using MVMCOptimizers
using LinearAlgebra
using MVMCOptimizers.MVMCExpertModeParsers: ExpertModeData
const Parsers = MVMCOptimizers.MVMCExpertModeParsers
const CPU = MVMCOptimizers

@inline pivot_score(x::Float64) = abs(x)
@inline pivot_score(x::ComplexF64) = abs(real(x)) + abs(imag(x))
@inline pf_kernel_mul(x::Float64,y::Float64) = CUDA.mul_rn(x,y)
@inline pf_kernel_mul(x,y) = x*y
@inline pf_kernel_add(x::Float64,y::Float64) = CUDA.add_rn(x,y)
@inline pf_kernel_add(x,y) = x+y
@inline pf_kernel_reciprocal(x::Float64) = CUDA.div_rn(1.0,x)
@inline pf_kernel_reciprocal(x) = one(x)/x

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
        alpha = pf_kernel_reciprocal(A[kk,k,b])
        for j in 1:k-2
            if A[j, k, b] != 0 || A[j, kk, b] != 0
                temp1 = pf_kernel_mul(alpha,A[j,kk,b])
                temp2 = pf_kernel_mul(alpha,A[j,k,b])
                for i in 1:j-1
                    A[i,j,b] = pf_kernel_add(pf_kernel_add(A[i,j,b],
                        pf_kernel_mul(A[i,k,b],temp1)),-pf_kernel_mul(A[i,kk,b],temp2))
                end
            end
            A[j, j, b] = zero(T)
        end
        for j in 1:k-2
            A[j,k,b] = pf_kernel_mul(A[j,k,b],alpha)
        end
        pivots[kk, b] = kp
    end
    pf = one(T)
    @inbounds for i in 1:2:n-1
        pf = pf_kernel_mul(pf,A[i,i+1,b])
    end
    @inbounds values[b] = pf_kernel_mul(sign,pf)
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

include("cuda_cooperative_pf.jl")
include("cuda_slater_assembly.jl")
include("cuda_state.jl")
include("cuda_cooperative_sampling.jl")
include("cuda_correlation_delta.jl")
include("cuda_tiled_publication.jl")
include("cuda_batched_inverse.jl")
include("cuda_compact_transform.jl")
include("cuda_observables.jl")
include("cuda_sampling_correlation.jl")
include("cuda_delayed_sampling.jl")
include("cuda_sampling_update.jl")
include("cuda_observable_sampling_cache.jl")
include("cuda_sampling_inverse.jl")
include("cuda_paired_factor.jl")
include("cuda_paired_sampling_inverse.jl")
include("cuda_parallel_observables.jl")
include("cuda_warp_observables.jl")
include("cuda_sr.jl")
include("cuda_sparse_contraction.jl")
include("cuda_initial_sampling.jl")
include("cuda_initial_parallel_qp.jl")
include("cuda_sr_solver.jl")
include("cuda_parameter_sync.jl")
include("cuda_runners.jl")

end
