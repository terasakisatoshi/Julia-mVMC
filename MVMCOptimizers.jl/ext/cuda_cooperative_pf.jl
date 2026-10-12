# Cooperative translation of the upper/normal DSKTF2 kernel above. The same
# PFAPACK BSD license applies. Each block owns a matrix; independent entries
# of its trailing update run in parallel, without changing scalar arithmetic.
# Explicit round-to-nearest intrinsics prevent GPU multiply/add contraction:
# cancellation between projected Pfaffians can magnify local rounding errors.
@inline function cooperative_pfaffian!(A, pivots, n, b, ::Val{column_warps}=Val(true)) where {column_warps}
    lane = threadIdx().x
    stride = blockDim().x
    scalars = CuStaticSharedArray(Float64, 3) # alpha, sign, result
    control = CuStaticSharedArray(Int32, 2)  # pivot, zero even column
    @inbounds for i in lane:stride:n
        pivots[i, b] = i
    end
    if lane == 1
        scalars[2] = 1.0
        control[2] = 0
    end
    sync_threads()
    @inbounds for k in n:-1:2
        kk = k - 1
        if lane == 1
            kp = 1
            colmax = abs(A[1, k, b])
            for j in 2:k-1
                candidate = abs(A[j, k, b])
                if candidate > colmax
                    kp = j
                    colmax = candidate
                end
            end
            control[1] = colmax == 0.0 ? 0 : kp
            if colmax == 0.0 && iseven(k)
                control[2] = 1
            end
        end
        sync_threads()
        control[2] != 0 && return 0.0
        kp = Int(control[1])
        if kp == 0
            if lane == 1
                pivots[kk, b] = kk
            end
            sync_threads()
            continue
        end
        if kp != kk
            for j in lane:stride:kp-1
                A[j, kk, b], A[j, kp, b] = A[j, kp, b], A[j, kk, b]
            end
            for j in kp+lane:stride:kk-1
                A[j, kk, b], A[kp, j, b] = A[kp, j, b], A[j, kk, b]
            end
            for j in k+lane-1:stride:n
                A[kk, j, b], A[kp, j, b] = A[kp, j, b], A[kk, j, b]
            end
            sync_threads()
            for j in kp+lane-1:stride:kk-1
                A[j, kk, b] = -A[j, kk, b]
            end
            for j in kp+lane:stride:kk-1
                A[kp, j, b] = -A[kp, j, b]
            end
            if lane == 1
                scalars[2] = -scalars[2]
            end
            sync_threads()
        end
        if lane == 1
            scalars[1] = CUDA.div_rn(1.0, A[kk, k, b])
        end
        sync_threads()
        alpha = scalars[1]
        # Both traversals retain the original (a + b*c) - d*e order; the two
        # source columns remain unchanged until every trailing update finishes.
        width = k - 2
        if column_warps
            # One warp owns a column. Its lanes touch neighboring rows without
            # scanning the discarded lower triangle or dividing per entry.
            warp = (lane-1) ÷ 32+1
            warp_lane = (lane-1) % 32+1
            warps = stride ÷ 32
            for j in warp:warps:width
                if A[j,k,b] != 0.0 || A[j,kk,b] != 0.0
                    temp1 = CUDA.mul_rn(alpha,A[j,kk,b])
                    temp2 = CUDA.mul_rn(alpha,A[j,k,b])
                    for i in warp_lane:32:j-1
                        first = CUDA.add_rn(A[i,j,b],CUDA.mul_rn(A[i,k,b],temp1))
                        A[i,j,b] = CUDA.add_rn(first,-CUDA.mul_rn(A[i,kk,b],temp2))
                    end
                end
            end
        else
            for index in lane:stride:width*width
                i = (index - 1) % width + 1
                j = (index - 1) ÷ width + 1
                if i < j && (A[j, k, b] != 0.0 || A[j, kk, b] != 0.0)
                    temp1 = CUDA.mul_rn(alpha, A[j, kk, b])
                    temp2 = CUDA.mul_rn(alpha, A[j, k, b])
                    first = CUDA.add_rn(A[i, j, b], CUDA.mul_rn(A[i, k, b], temp1))
                    A[i, j, b] = CUDA.add_rn(first, -CUDA.mul_rn(A[i, kk, b], temp2))
                end
            end
        end
        sync_threads()
        for j in lane:stride:width
            A[j, j, b] = 0.0
            A[j, k, b] = CUDA.mul_rn(A[j, k, b], alpha)
        end
        if lane == 1
            pivots[kk, b] = kp
        end
        sync_threads()
    end
    if lane == 1
        pf = 1.0
        @inbounds for i in 1:2:n-1
            pf = CUDA.mul_rn(pf, A[i, i+1, b])
        end
        scalars[3] = CUDA.mul_rn(scalars[2], pf)
    end
    sync_threads()
    return scalars[3]
end

function cooperative_pfaffian_kernel!(values, A, pivots, n)
    b = blockIdx().x
    pf = cooperative_pfaffian!(A, pivots, n, b)
    if threadIdx().x == 1
        @inbounds values[b] = pf
    end
    return nothing
end

function cooperative_pfaffians!(values, A, pivots; threads=128)
    n = size(A, 1)
    batch = size(A, 3)
    batch == 0 && return values
    @cuda threads=threads blocks=batch fastmath=false cooperative_pfaffian_kernel!(values, A, pivots, n)
    return values
end
