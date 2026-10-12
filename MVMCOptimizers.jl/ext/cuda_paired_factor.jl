# Paired skew elimination, adapted from the upper DSKTF2 translation in
# cuda_cooperative_pf.jl (the same PFAPACK BSD license applies).
# P*A*P' = U*D*U', with unit upper U and independent 2-by-2 skew blocks D.
# Eliminating two columns per step supplies both the Pfaffian and inverse:
# Pf(A) = det(P)*product(D[2k-1,2k]), A^-1 = P'*U'^-1*D^-1*U^-1*P.
# The trailing update keeps ordered FP64 arithmetic; only the inverse product
# uses FMA. Strongly cancelling projections use the independent normal path.
@inline function paired_skew_factor!(A, pivots, diagonal, n, b)
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
    @inbounds for k in n:-2:2
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
        # Retain the original (a + b*c) - d*e update order; the two
        # source columns remain unchanged until every trailing update finishes.
        width = k - 2
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
        sync_threads()
        for j in lane:stride:width
            A[j, j, b] = 0.0
            u=A[j,kk,b];v=A[j,k,b]
            A[j,kk,b]=CUDA.mul_rn(v,alpha)
            A[j,k,b]=-CUDA.mul_rn(u,alpha)
        end
        if lane == 1
            pivots[kk, b] = kp
            diagonal[k÷2,b]=A[kk,k,b]
            A[kk,k,b]=0.0
        end
        sync_threads()
    end
    if lane == 1
        pf = 1.0
        @inbounds for i in 1:n÷2
            pf = CUDA.mul_rn(pf, diagonal[i,b])
        end
        scalars[3] = CUDA.mul_rn(scalars[2], pf)
    end
    sync_threads()
    return scalars[3]
end

function paired_skew_factor_kernel!(values, A, pivots, diagonal, n)
    b = blockIdx().x
    pf = paired_skew_factor!(A, pivots, diagonal, n, b)
    if threadIdx().x == 1
        @inbounds values[b] = pf
    end
    return nothing
end


function paired_identity_kernel!(M,n,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    index>n*n*count && return nothing
    i=(index-1)%n+1;j=((index-1)÷n)%n+1
    @inbounds M[index]=i==j ? 1.0 : 0.0
    return nothing
end
function paired_permutation_kernel!(perm,piv,n,count)
    b=(blockIdx().x-1)*blockDim().x+threadIdx().x
    b>count && return nothing
    @inbounds begin
        for i in 1:n;perm[i,b]=i;end
        for j in 1:n
            target=piv[j,b]
            perm[j,b],perm[target,b]=perm[target,b],perm[j,b]
        end
    end
    return nothing
end
function paired_permute_inverse_kernel!(output,input,perm,n,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    index>n*n*count && return nothing
    i=(index-1)%n+1;j=((index-1)÷n)%n+1;b=(index-1)÷(n*n)+1
    @inbounds output[i,j,b]=input[perm[i,b],perm[j,b],b]
    return nothing
end
function paired_reciprocal_diagonal_kernel!(output,input)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>length(input) && return nothing
    @inbounds output[i]=CUDA.div_rn(1.0,input[i])
    return nothing
end
function paired_tridiagonal_solve!(C,M,diagonal,n,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    index>n*n*count && return nothing
    i=(index-1)%n+1;j=((index-1)÷n)%n+1;b=(index-1)÷(n*n)+1
    @inbounds C[i,j,b]=iseven(i) ? CUDA.mul_rn(M[i-1,j,b],diagonal[i÷2,b]) : CUDA.mul_rn(-M[i+1,j,b],diagonal[(i+1)÷2,b])
    return nothing
end
function paired_triangular_skew_product!(output,MT,C,n,count)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    j=(blockIdx().y-1)*blockDim().y+threadIdx().y
    b=blockIdx().z
    (i>n || j>n || i>j) && return nothing
    value=0.0
    @inbounds if i<j
        for k in 1:i
            value=muladd(MT[i,k,b],C[k,j,b],value)
        end
        output[i,j,b]=value
        output[j,i,b]=-value
    else
        output[i,i,b]=0.0
    end
    return nothing
end

