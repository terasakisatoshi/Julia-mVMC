# Paired upper skew elimination, adapted from PFAPACK DSKTF2 via
# cuda_cooperative_pf.jl; the same PFAPACK BSD license/provenance applies.
# P*A*P' = U*D*U', with unit upper U and 2-by-2 skew blocks D.
# Keep up to K rank-two changes pending in shared memory. Materialize each
# pivot column before reading it, permute every pending vector with the
# matrix, and flush the remaining leading block once per panel. Four vectors
# per pair retain both raw columns and their rounded alpha-scaled columns.
# FMA and mirrored entries after pivot permutations reassociate arithmetic;
# independent CPU residuals/Pfaffians and physical-result gates cover it.
@inline function paired_pending_entry(A,i,j,b,raw,n,pending)
    @inbounds value=A[i,j,b]
    @inbounds for p in 1:pending
        offset=(p-1)*4n
        uj=raw[offset+j];vj=raw[offset+n+j]
        if uj!=0.0 || vj!=0.0
            t1=raw[offset+2n+j];t2=raw[offset+3n+j]
            ui=raw[offset+i];vi=raw[offset+n+i]
            value=muladd(-vi,t2,muladd(ui,t1,value))
        end
    end
    return value
end

@inline function paired_skew_factor!(A, pivots, diagonal, n, b, capacity::Val{K}) where {K}
    lane = threadIdx().x
    stride = blockDim().x
    scalars = CuStaticSharedArray(Float64, 3) # alpha, sign, result
    control = CuStaticSharedArray(Int32, 2)  # pivot, zero even column
    raw=CuDynamicSharedArray(Float64,4n*K)
    pending=0
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
        if pending>0
            for i in lane:stride:k-1
                A[i,k,b]=paired_pending_entry(A,i,k,b,raw,n,pending)
            end
            sync_threads()
        end
        if lane <= 32
            kp = Int32(1)
            colmax = abs(A[1,k,b])
            for j in lane:32:k-1
                candidate = abs(A[j,k,b])
                if candidate > colmax || (candidate == colmax && j < kp)
                    colmax = candidate; kp = Int32(j)
                end
            end
            for offset in (16,8,4,2,1)
                value = CUDA.shfl_down_sync(typemax(UInt32),colmax,offset)
                index = CUDA.shfl_down_sync(typemax(UInt32),kp,offset)
                if value > colmax || (value == colmax && index < kp)
                    colmax = value; kp = index
                end
            end
            if lane == 1
                control[1] = colmax == 0.0 ? 0 : kp
                if colmax == 0.0 && iseven(k)
                    control[2] = 1
                end
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
            for index in lane:stride:4pending
                p=(index-1)÷4+1;component=(index-1)%4
                offset=(p-1)*4n+component*n
                raw[offset+kk],raw[offset+kp]=raw[offset+kp],raw[offset+kk]
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
        if pending>0
            for i in lane:stride:kk-1
                A[i,kk,b]=paired_pending_entry(A,i,kk,b,raw,n,pending)
            end
        end
        if lane == 1
            scalars[1] = CUDA.div_rn(1.0, A[kk, k, b])
        end
        sync_threads()
        alpha = scalars[1]
        width=k-2
        next=pending+1
        offset=(next-1)*4n
        for j in lane:stride:width
            u=A[j,kk,b];v=A[j,k,b]
            raw[offset+j]=v;raw[offset+n+j]=u
            A[j,j,b]=0.0
            tv=CUDA.mul_rn(v,alpha);tu=CUDA.mul_rn(u,alpha)
            raw[offset+2n+j]=tu;raw[offset+3n+j]=tv
            A[j,kk,b]=tv;A[j,k,b]=-tu
        end
        if lane==1
            pivots[kk,b]=kp
            diagonal[k÷2,b]=A[kk,k,b]
            A[kk,k,b]=0.0
        end
        sync_threads()
        pending=next
        if pending==K
            warp=(lane-1)÷32+1;warplane=(lane-1)%32+1;warps=stride÷32
            for j in warp:warps:width
                for i in warplane:32:j-1
                    A[i,j,b]=paired_pending_entry(A,i,j,b,raw,n,pending)
                end
            end
            sync_threads()
            pending=0
        end
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

function paired_skew_factor_kernel!(values,A,pivots,diagonal,n,capacity::Val{K}) where {K}
    b=blockIdx().x
    pf=paired_skew_factor!(A,pivots,diagonal,n,b,capacity)
    if threadIdx().x==1
        @inbounds values[b]=pf
    end
    return nothing
end

# Invert U*D*U' by extending its leading block two columns at a time.
# [B -B*V; -V'*B Dpair^-1+V'*B*V] is the next inverse block.
function paired_growing_inverse!(B,U,diagonal,n)
    b=blockIdx().x;lane=threadIdx().x;stride=blockDim().x
    shared=CuDynamicSharedArray(Float64,4n)
    @inbounds for k in 1:2:n
        m=k-1
        for i in lane:stride:m
            shared[i]=U[i,k,b];shared[n+i]=U[i,k+1,b]
        end
        sync_threads()
        for i in lane:stride:m
            x=0.0;y=0.0
            for j in 1:m
                value=B[i,j,b]
                x=muladd(value,shared[j],x)
                y=muladd(value,shared[n+j],y)
            end
            shared[2n+i]=x;shared[3n+i]=y
        end
        sync_threads()
        for i in lane:stride:m
            x=shared[2n+i];y=shared[3n+i]
            B[i,k,b]=-x;B[k,i,b]=x
            B[i,k+1,b]=-y;B[k+1,i,b]=y
        end
        if lane==1
            value=-1.0/diagonal[(k+1)÷2,b]
            for i in 1:m
                value=muladd(shared[i],shared[3n+i],value)
            end
            B[k,k,b]=0.0;B[k+1,k+1,b]=0.0
            B[k,k+1,b]=value;B[k+1,k,b]=-value
        end
        sync_threads()
    end
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
