# Packed batched LU inversion uses device pointer tables prepared at setup.
# Calling the native wrappers directly avoids rebuilding/uploading those tables
# on each observable evaluation. Both singularity-info arrays stay on device.
struct BatchedInverse
    lu::CuArray{Float64,3}
    factors::CuArray{Float64,3}
    inverse::CuArray{Float64,3}
    lu_ptrs::CuVector{CuPtr{Float64}}
    inverse_ptrs::CuVector{CuPtr{Float64}}
    pivots::CuMatrix{Int32}
    pf_pivots::CuMatrix{Int32}
    pfaffians::CuVector{Float64}
    lu_info::CuVector{Int32}
    inverse_info::CuVector{Int32}
    slater_rows::CuArray{Float64,3}
    transformed::CuArray{Float64,3}
    alpha::CuVector{Float64}
    beta::CuVector{Float64}
    target_sites::CuMatrix{Int32}
    target_map::CuMatrix{Int32}
end
function BatchedInverse(s::CUDAState; allocate_transform::Bool=true)
    n = 2s.nelec
    count = size(s.configurations,2)*s.nqp
    lu = CUDA.zeros(Float64,n,n,count)
    inverse = similar(lu)
    lu_ptrs = CuArray([pointer(lu,(i-1)*n*n+1) for i in 1:count])
    inverse_ptrs = CuArray([pointer(inverse,(i-1)*n*n+1) for i in 1:count])
    return BatchedInverse(lu,similar(lu),inverse,lu_ptrs,inverse_ptrs,
        CUDA.zeros(Int32,n,count),CUDA.zeros(Int32,n,count),CUDA.zeros(Float64,count),
        CUDA.zeros(Int32,count),CUDA.zeros(Int32,count),
        CUDA.zeros(Float64,n,allocate_transform ? 2s.nsite-n : 0,count),
        CUDA.zeros(Float64,n,allocate_transform ? 2s.nsite-n : 0,count),
        CuArray([1.0]),CuArray([0.0]),CUDA.zeros(Int32,2s.nsite-n,size(s.configurations,2)),
        CUDA.zeros(Int32,2s.nsite,size(s.configurations,2)))
end
function fill_batched_slater!(lu, factors, cfg, slater, ns, ne, batch, count)
    index = (blockIdx().x-1)*blockDim().x+threadIdx().x
    n = 2ne
    index > n*n*count && return nothing
    i = (index-1)%n+1
    j = ((index-1)÷n)%n+1
    m = (index-1)÷(n*n)+1
    b = (m-1)%batch+1
    q = (m-1)÷batch+1
    @inbounds begin
        ri = cfg[i,b]+(i > ne ? ns : 0)+1
        rj = cfg[j,b]+(j > ne ? ns : 0)+1
        x = -slater[ri,rj,q]
        lu[i,j,m] = x
        factors[i,j,m] = x
    end
    return nothing
end
function publish_batched_inverse!(cache, pf, valid, inv, pfs, lu_info, inverse_info, n, batch, count)
    index = (blockIdx().x-1)*blockDim().x+threadIdx().x
    index > n*n*count && return nothing
    i = (index-1)%n+1
    j = ((index-1)÷n)%n+1
    m = (index-1)÷(n*n)+1
    b = (m-1)%batch+1
    q = (m-1)÷batch+1
    @inbounds begin
        cache[i,j,q,b] = inv[i,j,m]
        if i == 1 && j == 1
            pf[q,b] = pfs[m]
            valid[q,b] = pfs[m] != 0 && lu_info[m] == 0 && inverse_info[m] == 0
        end
    end
    return nothing
end
function batched_inverse_cache!(s::CUDAState,o,work::BatchedInverse)
    n = 2s.nelec
    batch = size(s.configurations,2)
    count = batch*s.nqp
    entries = n*n*count
    @cuda threads=256 blocks=cld(entries,256) fastmath=false fill_batched_slater!(
        work.lu,work.factors,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
    @cuda threads=128 blocks=count maxregs=64 fastmath=false cooperative_pfaffian_kernel!(
        work.pfaffians,work.factors,work.pf_pivots,n)
    CUDA.cuBLAS.cublasDgetrfBatched(CUDA.cuBLAS.handle(),n,work.lu_ptrs,n,
        work.pivots,work.lu_info,count)
    CUDA.cuBLAS.cublasDgetriBatched(CUDA.cuBLAS.handle(),n,work.lu_ptrs,n,
        work.pivots,work.inverse_ptrs,n,work.inverse_info,count)
    @cuda threads=256 blocks=cld(entries,256) fastmath=false skew_inverse_kernel!(work.inverse,n,count)
    @cuda threads=256 blocks=cld(entries,256) fastmath=false publish_batched_inverse!(
        cache_view(o.inverse),o.pfaffians,o.valid,work.inverse,work.pfaffians,
        work.lu_info,work.inverse_info,n,batch,count)
    return nothing
end

# The inverse of a nonsingular skew matrix is skew. C/PfaPack inversion stores
# the opposite triangle by sign; projecting generic LU inversion removes its
# symmetric roundoff before exchange ratios amplify cancellation errors.
function skew_inverse_kernel!(inverse,n,count)
    index = (blockIdx().x-1)*blockDim().x+threadIdx().x
    index > n*n*count && return nothing
    i = (index-1)%n+1
    j = ((index-1)÷n)%n+1
    m = (index-1)÷(n*n)+1
    @inbounds if i < j
        x = 0.5*(inverse[i,j,m]-inverse[j,i,m])
        inverse[i,j,m] = x
        inverse[j,i,m] = -x
    elseif i == j
        inverse[i,j,m] = 0.0
    end
    return nothing
end

function fill_batched_slater_rows!(rows, targets, cfg, slater, ns, ne, batch, count)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    n = 2ne
    width = size(rows,2)
    idx > n*width*count && return nothing
    j = (idx-1)%n+1
    column = ((idx-1)÷n)%width+1
    m = (idx-1)÷(n*width)+1
    b = (m-1)%batch+1
    q = (m-1)÷batch+1
    @inbounds begin
        rj = cfg[j,b]+(j > ne ? ns : 0)+1
        r = targets[column,b]
        rows[j,column,m] = slater[rj,r,q]
    end
    return nothing
end
function publish_batched_transform!(transformed, packed, target_map, n, ns, batch, count)
    idx = (blockIdx().x-1)*blockDim().x+threadIdx().x
    idx > n*2ns*count && return nothing
    i = (idx-1)%n+1
    r = ((idx-1)÷n)%(2ns)+1
    m = (idx-1)÷(n*2ns)+1
    b = (m-1)%batch+1
    q = (m-1)÷batch+1
    @inbounds begin
        column = target_map[r,b]
        transformed[i,r,q,b] = column > 0 ? packed[i,column,m] : (i == -column ? -1.0 : 0.0)
    end
    return nothing
end
function batched_transform_cache!(s,o,work::BatchedInverse)
    n = 2s.nelec
    batch = size(s.configurations,2)
    count = batch*s.nqp
    @cuda threads=128 blocks=cld(batch,128) target_sites_kernel!(
        work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
    entries = length(work.slater_rows)
    if entries > 0
        @cuda threads=256 blocks=cld(entries,256) fastmath=false fill_batched_slater_rows!(
            work.slater_rows,work.target_sites,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
        CUDA.cuBLAS.gemm_strided_batched!('N','N',work.alpha,work.inverse,
            work.slater_rows,work.beta,work.transformed)
    end
    @cuda threads=(32,8) blocks=(cld(n*2s.nsite,32),cld(batch,32),s.nqp) fastmath=false tiled_publish_batched_transform!(
        cache_view(o.transformed),work.transformed,work.target_map,n,s.nsite,batch,count)
    return nothing
end

# Occupied columns obey A^-1*S_occ = -I. Only unoccupied destinations need GEMM.
function target_sites_kernel!(targets, mapping, cfg, ns, ne, batch)
    b = (blockIdx().x-1)*blockDim().x+threadIdx().x
    b > batch && return nothing
    @inbounds begin
        for r in 1:2ns
            mapping[r,b] = 0
        end
        for j in 1:2ne
            r = cfg[j,b]+(j > ne ? ns : 0)+1
            mapping[r,b] = -j
        end
        column = 0
        for r in 1:2ns
            if mapping[r,b] == 0
                column += 1
                mapping[r,b] = column
                targets[column,b] = r
            end
        end
    end
    return nothing
end
