# Sampling boundaries factor all walkers together through native Float64 LU.
# Rare in-kernel refactors keep the independent cooperative fallback.
struct BatchedSamplingInverse
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
end
function BatchedSamplingInverse(s::CUDAState)
    n,count=2s.nelec,size(s.configurations,2)*s.nqp
    # Both refresh branches gather every LU entry before native factorization.
    lu=CuArray{Float64}(undef,n,n,count)
    inverse=similar(lu)
    return BatchedSamplingInverse(lu,similar(lu),inverse,
        CuArray([pointer(lu,(i-1)*n*n+1) for i=1:count]),
        CuArray([pointer(inverse,(i-1)*n*n+1) for i=1:count]),
        CUDA.zeros(Int32,n,count),CUDA.zeros(Int32,n,count),CUDA.zeros(Float64,count),
        CUDA.zeros(Int32,count),CUDA.zeros(Int32,count))
end
function publish_sampling_boundary!(s,c)
    b=(blockIdx().x-1)*blockDim().x+threadIdx().x
    b>size(s.configurations,2) && return nothing
    ip=0.0
    @inbounds for q=1:s.nqp
        ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],c.pfaffians[q,b]))
    end
    @inbounds begin
        s.overlaps[b]=ip
        s.logcorrelations[b]=correlation!(s.counts,s.occupancy,s.configurations,
            s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
        c.accepted_since_refactor[b]=0
    end
    return nothing
end
function refresh_sampling_state!(s,cache::GPUSamplingCache,work::BatchedSamplingInverse;
                                 initial_pfaffians=nothing)
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    n,batch=2s.nelec,size(s.configurations,2)
    count=batch*s.nqp
    if initial_pfaffians === nothing
        @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false fill_batched_slater!(
            work.lu,work.factors,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
        @cuda threads=128 blocks=count maxregs=64 fastmath=false cooperative_pfaffian_kernel!(
            work.pfaffians,work.factors,work.pf_pivots,n)
    else
        initial_pfaffians isa CuMatrix{Float64} && size(initial_pfaffians)==(s.nqp,batch) ||
            throw(DimensionMismatch("initial Pfaffian table"))
        @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false fill_sampling_lu!(
            work.lu,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
        @cuda threads=128 blocks=cld(count,128) fastmath=false pack_initial_pfaffians!(
            work.pfaffians,initial_pfaffians,batch,count)
    end
    sampling_lu_inverse!(work,n,count)
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false publish_batched_inverse!(
        cache.inverse,cache.pfaffians,cache.valid,work.inverse,work.pfaffians,
        work.lu_info,work.inverse_info,n,batch,count)
    @cuda threads=32 blocks=cld(batch,32) fastmath=false publish_sampling_boundary!(
        cooperative_state(s),sampling_cache_state(cache))
    cache.initialized=true
    return cache
end

function fill_sampling_lu!(lu,cfg,slater,ns,ne,batch,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    n=2ne
    index>n*n*count && return nothing
    i=(index-1)%n+1
    j=((index-1)÷n)%n+1
    m=(index-1)÷(n*n)+1
    b=(m-1)%batch+1
    q=(m-1)÷batch+1
    @inbounds begin
        ri=cfg[i,b]+(i>ne ? ns : 0)+1
        rj=cfg[j,b]+(j>ne ? ns : 0)+1
        lu[i,j,m]=-slater[ri,rj,q]
    end
    return nothing
end
function pack_initial_pfaffians!(packed,initial,batch,count)
    m=(blockIdx().x-1)*blockDim().x+threadIdx().x
    m>count && return nothing
    b=(m-1)%batch+1
    q=(m-1)÷batch+1
    @inbounds packed[m]=initial[q,b]
    return nothing
end

# Shared native fallback; its status arrays retain cuBLAS singularity reports.
function sampling_lu_inverse!(work::BatchedSamplingInverse,n,count)
    handle=CUDA.cuBLAS.handle()
    CUDA.cuBLAS.cublasDgetrfBatched(handle,n,work.lu_ptrs,n,work.pivots,work.lu_info,count)
    CUDA.cuBLAS.cublasDgetriBatched(handle,n,work.lu_ptrs,n,work.pivots,
        work.inverse_ptrs,n,work.inverse_info,count)
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false skew_inverse_kernel!(work.inverse,n,count)
    return nothing
end
