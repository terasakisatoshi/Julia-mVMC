# Workspace owned by one Opt run; all device buffers and pointer tables remain
# alive through their asynchronous cuBLAS consumers. No process-wide cache.
struct PairedSamplingInverse
    native::BatchedSamplingInverse
    diagonal::CuMatrix{Float64}
    inverse_diagonal::CuMatrix{Float64}
    C::CuArray{Float64,3}       # D^-1 * U^-1
    MT::CuArray{Float64,3}      # transpose(U^-1), for coalesced product reads
    product::CuArray{Float64,3} # inverse before undoing pivot permutations
    permutation::CuMatrix{Int32}
    fallback::CuVector{Bool}
    factor_ptrs::CuVector{CuPtr{Float64}}
    M_ptrs::CuVector{CuPtr{Float64}}
    one::CuVector{Float64}
end
PairedSamplingInverse(s::CUDAState) = PairedSamplingInverse(BatchedSamplingInverse(s),size(s.configurations,2))
function PairedSamplingInverse(work::BatchedSamplingInverse,batch::Integer)
    n,_,count=size(work.lu)
    return PairedSamplingInverse(work,CUDA.zeros(Float64,n÷2,count),CUDA.zeros(Float64,n÷2,count),similar(work.lu),similar(work.lu),similar(work.lu),similar(work.pf_pivots),CUDA.zeros(Bool,batch),CuArray([pointer(work.factors,(m-1)*n*n+1) for m in 1:count]),work.lu_ptrs,CuArray([1.0]))
end

# One writer per matrix. Inspect every inverse entry, including singular cases
# that may have left unusable factors; the boundary fallback replaces them.
function paired_inverse_status!(status,inverse,n)
    b=blockIdx().x
    lane=threadIdx().x
    invalid=CuStaticSharedArray(Int32,128)
    bad=false
    @inbounds for index in lane:128:n*n
        bad |= !isfinite(inverse[index+(b-1)*n*n])
    end
    invalid[lane]=bad
    sync_threads()
    if lane==1
        value=Int32(0)
        for i in 1:128
            value |= invalid[i]
        end
        @inbounds status[b]=value
    end
    return nothing
end

# Conservative cancellation guard: normal elimination is retained when
# projection amplifies sector rounding by more than 1024. This is a fallback
# policy, not a universal forward-error bound; independent CPU tests cover
# cancellation, inverse residuals, singular sectors and complete Opt results.
function paired_fallback_flags!(flags,pfs,weights,inverse_info,batch,nq)
    b=(blockIdx().x-1)*blockDim().x+threadIdx().x
    b>batch && return nothing
    ip=0.0;scale=0.0;fallback=false
    @inbounds for q in 1:nq
        pf=pfs[(q-1)*batch+b]
        term=CUDA.mul_rn(weights[q],pf)
        ip=CUDA.add_rn(ip,term);scale=CUDA.add_rn(scale,abs(term))
        fallback |= pf==0.0 || !isfinite(pf) || inverse_info[(q-1)*batch+b]!=0
    end
    @inbounds flags[b]=fallback || !isfinite(ip) || abs(ip)*1024.0<scale
    return nothing
end
function paired_boundary_fallback!(s,c,flags)
    b=blockIdx().x
    @inbounds flags[b] || return nothing
    rebuild_sampling_cache!(s,c,b)
    return nothing
end
function refresh_sampling_state!(s,cache::GPUSamplingCache,extra::PairedSamplingInverse;initial_pfaffians=nothing)
    work=extra.native
    # Startup has already computed normal Pfaffians; reuse them and its LU
    # inverse. Paired elimination is used after parameter changes in Opt.
    if initial_pfaffians !== nothing
        return refresh_sampling_state!(s,cache,work;initial_pfaffians)
    end
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    n,batch=2s.nelec,size(s.configurations,2);count=batch*s.nqp
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false fill_batched_slater!(work.lu,work.factors,s.configurations,s.slater,s.nsite,s.nelec,batch,count)
    paired_factor_inverse!(extra,n,count)
    fill!(work.lu_info,Int32(0))
    @cuda threads=128 blocks=count paired_inverse_status!(work.inverse_info,work.inverse,n)
    @cuda threads=128 blocks=cld(batch,128) fastmath=false paired_fallback_flags!(
        extra.fallback,work.pfaffians,s.weights,work.inverse_info,batch,s.nqp)
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false publish_batched_inverse!(cache.inverse,cache.pfaffians,cache.valid,work.inverse,work.pfaffians,work.lu_info,work.inverse_info,n,batch,count)
    @cuda threads=128 blocks=batch shmem=16n maxregs=64 fastmath=false paired_boundary_fallback!(cooperative_state(s),sampling_cache_state(cache),extra.fallback)
    @cuda threads=32 blocks=cld(batch,32) fastmath=false publish_sampling_boundary!(cooperative_state(s),sampling_cache_state(cache))
    cache.initialized=true
    return cache
end

# Factor storage must contain the gathered skew matrices on entry.
function paired_factor_inverse!(extra::PairedSamplingInverse,n,count)
    work=extra.native
    @cuda threads=128 blocks=count maxregs=64 fastmath=false paired_skew_factor_kernel!(work.pfaffians,work.factors,work.pf_pivots,extra.diagonal,n)
    @cuda threads=256 blocks=cld(n*n*count,256) paired_identity_kernel!(work.lu,n,count)
    CUDA.cuBLAS.cublasDtrsmBatched(CUDA.cuBLAS.handle(),'L','U','N','U',n,n,extra.one,extra.factor_ptrs,n,extra.M_ptrs,n,count)
    @cuda threads=256 blocks=cld(length(extra.diagonal),256) fastmath=false paired_reciprocal_diagonal_kernel!(extra.inverse_diagonal,extra.diagonal)
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false paired_tridiagonal_solve!(extra.C,work.lu,extra.inverse_diagonal,n,count)
    permutedims!(extra.MT,work.lu,(2,1,3))
    @cuda threads=(32,8) blocks=(cld(n,32),cld(n,8),count) fastmath=false paired_triangular_skew_product!(extra.product,extra.MT,extra.C,n,count)
    @cuda threads=128 blocks=cld(count,128) paired_permutation_kernel!(extra.permutation,work.pf_pivots,n,count)
    @cuda threads=256 blocks=cld(n*n*count,256) paired_permute_inverse_kernel!(work.inverse,extra.product,extra.permutation,n,count)
    return nothing
end
