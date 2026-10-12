# Publish the sampler's current inverse/Pfaffians into the observable layouts.
# Only device copies occur; parameter/Slater mutations must invalidate or refresh
# the sampling cache before reuse, as required by the cached sampler itself.
function publish_sampling_inverse_kernel!(output,packed,pf,valid,source,source_pf,source_valid,n,batch,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    index > n*n*count && return nothing
    i=(index-1)%n+1
    j=((index-1)÷n)%n+1
    m=(index-1)÷(n*n)+1
    b=(m-1)%batch+1
    q=(m-1)÷batch+1
    @inbounds begin
        x=source_valid[q,b] ? source[i,j,q,b] : 0.0
        output[i,j,q,b]=x
        packed[i,j,m]=x
        if i==1 && j==1
            pf[q,b]=source_pf[q,b]
            valid[q,b]=source_valid[q,b]
        end
    end
    return nothing
end

function observable_sampling_cache!(s::CUDAState,o::CUDAObservables,cache::GPUSamplingCache)
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    o.batched_inverse!==nothing || throw(ArgumentError("sampling cache reuse requires batched observables"))
    cache.initialized || refresh_sampling_state!(s,cache)
    n,batch=2s.nelec,size(s.configurations,2)
    count=batch*s.nqp
    @cuda threads=256 blocks=cld(n*n*count,256) fastmath=false publish_sampling_inverse_kernel!(
        cache_view(o.inverse),o.batched_inverse.inverse,o.pfaffians,o.valid,
        cache.inverse,cache.pfaffians,cache.valid,n,batch,count)
    return nothing
end
