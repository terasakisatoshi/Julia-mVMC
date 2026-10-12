# Each site contributes exact Int32 projection-count changes. Integer atomics
# commute, and the barrier precedes both the occupancy mutation and log sum.
# The fixed warp reduction changes only floating-point log-correlation order.
@inline function parallel_hopping_counts!(s,source,destination,spin,b)
    ns=s.nsite;ri=source+1;rj=destination+1
    @inbounds for r in threadIdx().x:blockDim().x:ns
        if r!=ri && r!=rj
            xr=s.occupancy[r,b]+s.occupancy[r+ns,b]-1
            if xr!=0
                p=s.jastrow[min(ri,r),max(ri,r)]+1
                if 1<=p<=s.njastrow
                    CUDA.@atomic s.counts[s.ngutz+p,b] += Int32(-xr)
                end
                p=s.jastrow[min(rj,r),max(rj,r)]+1
                if 1<=p<=s.njastrow
                    CUDA.@atomic s.counts[s.ngutz+p,b] += Int32(xr)
                end
            end
        end
    end
    sync_threads()
    if threadIdx().x==1
        @inbounds begin
            p=s.gutz[ri]+1
            1<=p<=s.ngutz && (s.counts[p,b]-=s.occupancy[ri+(1-spin)*ns,b])
            p=s.gutz[rj]+1
            1<=p<=s.ngutz && (s.counts[p,b]+=s.occupancy[rj+(1-spin)*ns,b])
            xi=s.occupancy[ri,b]+s.occupancy[ri+ns,b]-1
            xj=s.occupancy[rj,b]+s.occupancy[rj+ns,b]-1
            p=s.jastrow[min(ri,rj),max(ri,rj)]+1
            1<=p<=s.njastrow && (s.counts[s.ngutz+p,b]+=xi-xj-1)
            s.occupancy[ri+spin*ns,b]=0
            s.occupancy[rj+spin*ns,b]=1
        end
    end
    sync_threads()
    return nothing
end
@inline function parallel_counts_logcorrelation(s,b)
    tid=threadIdx().x
    lane=(tid-1)%32+1;warp=(tid-1)÷32+1
    shared=CuStaticSharedArray(Float64,32)
    value=0.0
    @inbounds for p in tid:blockDim().x:s.ngutz+s.njastrow
        value=CUDA.add_rn(value,CUDA.mul_rn(s.parameters[p],Float64(s.counts[p,b])))
    end
    for offset in (16,8,4,2,1)
        value=CUDA.add_rn(value,CUDA.shfl_down_sync(typemax(UInt32),value,offset))
    end
    lane==1 && (shared[warp]=value)
    sync_threads()
    value=0.0
    if tid==1
        for w in 1:blockDim().x÷32
            value=CUDA.add_rn(value,shared[w])
        end
    end
    return value
end
