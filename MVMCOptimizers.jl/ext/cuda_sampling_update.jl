# Optional resident hopping cache. The C update order below follows
# vmc_sampling.jl:update_m_all_real! / CalculateNewPfM2_real.
mutable struct GPUSamplingCache
    owner::CUDAState
    inverse::CuArray{Float64,4}
    pfaffians::CuMatrix{Float64}
    valid::CuMatrix{Bool}
    candidate_pf::CuMatrix{Float64}
    factors::CuArray{Float64,3}
    inverse_scratch::CuArray{Float64,3}
    pivots::CuMatrix{Int32}
    accepted_since_refactor::CuVector{Int32}
    initialized::Bool
end

function GPUSamplingCache(s::CUDAState)
    n,batch=2s.nelec,size(s.configurations,2)
    return GPUSamplingCache(s,CUDA.zeros(Float64,n,n,s.nqp,batch),
        CUDA.zeros(Float64,s.nqp,batch),CUDA.zeros(Bool,s.nqp,batch),
        CUDA.zeros(Float64,s.nqp,batch),CUDA.zeros(Float64,n,n,batch),
        CUDA.zeros(Float64,n,n,batch),CUDA.zeros(Int32,n,batch),
        CUDA.zeros(Int32,batch),false)
end

function invalidate_sampling_cache!(cache::GPUSamplingCache)
    cache.initialized=false
    return cache
end

function sampling_cache_state(cache)
    return (;inverse=cache.inverse,pfaffians=cache.pfaffians,valid=cache.valid,
        candidate_pf=cache.candidate_pf,factors=cache.factors,
        inverse_scratch=cache.inverse_scratch,pivots=cache.pivots,
        accepted_since_refactor=cache.accepted_since_refactor)
end

@inline function rebuild_sampling_cache!(s,c,b)
    n=2s.nelec
    cooperative_inverse_cache_kernel!(c.inverse,c.pfaffians,c.valid,
        s.configurations,s.slater,c.factors,c.inverse_scratch,c.pivots,
        s.nsite,s.nelec,s.nqp,size(s.configurations,2))
    # C publishes a skew inverse. Mirroring one rounded pair avoids relying on
    # the tiny antisymmetry residual of the independent Gauss-Jordan setup.
    @inbounds for q in 1:s.nqp
        if c.valid[q,b]
            for index in threadIdx().x:blockDim().x:n*n
                i=(index-1)%n+1
                j=(index-1)÷n+1
                if i<j
                    x=CUDA.mul_rn(0.5,CUDA.add_rn(c.inverse[i,j,q,b],-c.inverse[j,i,q,b]))
                    c.inverse[i,j,q,b]=x
                    c.inverse[j,i,q,b]=-x
                elseif i==j
                    c.inverse[i,i,q,b]=0.0
                end
            end
        end
    end
    sync_threads()
    if threadIdx().x==1
        @inbounds c.accepted_since_refactor[b]=0
    end
    sync_threads()
    return nothing
end

function initialize_sampling_cache_kernel!(s,c)
    b=blockIdx().x
    rebuild_sampling_cache!(s,c,b)
    if threadIdx().x==1
        @inbounds begin
            s.overlaps[b]=sampling_cache_overlap(s,c,b)
            s.logcorrelations[b]=correlation!(s.counts,s.occupancy,s.configurations,
                s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
        end
    end
    return nothing
end

@inline function cached_hopping_overlap!(s,c,a,b,::Val{N}) where {N}
    @inbounds begin
        ra=s.configurations[a,b]+(a>N÷2 ? s.nsite : 0)+1
        for q in threadIdx().x:blockDim().x:s.nqp
            ratio=0.0
            for j in 1:N
                rj=s.configurations[j,b]+(j>N÷2 ? s.nsite : 0)+1
                ratio=CUDA.add_rn(ratio,CUDA.mul_rn(c.inverse[a,j,q,b],s.slater[rj,ra,q]))
            end
            pf=CUDA.mul_rn(-ratio,c.pfaffians[q,b])
            c.candidate_pf[q,b]=pf
        end
    end
    sync_threads()
    ip=0.0
    if threadIdx().x==1
        @inbounds for q in 1:s.nqp
            ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],c.candidate_pf[q,b]))
        end
    end
    return ip
end

@inline function update_sampling_inverse!(s,c,a,b,::Val{N},::Val{G}) where {N,G}
    n=N
    # Rebuild uses the first n entries of this same scratch only after all
    # inverse updates finish; the two phases do not overlap.
    group=(threadIdx().x-1)÷128
    local_lane=(threadIdx().x-1)%128+1
    offset=2n*group*sizeof(Float64)
    vec1=CuDynamicSharedArray(Float64,n,offset)
    vec2=CuDynamicSharedArray(Float64,n,offset+n*sizeof(Float64))
    @inbounds for phase in 0:cld(s.nqp,G)-1
        q=phase*G+group+1
        ra=s.configurations[a,b]+(a>s.nelec ? s.nsite : 0)+1
        for i in local_lane:128:n
            if q<=s.nqp
            value=0.0
            for j in 1:n
                rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
                # Initialization mirrors a skew pair, and the C update keeps
                # that invariant. Inv[i,j] equals -Inv[j,i], so adjacent lanes
                # can read adjacent rows without changing the ordered j sum.
                value=CUDA.add_rn(value,CUDA.mul_rn(c.inverse[i,j,q,b],s.slater[rj,ra,q]))
            end
            vec1[i]=value
            end
        end
        sync_threads()
        scale=q<=s.nqp ? CUDA.div_rn(-1.0,vec1[a]) : 0.0
        for i in local_lane:128:n
            if q<=s.nqp
            vec2[i]=CUDA.mul_rn(c.inverse[a,i,q,b],scale)
            end
        end
        sync_threads()
        for index in local_lane:128:n*n
            if q<=s.nqp
            i=(index-1)%n+1
            j=(index-1)÷n+1
            delta=CUDA.add_rn(CUDA.mul_rn(vec1[i],vec2[j]),-CUDA.mul_rn(vec1[j],vec2[i]))
            x=CUDA.add_rn(c.inverse[i,j,q,b],delta)
            j==a && (x=CUDA.add_rn(x,-vec2[i]))
            i==a && (x=CUDA.add_rn(x,vec2[j]))
            c.inverse[i,j,q,b]=x
            end
        end
        if local_lane==1 && q<=s.nqp
            c.pfaffians[q,b]=c.candidate_pf[q,b]
        end
        sync_threads()
    end
    return nothing
end

@inline function sampling_cache_overlap(s,c,b)
    ip=0.0
    @inbounds for q in 1:s.nqp
        ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],c.pfaffians[q,b]))
    end
    return ip
end

@inline function empty_hopping_destination(occ,localspins,spin,word,ns,b)
    count=0
    @inbounds for r in 1:ns
        if occ[r+spin*ns,b]==0 && localspins[r]==0
            count+=1
        end
    end
    count==0 && return -1
    rank=Int(word%UInt32(count))
    @inbounds for r in 1:ns
        if occ[r+spin*ns,b]==0 && localspins[r]==0
            rank==0 && return r-1
            rank-=1
        end
    end
    return -1
end

@inline function counts_logcorrelation(s,b)
    value=0.0
    @inbounds for p in 1:s.ngutz+s.njastrow
        value+=s.parameters[p]*s.counts[p,b]
    end
    return value
end

# Exact integer version of the ordinary hopping projection-count update.
# Only pairs touching either moved site change; their input table order is
# irrelevant to integer counts, and the parameter dot retains the full order.
@inline function update_hopping_counts!(s,source,destination,spin,b)
    ns=s.nsite
    ri,rj=source+1,destination+1
    opposite=1-spin
    @inbounds begin
        p=s.gutz[ri]+1
        1<=p<=s.ngutz && (s.counts[p,b]-=s.occupancy[ri+opposite*ns,b])
        p=s.gutz[rj]+1
        1<=p<=s.ngutz && (s.counts[p,b]+=s.occupancy[rj+opposite*ns,b])
        xi=s.occupancy[ri,b]+s.occupancy[ri+ns,b]-1
        xj=s.occupancy[rj,b]+s.occupancy[rj+ns,b]-1
        for r in 1:ns
            if r!=ri && r!=rj
                xr=s.occupancy[r,b]+s.occupancy[r+ns,b]-1
                p=s.jastrow[min(ri,r),max(ri,r)]+1
                1<=p<=s.njastrow && (s.counts[s.ngutz+p,b]-=xr)
                p=s.jastrow[min(rj,r),max(rj,r)]+1
                1<=p<=s.njastrow && (s.counts[s.ngutz+p,b]+=xr)
            end
        end
        p=s.jastrow[min(ri,rj),max(ri,rj)]+1
        1<=p<=s.njastrow && (s.counts[s.ngutz+p,b]+=xi-xj-1)
        s.occupancy[ri+spin*ns,b]=0
        s.occupancy[rj+spin*ns,b]=1
    end
    return nothing
end

@inline function exchange_occupancy!(s,source,destination,spin,b)
    @inbounds begin
        s.occupancy[source+1+spin*s.nsite,b]=0
        s.occupancy[destination+1+spin*s.nsite,b]=1
        s.occupancy[destination+1+(1-spin)*s.nsite,b]=0
        s.occupancy[source+1+(1-spin)*s.nsite,b]=1
    end
    return nothing
end

function cached_sampling_kernel!(s,c,tape,exchange_fraction,empty_destinations,
                                 incremental_correlation,dimension::Val{N},groups::Val{G}=Val(1)) where {N,G}
    b=blockIdx().x
    lane=threadIdx().x
    move=CuStaticSharedArray(Int32,8) # valid,a,c,old site a,old site c,rank,accept,rebuild
    values=CuStaticSharedArray(Float64,4) # old overlap/log, new log/overlap
    if lane==1
        @inbounds begin
            values[1]=s.overlaps[b]
            values[2]=s.logcorrelations[b]
            s.accepted[b]=0
        end
    end
    sync_threads()
    @inbounds for step in 1:size(tape,3)
        if lane==1
            spin=tape[2,b,step]<UInt32(0x80000000) ? 0 : 1
            a=Int(tape[1,b,step]%UInt32(s.nelec))+1+spin*s.nelec
            ri=s.configurations[a,b]
            exchange=Float64(tape[4,b,step])/4294967296.0<exchange_fraction
            second=0
            valid=true
            if exchange
                second=Int(tape[3,b,step]%UInt32(s.nelec))+1+(1-spin)*s.nelec
                rj=s.configurations[second,b]
                valid=ri!=rj
                for k in 1:s.nelec
                    if s.configurations[k+spin*s.nelec,b]==rj ||
                       s.configurations[k+(1-spin)*s.nelec,b]==ri
                        valid=false
                    end
                end
            else
                if empty_destinations
                    rj=empty_hopping_destination(s.occupancy,s.localspins,spin,
                        tape[3,b,step],s.nsite,b)
                    valid=rj>=0 && s.localspins[ri+1]==0
                else
                    rj=Int(tape[3,b,step]%UInt32(s.nsite))
                    valid=s.localspins[ri+1]==0 && s.localspins[rj+1]==0
                    for k in 1:s.nelec
                        s.configurations[k+spin*s.nelec,b]==rj && (valid=false)
                    end
                end
            end
            rank=second==0
            for q in 1:s.nqp
                rank &= c.valid[q,b]
            end
            move[1]=valid;move[2]=a;move[3]=second;move[4]=ri;move[5]=rj
            move[6]=rank;move[7]=0;move[8]=0
            if valid
                if incremental_correlation
                    if second==0
                        update_hopping_counts!(s,Int(ri),Int(rj),spin,b)
                    else
                        exchange_occupancy!(s,Int(ri),Int(rj),spin,b)
                    end
                    values[3]=counts_logcorrelation(s,b)
                end
                s.configurations[a,b]=rj
                second!=0 && (s.configurations[second,b]=ri)
                if !incremental_correlation
                    values[3]=correlation!(s.counts,s.occupancy,s.configurations,s.gutz,
                        s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
                end
            end
        end
        sync_threads()
        if move[1]!=0
            if move[6]!=0
                ip=cached_hopping_overlap!(s,c,Int(move[2]),b,dimension)
                if lane==1
                    values[4]=ip
                end
            else
                ip=cooperative_overlap!(s,b)
                if lane==1
                    values[4]=ip
                end
            end
            sync_threads()
            if lane==1
                logratio=2*(values[3]-values[2]+log(abs(values[4])+1e-100)-log(abs(values[1])+1e-100))
                u=Float64(tape[5,b,step])/4294967296.0
                if values[4]!=0 && log(u)<logratio
                    values[1]=values[4];values[2]=values[3]
                    s.accepted[b]+=1
                    c.accepted_since_refactor[b]+=1
                    move[7]=1
                    move[8]=move[6]==0 || c.accepted_since_refactor[b]>s.nsite
                    for q in 1:s.nqp
                        if move[6]!=0 && (c.candidate_pf[q,b]==0 || !isfinite(c.candidate_pf[q,b]))
                            move[8]=1
                        end
                    end
                else
                    spin=move[2]>s.nelec ? 1 : 0
                    if move[3]==0
                        if incremental_correlation
                            update_hopping_counts!(s,Int(move[5]),Int(move[4]),spin,b)
                        else
                            s.occupancy[move[4]+1+spin*s.nsite,b]=1
                            s.occupancy[move[5]+1+spin*s.nsite,b]=0
                        end
                    else
                        exchange_occupancy!(s,Int(move[5]),Int(move[4]),spin,b)
                    end
                    s.configurations[move[2],b]=move[4]
                    move[3]!=0 && (s.configurations[move[3],b]=move[5])
                end
            end
            sync_threads()
            if move[7]!=0
                if move[8]!=0
                    rebuild_sampling_cache!(s,c,b)
                    if lane==1
                        values[1]=sampling_cache_overlap(s,c,b)
                    end
                else
                    update_sampling_inverse!(s,c,Int(move[2]),b,dimension,groups)
                end
            end
        end
        sync_threads()
    end
    if lane==1
        @inbounds begin
            s.overlaps[b]=values[1]
            s.logcorrelations[b]=incremental_correlation ? values[2] :
                correlation!(s.counts,s.occupancy,s.configurations,
                    s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
        end
    end
    return nothing
end

function refresh_sampling_state!(s,cache::GPUSamplingCache)
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    batch=size(s.configurations,2)
    shared=16*(2s.nelec)
    # Small production states omit the large-wavefunction scratch allocation;
    # the cache owns a full buffer for its fallback on every supported size.
    state=merge(cooperative_state(s),(;cooperative_factors=cache.factors))
    @cuda threads=128 blocks=batch shmem=shared fastmath=false initialize_sampling_cache_kernel!(
        state,sampling_cache_state(cache))
    cache.initialized=true
    return s
end

function sample_with_cache!(s,cache::GPUSamplingCache,draws,exchange_fraction;
                            empty_destinations=false,incremental_correlation=false,qp_groups::Int=1,
                            kernel_maxregs::Int=64)
    qp_groups in (1,2,4,8) || throw(ArgumentError("qp_groups must be 1, 2, 4, or 8"))
    1<=kernel_maxregs<=255 || throw(ArgumentError("kernel_maxregs must be between 1 and 255"))
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    batch=size(s.configurations,2)
    shared=16*(2s.nelec)*qp_groups
    state=merge(cooperative_state(s),(;cooperative_factors=cache.factors))
    !cache.initialized && refresh_sampling_state!(s,cache)
    @cuda threads=128qp_groups blocks=batch shmem=shared maxregs=kernel_maxregs fastmath=false cached_sampling_kernel!(
        state,sampling_cache_state(cache),draws,Float64(exchange_fraction),
        empty_destinations,incremental_correlation,Val(2s.nelec),Val(qp_groups))
    return s
end
