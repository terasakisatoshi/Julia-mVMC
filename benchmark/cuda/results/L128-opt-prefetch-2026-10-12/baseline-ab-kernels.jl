@inline function baseline_update_sampling_inverse!(s,c,a,b,::Val{N},::Val{G},::Val{H}) where {N,G,H}
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
        if H && G==1
            tile=CuDynamicSharedArray(Float64,(33,32),16n)
            for column_tile in 0:cld(n,32)-1, row_tile in 0:column_tile
                # Compute only upper entries with their original ordered RN
                # operations. Adjacent lanes read/store adjacent matrix rows.
                for index in local_lane:128:1024
                    ri=(index-1)%32+1;cj=(index-1)÷32+1
                    i=32row_tile+ri;j=32column_tile+cj
                    if i<=n && j<=n && i<j
                        delta=CUDA.add_rn(CUDA.mul_rn(vec1[i],vec2[j]),-CUDA.mul_rn(vec1[j],vec2[i]))
                        x=CUDA.add_rn(c.inverse[i,j,q,b],delta)
                        j==a && (x=CUDA.add_rn(x,-vec2[i]))
                        i==a && (x=CUDA.add_rn(x,vec2[j]))
                        c.inverse[i,j,q,b]=x
                        tile[ri,cj]=x
                    elseif i==j && i<=n
                        c.inverse[i,j,q,b]=0.0
                    end
                end
                sync_threads()
                # Transpose shared coordinates, rather than global stores,
                # so the sign-mirrored lower entries are also coalesced.
                for index in local_lane:128:1024
                    ri=(index-1)%32+1;cj=(index-1)÷32+1
                    i=32column_tile+ri;j=32row_tile+cj
                    if i<=n && j<=n && i>j
                        c.inverse[i,j,q,b]=-tile[cj,ri]
                    end
                end
                sync_threads()
            end
        else
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
        end
        if local_lane==1 && q<=s.nqp
            c.pfaffians[q,b]=c.candidate_pf[q,b]
        end
        sync_threads()
    end
    return nothing
end


function baseline_cached_sampling_kernel!(s,c,tape,exchange_fraction,empty_destinations,
                                 incremental_correlation,dimension::Val{N},groups::Val{G}=Val(1),
                                 triangle::Val{H}=Val(false)) where {N,G,H}
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
                    baseline_update_sampling_inverse!(s,c,Int(move[2]),b,dimension,groups,triangle)
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


function sample_with_cache!(s,cache::GPUSamplingCache,draws,exchange_fraction;
                            empty_destinations=false,incremental_correlation=false,qp_groups::Int=1,
                            kernel_maxregs::Int=64,triangle_update::Bool=false)
    qp_groups in (1,2,4,8) || throw(ArgumentError("qp_groups must be 1, 2, 4, or 8"))
    1<=kernel_maxregs<=255 || throw(ArgumentError("kernel_maxregs must be between 1 and 255"))
    cache.owner===s || throw(ArgumentError("sampling cache belongs to a different state"))
    batch=size(s.configurations,2)
    shared=16*(2s.nelec)*qp_groups
    triangle_update && qp_groups==1 && (shared+=33*32*sizeof(Float64))
    state=merge(cooperative_state(s),(;cooperative_factors=cache.factors))
    !cache.initialized && refresh_sampling_state!(s,cache)
    if get(ENV,"JULIA_MVMC_PREFETCH_AB","candidate")=="baseline"
        @cuda threads=128qp_groups blocks=batch shmem=shared maxregs=kernel_maxregs fastmath=false baseline_cached_sampling_kernel!(
        state,sampling_cache_state(cache),draws,Float64(exchange_fraction),
        empty_destinations,incremental_correlation,Val(2s.nelec),Val(qp_groups),Val(triangle_update))
    else
        @cuda threads=128qp_groups blocks=batch shmem=shared maxregs=kernel_maxregs fastmath=false cached_sampling_kernel!(state,sampling_cache_state(cache),draws,Float64(exchange_fraction),empty_destinations,incremental_correlation,Val(2s.nelec),Val(qp_groups),Val(triangle_update))
    end
    return s
end
