@inline function baseline_counts_logcorrelation(s,b)
    value=0.0
    @inbounds for p in 1:s.ngutz+s.njastrow
        value+=s.parameters[p]*s.counts[p,b]
    end
    return value
end

# Exact integer version of the ordinary hopping projection-count update.
# Only pairs touching either moved site change; their input table order is
# irrelevant to integer counts, and the parameter dot retains the full order.
@inline function baseline_update_hopping_counts!(s,source,destination,spin,b)
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
                        baseline_update_hopping_counts!(s,Int(ri),Int(rj),spin,b)
                    else
                        exchange_occupancy!(s,Int(ri),Int(rj),spin,b)
                    end
                    values[3]=baseline_counts_logcorrelation(s,b)
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
                            baseline_update_hopping_counts!(s,Int(move[5]),Int(move[4]),spin,b)
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
                    update_sampling_inverse!(s,c,Int(move[2]),b,dimension,groups,triangle)
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


function baseline_sample_with_cache!(s,cache::GPUSamplingCache,draws,exchange_fraction;
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
    @cuda threads=128qp_groups blocks=batch shmem=shared maxregs=kernel_maxregs fastmath=false baseline_cached_sampling_kernel!(
        state,sampling_cache_state(cache),draws,Float64(exchange_fraction),
        empty_destinations,incremental_correlation,Val(2s.nelec),Val(qp_groups),Val(triangle_update))
    return s
end

function candidate_sample_with_cache!(s,cache::GPUSamplingCache,draws,exchange_fraction;
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
    c=sampling_cache_state(cache)
    capacity=0
    if cache.updates !== nothing && qp_groups==1 && triangle_update
        capacity=SAMPLING_DELAY_CAPACITY
        shared=delayed_sampling_shared(2s.nelec,capacity)
        c=merge(c,delayed_sampling_state(cache.updates))
    end
    @cuda threads=128qp_groups blocks=batch shmem=shared maxregs=kernel_maxregs fastmath=false cached_sampling_kernel!(
        state,c,draws,Float64(exchange_fraction),empty_destinations,
        incremental_correlation,Val(2s.nelec),Val(qp_groups),Val(triangle_update),Val(capacity))
    return s
end

function baseline_fused_sparse_derivative_kernel!(O, colptr, coordinates, coefficients,
                                        mapping, inv, pf, valid, overlaps, weights,
                                        ns, nq, np, batch)
    thread = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    output = thread ÷ 32
    lane = thread % 32
    output >= np*batch && return nothing
    # Adjacent warps reuse a column's coefficient/coordinate stream across
    # walkers. Output ownership and each dot-product's arithmetic stay fixed.
    b = output % batch+1
    p = output ÷ batch+1
    @inbounds for q in 1:nq
        !valid[q,b] && return nothing
    end
    result = 0.0
    @inbounds for k in colptr[p]+lane:32:colptr[p+1]-1
        ri = Int(coordinates.ri[k])
        rj = Int(coordinates.rj[k])
        q = Int(coordinates.q[k])
        i = -Int(mapping[ri,b])
        j = -Int(mapping[rj,b])
        if i > 0 && j > 0 && i < j
            cofactor = pf[q,b]*inv[j,i,q,b]
            sensitivity = -weights[q]*cofactor/overlaps[b]
            result += coefficients[k]*sensitivity
        end
    end
    for offset in (16,8,4,2,1)
        result += shfl_down_sync(UInt32(0xffffffff),result,offset)
    end
    if lane == 0
        @inbounds O[p,b] = result
    end
    return nothing
end

function baseline_fused_sparse_derivatives!(s, sr, o)
    work = o.batched_inverse
    basis = sr.sparse_basis
    batch = size(s.configurations,2)
    np = size(sr.derivatives,1)
    # Cache-only callers may not have built the observable transform. Refresh
    # the resident occupied-index lookup from the current configurations.
    @cuda threads=128 blocks=cld(batch,128) target_sites_kernel!(
        work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
    @cuda threads=256 blocks=cld(np*batch,8) fastmath=false baseline_fused_sparse_derivative_kernel!(
        sr.derivatives,basis.colPtr,sr.decoded_basis,basis.nzVal,work.target_map,
        cache_view(o.inverse),o.pfaffians,o.valid,s.overlaps,s.weights,
        s.nsite,s.nqp,np,batch)
    return sr.derivatives
end


function candidate_fused_sparse_derivatives!(s, sr, o)
    work = o.batched_inverse
    basis = sr.sparse_basis
    batch = size(s.configurations,2)
    np = size(sr.derivatives,1)
    # Cache-only callers may not have built the observable transform. Refresh
    # the resident occupied-index lookup from the current configurations.
    @cuda threads=128 blocks=cld(batch,128) target_sites_kernel!(
        work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
    @cuda threads=256 blocks=cld(np*batch,8) fastmath=false fused_sparse_derivative_kernel!(
        sr.derivatives,basis.colPtr,sr.decoded_basis,basis.nzVal,work.target_map,
        cache_view(o.inverse),o.pfaffians,o.valid,s.overlaps,s.weights,
        s.nsite,s.nqp,np,batch)
    return sr.derivatives
end


function candidate_runner_sampling_cache(s::CUDAState;delayed::Bool=false)
    mode = get(ENV,"JULIA_MVMC_GPU_SAMPLING","update")
    mode in ("update","refactor") || throw(ArgumentError("GPU sampling must be update or refactor"))
    return s.nsite >= 64 && mode == "update" ? GPUSamplingCache(s;delayed=delayed && supports_delayed_sampling(s)) : nothing
end

function sample_with_cache!(s,cache::GPUSamplingCache,draws,exchange_fraction;kwargs...)
    if ENV["JULIA_MVMC_DELAYED_AB"]=="baseline"
        return baseline_sample_with_cache!(s,cache,draws,exchange_fraction;kwargs...)
    end
    return candidate_sample_with_cache!(s,cache,draws,exchange_fraction;kwargs...)
end
function fused_sparse_derivatives!(s,sr,o)
    return ENV["JULIA_MVMC_DELAYED_AB"]=="baseline" ?
        baseline_fused_sparse_derivatives!(s,sr,o) : candidate_fused_sparse_derivatives!(s,sr,o)
end
function runner_sampling_cache(s::CUDAState;delayed::Bool=false)
    return candidate_runner_sampling_cache(s;delayed=delayed && ENV["JULIA_MVMC_DELAYED_AB"]!="baseline")
end
