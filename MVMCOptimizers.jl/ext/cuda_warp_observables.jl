# The constructor selects this route only for a Hamiltonian made from
# one-body operators and diagonal two-body densities, without requested Green
# functions. Invalid sectors retain the independent scalar fallback. One warp
# owns an operator/walker; packed inverse columns give coalesced reads.
# Projection deltas and inverse dots use fixed FP64 warp reductions. The final
# Hamiltonian sum retains the original term order.
@inline function operator_warp_sum(x)
    for offset in (16,8,4,2,1)
        x=CUDA.add_rn(x,CUDA.shfl_down_sync(typemax(UInt32),x,offset))
    end
    return x
end
@inline function warp_hopping_correlation(op,occ,gutz,jastrow,parameters,ns,ng,nj,b,lane)
    dest=op[1]+1;source=op[2]+1;spin=op[3]
    value=0.0
    @inbounds for r in lane:32:ns
        if r!=source && r!=dest
            x=occ[r,b]+occ[r+ns,b]-1
            p=jastrow[min(dest,r),max(dest,r)]+1
            if 1<=p<=nj
                value=CUDA.add_rn(value,CUDA.mul_rn(parameters[ng+p],Float64(x)))
            end
            p=jastrow[min(source,r),max(source,r)]+1
            if 1<=p<=nj
                value=CUDA.add_rn(value,-CUDA.mul_rn(parameters[ng+p],Float64(x)))
            end
        end
    end
    @inbounds if lane==1
        p=gutz[dest]+1
        1<=p<=ng && (value=CUDA.add_rn(value,CUDA.mul_rn(parameters[p],Float64(occ[dest+(1-spin)*ns,b]))))
        p=gutz[source]+1
        1<=p<=ng && (value=CUDA.add_rn(value,-CUDA.mul_rn(parameters[p],Float64(occ[source+(1-spin)*ns,b]))))
        xs=occ[source,b]+occ[source+ns,b]-1
        xd=occ[dest,b]+occ[dest+ns,b]-1
        p=jastrow[min(dest,source),max(dest,source)]+1
        1<=p<=nj && (value=CUDA.add_rn(value,CUDA.mul_rn(parameters[ng+p],Float64(xs-xd-1))))
    end
    return operator_warp_sum(value)
end
function warp_hamiltonian_kernel!(termvalues,ops,cfg,occ,overlaps,logs,gutz,jastrow,
    parameters,slater,weights,pf,valid,packed,
    ns,ne,ng,nj,nq,batch)
    thread=(blockIdx().x-1)*blockDim().x+threadIdx().x-1
    index=thread÷32
    lane=thread%32+1
    count=size(termvalues,1)
    index>=count*batch && return nothing
    t=index%count+1;b=index÷count+1
    @inbounds for q in 1:nq
        !valid[q,b] && return nothing
    end
    @inbounds op=ops[t]
    if op[7]!=1
        if lane==1
            @inbounds termvalues[t,b]=Float64(occ[op[2]+op[3]*ns+1,b]*occ[op[5]+op[6]*ns+1,b])
        end
        return nothing
    end
    a=Int32(0)
    @inbounds for index in lane:32:ne
        k=op[3]*ne+index
        site=cfg[k,b]
        if op[1]!=op[2] && site==op[1]
            a=typemax(Int32)
        elseif site==op[2]
            a=max(a,Int32(k))
        end
    end
    for offset in (16,8,4,2,1)
        a=max(a,CUDA.shfl_xor_sync(typemax(UInt32),a,offset))
    end
    a==typemax(Int32) && (a=Int32(0))
    if a==0 || op[1]==op[2]
        lane==1 && (@inbounds termvalues[t,b]=a==0 ? 0.0 : 1.0)
        return nothing
    end
    delta=warp_hopping_correlation(op,occ,gutz,jastrow,parameters,ns,ng,nj,b,lane)
    ra=op[1]+op[3]*ns+1
    ip=0.0
    @inbounds for q in 1:nq
        ratio=0.0;m=(q-1)*batch+b
        for j in lane:32:2ne
            rj=j==a ? ra : cfg[j,b]+(j>ne ? ns : 0)+1
            ratio=muladd(packed[j,a,m],slater[rj,ra,q],ratio)
        end
        ratio=operator_warp_sum(ratio)
        if lane==1
            ip=CUDA.add_rn(ip,CUDA.mul_rn(weights[q],CUDA.mul_rn(ratio,pf[q,b])))
        end
    end
    if lane==1
        @inbounds termvalues[t,b]=exp(delta)*ip/overlaps[b]
    end
    return nothing
end
function warp_hamiltonian_observables!(s,o)
    batch=size(s.configurations,2);count=size(o.termvalues,1)
    if count>0
        @cuda threads=128 blocks=cld(count*batch,4) maxregs=64 fastmath=false warp_hamiltonian_kernel!(
            o.termvalues,o.ops,s.configurations,s.occupancy,s.overlaps,s.logcorrelations,
            s.gutz,s.jastrow,s.parameters,s.slater,s.weights,o.pfaffians,o.valid,
            o.batched_inverse.inverse,
            s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch)
    end
    @cuda threads=128 blocks=cld(batch,128) fastmath=false ordered_observable_reduce_kernel!(
        o.values,o.termvalues,o.coeffs,o.valid,s.nqp,batch)
    return nothing
end
