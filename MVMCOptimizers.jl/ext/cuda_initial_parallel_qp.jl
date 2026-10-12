# Large PhysCal initialization factors QPs independently, then retains the
# original ordered projection sum and final-QP factor/pivot scratch.
function initial_qp_wavefunction_kernel!(s,pending,candidate,factors,pivots,pfs,batch)
    m=blockIdx().x
    b=(m-1)%batch+1;q=(m-1)÷batch+1
    @inbounds (!pending[b] || !candidate[b]) && return nothing
    n=2s.nelec;lane=threadIdx().x;stride=blockDim().x
    @inbounds for index in lane:stride:n*n
        i=(index-1)%n+1;j=(index-1)÷n+1
        ri=s.configurations[i,b]+(i>s.nelec ? s.nsite : 0)+1
        rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
        factors[i,j,m]=-s.slater[ri,rj,q]
    end
    sync_threads()
    pf=cooperative_pfaffian!(factors,pivots,n,m)
    lane==1 && (@inbounds pfs[q,b]=pf)
    return nothing
end
function initial_qp_publish_kernel!(s,pending,candidate,factors,pivots,pfs,batch)
    b=blockIdx().x
    @inbounds (!pending[b] || !candidate[b]) && return nothing
    lane=threadIdx().x;stride=blockDim().x;n=2s.nelec;m=(s.nqp-1)*batch+b
    @inbounds for index in lane:stride:n*n
        i=(index-1)%n+1;j=(index-1)÷n+1
        s.cooperative_factors[i,j,b]=factors[i,j,m]
    end
    @inbounds for i in lane:stride:n
        s.pivots[i,b]=pivots[i,m]
    end
    if lane==1
        ip=0.0;valid=true
        @inbounds for q=1:s.nqp
            pf=pfs[q,b]
            valid &= isfinite(pf) && pf!=0
            ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],pf))
        end
        @inbounds begin
            log=correlation!(s.counts,s.occupancy,s.configurations,s.gutz,s.jastrow,
                s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
            s.logcorrelations[b]=log;s.overlaps[b]=ip
            pending[b]=!valid || !isfinite(ip) || ip==0 || !isfinite(log)
        end
    end
    return nothing
end
function parallel_initial_workspace(s)
    n=2s.nelec;batch=size(s.configurations,2);count=batch*s.nqp
    # Active matrices are fully gathered before PF; skipped walkers are not published.
    return (;factors=CuArray{Float64}(undef,n,n,count),pivots=CUDA.zeros(Int32,n,count),
        pfs=CUDA.zeros(Float64,s.nqp,batch))
end
function initial_wavefunctions_parallel!(s,pending,candidate,work)
    batch=size(s.configurations,2);view=initial_state_view(s)
    @cuda threads=128 blocks=batch*s.nqp maxregs=64 fastmath=false initial_qp_wavefunction_kernel!(
        view,pending,candidate,work.factors,work.pivots,work.pfs,batch)
    @cuda threads=128 blocks=batch fastmath=false initial_qp_publish_kernel!(
        view,pending,candidate,work.factors,work.pivots,work.pfs,batch)
    return work.pfs
end
