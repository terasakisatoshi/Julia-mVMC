# A sampling call accumulates four rank-two changes before materializing the
# inverse. Every return and independent refactor leaves an empty queue.
# The pending representation retains the original unit-vector corrections:
# inv += u*v' - v*u' + e_a*v' - v*e_a'. Here u is the inverse-column product,
# and v is the scaled inverse row. Keeping a separately avoids cancelling the
# unit correction when constructing u. FMA and fixed warp reductions change
# rounding; tests check independent inverse residuals and physical results.
struct DelayedSamplingUpdates
    u::CuArray{Float64,4}
    v::CuArray{Float64,4}
    count::CuVector{Int32}
    site::CuMatrix{Int32}
    dots::CuArray{Float64,4}
    ratio::CuMatrix{Float64}
end
const SAMPLING_DELAY_CAPACITY=4
delayed_sampling_shared(n,k=SAMPLING_DELAY_CAPACITY)=8*(3n+2k+33*32)
function supports_delayed_sampling(s::CUDAState)
    # Include a conservative allowance for the kernel's small static arrays.
    budget=CUDA.attribute(CUDA.device(),CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
    return 2s.nelec>=128 && delayed_sampling_shared(2s.nelec)+1024<=budget
end
function DelayedSamplingUpdates(s::CUDAState)
    n,batch,k=2s.nelec,size(s.configurations,2),SAMPLING_DELAY_CAPACITY
    u=CUDA.zeros(Float64,n,k,s.nqp,batch)
    return DelayedSamplingUpdates(u,similar(u),CUDA.zeros(Int32,batch),
        CUDA.zeros(Int32,k,batch),CUDA.zeros(Float64,2,k,s.nqp,batch),
        CUDA.zeros(Float64,s.nqp,batch))
end
function delayed_sampling_state(w::DelayedSamplingUpdates)
    return (;delay_u=w.u,delay_v=w.v,delay_count=w.count,delay_site=w.site,
        delay_dots=w.dots,delay_ratio=w.ratio)
end

@inline function delayed_inverse_entry(c,i,j,q,b,count)
    i==j && return 0.0
    row=min(i,j);col=max(i,j)
    @inbounds value=c.inverse[row,col,q,b]
    @inbounds for k in 1:count
        value=muladd(c.delay_u[row,k,q,b],c.delay_v[col,k,q,b],muladd(-c.delay_u[col,k,q,b],c.delay_v[row,k,q,b],value))
        site=c.delay_site[k,b]
        col==site && (value=CUDA.add_rn(value,-c.delay_v[row,k,q,b]))
        row==site && (value=CUDA.add_rn(value,c.delay_v[col,k,q,b]))
    end
    return i<j ? value : -value
end
@inline function delayed_warp_sum(x)
    for offset in (16,8,4,2,1)
        x=CUDA.add_rn(x,CUDA.shfl_down_sync(typemax(UInt32),x,offset))
    end
    return x
end
@inline function delayed_hopping_overlap!(s,c,a,b,::Val{N}) where {N}
    count=Int(c.delay_count[b])
    warp=(threadIdx().x-1)÷32+1;lane=(threadIdx().x-1)%32+1
    warps=blockDim().x÷32
    @inbounds begin
        ra=s.configurations[a,b]+(a>s.nelec ? s.nsite : 0)+1
        for q in warp:warps:s.nqp
            partial=0.0
            for j in lane:32:N
                rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
                partial=muladd(-c.inverse[j,a,q,b],s.slater[rj,ra,q],partial)
            end
            ratio=delayed_warp_sum(partial)
            for k in 1:count
                udot=0.0;vdot=0.0
                for j in lane:32:N
                    rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
                    col=s.slater[rj,ra,q]
                    udot=muladd(c.delay_u[j,k,q,b],col,udot)
                    vdot=muladd(c.delay_v[j,k,q,b],col,vdot)
                end
                udot=delayed_warp_sum(udot);vdot=delayed_warp_sum(vdot)
                if lane==1
                    site=c.delay_site[k,b]
                    ri=s.configurations[site,b]+(site>s.nelec ? s.nsite : 0)+1
                    udot=CUDA.add_rn(udot,s.slater[ri,ra,q])
                    c.delay_dots[1,k,q,b]=udot;c.delay_dots[2,k,q,b]=vdot
                    ratio=muladd(c.delay_u[a,k,q,b],vdot,muladd(-c.delay_v[a,k,q,b],udot,ratio))
                    a==site && (ratio=CUDA.add_rn(ratio,vdot))
                end
            end
            if lane==1
                c.delay_ratio[q,b]=ratio
                c.candidate_pf[q,b]=CUDA.mul_rn(-ratio,c.pfaffians[q,b])
            end
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
@inline function delayed_flush!(s,c,b,::Val{N},::Val{K}) where {N,K}
    count=Int(c.delay_count[b]);count==0 && return nothing
    lane=threadIdx().x
    tile=CuDynamicSharedArray(Float64,(33,32),8*(3N+2K))
    @inbounds for q in 1:s.nqp
        for ct in 0:cld(N,32)-1,rt in 0:ct
            for index in lane:128:1024
                ri=(index-1)%32+1;cj=(index-1)÷32+1
                i=32rt+ri;j=32ct+cj
                if i<=N && j<=N && i<j
                    x=delayed_inverse_entry(c,i,j,q,b,count)
                    c.inverse[i,j,q,b]=x;tile[ri,cj]=x
                elseif i==j && i<=N
                    c.inverse[i,i,q,b]=0.0
                end
            end
            sync_threads()
            for index in lane:128:1024
                ri=(index-1)%32+1;cj=(index-1)÷32+1
                i=32ct+ri;j=32rt+cj
                if i<=N && j<=N && i>j
                    c.inverse[i,j,q,b]=-tile[cj,ri]
                end
            end
            sync_threads()
        end
    end
    if lane==1
        c.delay_count[b]=0
    end
    sync_threads()
    return nothing
end
@inline function delayed_update!(s,c,a,b,::Val{N},::Val{K}) where {N,K}
    lane=threadIdx().x;count=Int(c.delay_count[b]);next=count+1
    shared=CuDynamicSharedArray(Float64,3N+2K)
    @inbounds for q in 1:s.nqp
        ra=s.configurations[a,b]+(a>s.nelec ? s.nsite : 0)+1
        for j in lane:128:N
            rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
            shared[j]=s.slater[rj,ra,q]
        end
        sync_threads()
        for k in lane:128:count
            shared[3N+k]=c.delay_dots[1,k,q,b]
            shared[3N+K+k]=c.delay_dots[2,k,q,b]
        end
        sync_threads()
        for i in lane:128:N
            value=0.0
            if i==a
                value=c.delay_ratio[q,b]
            else
                for j in 1:N
                    value=muladd(c.inverse[i,j,q,b],shared[j],value)
                end
                for k in 1:count
                    value=muladd(c.delay_u[i,k,q,b],shared[3N+K+k],muladd(-c.delay_v[i,k,q,b],shared[3N+k],value))
                    i==c.delay_site[k,b] && (value=CUDA.add_rn(value,shared[3N+K+k]))
                end
            end
            shared[N+i]=value
            shared[2N+i]=delayed_inverse_entry(c,a,i,q,b,count)
        end
        sync_threads()
        scale=CUDA.div_rn(-1.0,shared[N+a])
        for i in lane:128:N
            c.delay_u[i,next,q,b]=shared[N+i]
            c.delay_v[i,next,q,b]=CUDA.mul_rn(shared[2N+i],scale)
        end
        if lane==1
            c.pfaffians[q,b]=c.candidate_pf[q,b]
        end
        sync_threads()
    end
    if lane==1
        c.delay_site[next,b]=a
        c.delay_count[b]=next
    end
    sync_threads()
    next==K && delayed_flush!(s,c,b,Val(N),Val(K))
    return nothing
end
