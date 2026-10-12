# Production initialization uses cuRAND UInt32 words on the selected device.
# CPU SFMT initialization remains a separate reference path, not this RNG stream.
@inline function initial_bounded_word(tape,b,cursor,bound)
    threshold = (UInt32(0)-UInt32(bound)) % UInt32(bound)
    @inbounds while cursor <= size(tape,1)
        word = tape[cursor,b]
        cursor += 1
        word >= threshold && return (Int(word%UInt32(bound)),cursor,true)
    end
    return (0,cursor,false)
end

function initial_configuration_kernel!(cfg,scratch,localspins,tape,pending,candidate,ns,ne,batch)
    b = (blockIdx().x-1)*blockDim().x+threadIdx().x
    b > batch && return nothing
    @inbounds pending[b] || return nothing
    cursor = 1
    good = true
    @inbounds begin
        for a in 1:2ne
            cfg[a,b] = -1
            scratch[a,b] = a-1
        end
        # A uniform permutation of electron labels assigns local-spin sites.
        for k in 2ne:-1:2
            j,cursor,ok = initial_bounded_word(tape,b,cursor,k)
            good &= ok
            scratch[k,b],scratch[j+1,b] = scratch[j+1,b],scratch[k,b]
        end
        nlocal = 0
        nonlocal = 0
        for site in 0:ns-1
            if localspins[site+1] == 1
                nlocal += 1
                cfg[scratch[nlocal,b]+1,b] = site
            else
                nonlocal += 1
                scratch[2ne+nonlocal,b] = site
            end
        end
        for spin in 0:1
            unused = 0
            for a in 1:ne
                unused += cfg[spin*ne+a,b] == -1
            end
            if unused > nonlocal
                good = false
            end
            # Shuffle the nonlocal sites independently for each spin. Applying
            # a fresh uniform permutation to the previous order remains uniform.
            for k in nonlocal:-1:2
                j,cursor,ok = initial_bounded_word(tape,b,cursor,k)
                good &= ok
                x,y = 2ne+k,2ne+j+1
                scratch[x,b],scratch[y,b] = scratch[y,b],scratch[x,b]
            end
            nextsite = 1
            for a in 1:ne
                if cfg[spin*ne+a,b] == -1
                    if nextsite <= nonlocal
                        cfg[spin*ne+a,b] = scratch[2ne+nextsite,b]
                    end
                    nextsite += 1
                end
            end
        end
        if !good
            # Keep every address valid for the amplitude kernel. This placeholder
            # is never published as a valid walker and is retried next iteration.
            for a in 1:ne
                cfg[a,b] = a-1
                cfg[a+ne,b] = a-1
            end
        end
        candidate[b] = good
    end
    return nothing
end

function initial_gpu_state(data::ExpertModeData,walkers::Integer;seed::Integer=79,max_attempts::Integer=100,
                           return_pfaffians::Bool=false)
    walkers > 0 && max_attempts > 0 || throw(ArgumentError("invalid GPU initialization lengths"))
    ns = data.modpara.nsite
    ne = data.modpara.nelec == 0 ? (data.modpara.nlocspin+data.modpara.ncond)÷2 : data.modpara.nelec
    localspins = CPU.get_loc_spn_array(data)
    nlocal = count(==(1),localspins)
    nlocal <= 2ne <= 2ns-nlocal || throw(ArgumentError("infeasible local-spin filling"))
    placeholder = repeat(Int32.(vcat(collect(0:ne-1),collect(0:ne-1))),1,walkers)
    s = CPU.gpu_state(data;configurations=placeholder,initialize_wavefunctions=false)
    CUDA.seed!(seed)
    scratch = CUDA.zeros(Int32,2ne+ns,walkers)
    pending = CUDA.fill(true,walkers)
    candidate = CUDA.zeros(Bool,walkers)
    pfaffians = return_pfaffians ? CUDA.zeros(Float64,s.nqp,walkers) : nothing
    remaining = walkers
    for attempt in 1:max_attempts
        tape = CUDA.rand(UInt32,2ne+2ns+16,walkers)
        @cuda threads=128 blocks=cld(walkers,128) fastmath=false initial_configuration_kernel!(
            s.configurations,scratch,s.localspins,tape,pending,candidate,ns,ne,walkers)
        @cuda threads=128 blocks=walkers fastmath=false initial_wavefunction_kernel!(
            initial_state_view(s),pending,candidate,pfaffians)
        remaining = count(identity,pending)
        remaining == 0 && return (return_pfaffians ? (s,pfaffians) : s)
    end
    throw(ArgumentError("GPU initialization failed after $max_attempts attempts ($remaining/$walkers unresolved walkers)"))
end

# C CalculateMAll initialization requires every QP factor nonsingular, in
# addition to the physically nonzero projected sum. No inverse is needed yet.
function initial_wavefunction_kernel!(s,pending,candidate,pfaffians=nothing)
    b = blockIdx().x
    @inbounds (!pending[b] || !candidate[b]) && return nothing
    lane,stride = threadIdx().x,blockDim().x
    n = 2s.nelec
    if lane == 1
        @inbounds s.logcorrelations[b] = correlation!(s.counts,s.occupancy,
            s.configurations,s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,
            s.ngutz,s.njastrow,b)
    end
    sync_threads()
    ip = 0.0
    valid = true
    @inbounds for q in 1:s.nqp
        for index in lane:stride:n*n
            i=(index-1)%n+1; j=(index-1)÷n+1
            ri=s.configurations[i,b]+(i>s.nelec ? s.nsite : 0)+1
            rj=s.configurations[j,b]+(j>s.nelec ? s.nsite : 0)+1
            s.cooperative_factors[i,j,b]=-s.slater[ri,rj,q]
        end
        sync_threads()
        pf=cooperative_pfaffian!(s.cooperative_factors,s.pivots,n,b)
        if pfaffians !== nothing && lane == 1
            pfaffians[q,b]=pf
        end
        valid &= isfinite(pf) && pf != 0
        ip=CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],pf))
        sync_threads()
    end
    if lane == 1
        @inbounds begin
            s.overlaps[b]=ip
            pending[b]=!valid || !isfinite(ip) || ip==0 || !isfinite(s.logcorrelations[b])
        end
    end
    return nothing
end

function initial_state_view(s)
    factors = s.nsite >= 64 ? s.cooperative_factors : factor_view(s.factors)
    return merge(cooperative_state(s),(;cooperative_factors=factors))
end
