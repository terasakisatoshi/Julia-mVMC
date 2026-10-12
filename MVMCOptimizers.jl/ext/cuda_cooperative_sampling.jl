# The block owns a walker, including all proposal and acceptance decisions.
# Only the Pfaffian trailing updates are distributed across its threads.
@inline function cooperative_overlap!(s, b)
    n = 2s.nelec
    lane = threadIdx().x
    stride = blockDim().x
    ip = 0.0
    @inbounds for q in 1:s.nqp
        for index in lane:stride:n*n
            i = (index-1)%n+1
            j = (index-1)÷n+1
            ri = s.configurations[i,b]+(i > s.nelec ? s.nsite : 0)+1
            rj = s.configurations[j,b]+(j > s.nelec ? s.nsite : 0)+1
            s.cooperative_factors[i,j,b] = -s.slater[ri,rj,q]
        end
        sync_threads()
        pf = cooperative_pfaffian!(s.cooperative_factors, s.pivots, n, b)
        ip = CUDA.add_rn(ip,CUDA.mul_rn(s.weights[q],pf))
        sync_threads()
    end
    return ip
end

# Passing the host CUDAState to a kernel is not supported; a named tuple of
# scalar metadata and device buffers lets CUDA adapt every owned array.
function cooperative_state(s)
    return (; nsite=s.nsite, nelec=s.nelec, nqp=s.nqp, ngutz=s.ngutz,
        njastrow=s.njastrow, configurations=s.configurations, counts=s.counts,
        occupancy=s.occupancy, gutz=s.gutz, jastrow=s.jastrow,
        parameters=s.parameters, slater=s.slater, weights=s.weights,
        cooperative_factors=s.cooperative_factors, pivots=pivot_view(s.pivots),
        overlaps=s.overlaps, logcorrelations=s.logcorrelations,
        accepted=s.accepted, localspins=s.localspins)
end

function cooperative_wavefunction_kernel!(s)
    b = blockIdx().x
    if threadIdx().x == 1
        @inbounds s.logcorrelations[b] = correlation!(s.counts,s.occupancy,
            s.configurations,s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,
            s.ngutz,s.njastrow,b)
    end
    sync_threads()
    ip = cooperative_overlap!(s,b)
    if threadIdx().x == 1
        @inbounds s.overlaps[b] = ip
    end
    return nothing
end

function cooperative_sampling_kernel!(s, tape, exchange_fraction)
    b = blockIdx().x
    lane = threadIdx().x
    # Shared proposal state is written by one lane, then read by the block.
    move = CuStaticSharedArray(Int32, 5) # valid, i, j, old site i, old site j
    values = CuStaticSharedArray(Float64, 3) # old overlap/log, new log
    if lane == 1
        @inbounds begin
            values[1] = s.overlaps[b]
            values[2] = s.logcorrelations[b]
            s.accepted[b] = 0
        end
    end
    sync_threads()
    @inbounds for step in 1:size(tape,3)
        if lane == 1
            spin = tape[2,b,step] < UInt32(0x80000000) ? 0 : 1
            i = Int(tape[1,b,step]%UInt32(s.nelec))+1+spin*s.nelec
            ri = s.configurations[i,b]
            exchange = Float64(tape[4,b,step])/4294967296.0 < exchange_fraction
            j = 0
            valid = true
            if exchange
                j = Int(tape[3,b,step]%UInt32(s.nelec))+1+(1-spin)*s.nelec
                rj = s.configurations[j,b]
                valid = ri != rj
                for k in 1:s.nelec
                    if s.configurations[k+spin*s.nelec,b] == rj ||
                       s.configurations[k+(1-spin)*s.nelec,b] == ri
                        valid = false
                    end
                end
            else
                rj = Int(tape[3,b,step]%UInt32(s.nsite))
                valid = s.localspins[ri+1] == 0 && s.localspins[rj+1] == 0
                for k in 1:s.nelec
                    if s.configurations[k+spin*s.nelec,b] == rj
                        valid = false
                    end
                end
            end
            move[1] = valid
            move[2] = i
            move[3] = j
            move[4] = ri
            move[5] = rj
            if valid
                s.configurations[i,b] = rj
                j != 0 && (s.configurations[j,b] = ri)
                values[3] = correlation!(s.counts,s.occupancy,s.configurations,
                    s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
            end
        end
        sync_threads()
        if move[1] != 0
            ip = cooperative_overlap!(s,b)
            if lane == 1
                logratio = 2*(values[3]-values[2]+log(abs(ip)+1e-100)-log(abs(values[1])+1e-100))
                u = Float64(tape[5,b,step])/4294967296.0
                if ip != 0 && log(u) < logratio
                    values[1] = ip
                    values[2] = values[3]
                    s.accepted[b] += 1
                else
                    s.configurations[move[2],b] = move[4]
                    move[3] != 0 && (s.configurations[move[3],b] = move[5])
                end
            end
        end
        sync_threads()
    end
    if lane == 1
        @inbounds begin
            s.overlaps[b] = values[1]
            s.logcorrelations[b] = correlation!(s.counts,s.occupancy,s.configurations,
                s.gutz,s.jastrow,s.parameters,s.nsite,s.nelec,s.ngutz,s.njastrow,b)
        end
    end
    return nothing
end
