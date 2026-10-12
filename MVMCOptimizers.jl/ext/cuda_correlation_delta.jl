# Evaluate Gutzwiller/Jastrow changes from the base occupation. At most four
# sites change for an ordinary Green operator; unchanged pairs cancel exactly.
@inline function operator_delta(op, site, spin)
    delta = Int32(0)
    if op[3] == spin
        delta += (op[1] == site) - (op[2] == site)
    end
    if op[7] == 2 && op[6] == spin
        delta += (op[4] == site) - (op[5] == site)
    end
    return delta
end

@inline function operator_correlation_delta(op, occ, gutz, jastrow, parameters, ns, ng, nj, b)
    sites = (op[1], op[2], op[4], op[5])
    count = op[7] == 1 ? 2 : 4
    result = 0.0
    @inbounds for k in 1:count
        r = sites[k]
        unique = true
        for l in 1:k-1
            unique &= sites[l] != r
        end
        unique || continue
        du = operator_delta(op,r,0)
        dd = operator_delta(op,r,1)
        u = occ[r+1,b]
        d = occ[ns+r+1,b]
        p = gutz[r+1]+1
        if 1 <= p <= ng
            result += parameters[p]*((u+du)*(d+dd)-u*d)
        end
        dr = du+dd
        xr = u+d-1
        for t in 0:ns-1
            t == r && continue
            changed = false
            for l in 1:count
                changed |= sites[l] == t
            end
            changed && t < r && continue
            dt = operator_delta(op,t,0)+operator_delta(op,t,1)
            xt = occ[t+1,b]+occ[ns+t+1,b]-1
            p = jastrow[min(r,t)+1,max(r,t)+1]+1
            if 1 <= p <= nj
                result += parameters[ng+p]*(dr*xt+xr*dt+dr*dt)
            end
        end
    end
    return result
end
