function cpu_state(data)
    ns, ne = data.modpara.nsite, data.modpara.nelec
    np = CPU.count_total_parameters(data)
    nproj = P.projection_layout(data).n_proj
    nq = CPU.get_n_qp_full(data)
    c = CPU.VMCOptimizationState(ns, ne, nproj, np, nq, 1, false, false)
    CPU.update_slater_elm_fcmp!(data, c)
    c.slater_matrix.slater_elm_real .= real.(c.slater_matrix.slater_elm)
    return c
end
function cpu_overlap(data, c, idx)
    ns, ne = data.modpara.nsite, data.modpara.nelec
    occ = zeros(Int, 2ns)
    cfg = fill(-1, 2ns)
    for i in 1:2ne
        r = idx[i] + (i > ne ? ns : 0) + 1
        occ[r] = 1
        cfg[r] = (i-1) % ne
    end
    counts = zeros(Int, P.projection_layout(data).n_proj)
    CPU.make_proj_cnt!(counts, occ, data)
    info = CPU.calculate_m_all_real!(Int.(idx), 1, CPU.get_n_qp_full(data)+1, data, c)
    ip = info == 0 ? CPU.calculate_ip_real(c.slater_matrix.pf_m_real, 1, CPU.get_n_qp_full(data)+1, data) : 0.0
    return (; ip, logcorr=CPU.log_proj_val(counts, data), counts, occ, cfg)
end
function cpu_observables(data, c, idx)
    x = cpu_overlap(data, c, idx)
    CPU.convert_from_real_arrays!(c)
    c.slater_matrix.pf_m .= c.slater_matrix.pf_m_real
    ip = ComplexF64(x.ip)
    energy = CPU.calculate_hamiltonian(ip, Int.(idx), x.cfg, x.occ, x.counts, data, c; all_complex=false)
    g1(t) = CPU.green_func1(t.site1, t.site2, t.spin1 === :up ? 0 : 1,
                           ip, Int.(idx), x.cfg, x.occ, x.counts, data, c; all_complex=false)
    g2(t) = CPU.green_func2(t.site1,t.site2,t.site3,t.site4,
                           t.spin1 === :up ? 0 : 1, t.spin3 === :up ? 0 : 1,
                           ip, Int.(idx),x.cfg,x.occ,x.counts,data,c;all_complex=false)
    out = ComplexF64[energy]
    append!(out, g1.(data.green_one_terms))
    append!(out, g2.(data.green_two_terms))
    for t in data.green_two_ex_terms
        a = CPU.green_func1(t.site_i1,t.site_j1,t.spin_i1,ip,Int.(idx),x.cfg,x.occ,x.counts,data,c;all_complex=false)
        b = CPU.green_func1(t.site_i2,t.site_j2,t.spin_i2,ip,Int.(idx),x.cfg,x.occ,x.counts,data,c;all_complex=false)
        push!(out, a*conj(b))
    end
    no = CPU.count_total_parameters(data)-length(x.counts)
    orbitalO = zeros(ComplexF64, 2no)
    CPU.slater_elm_diff_fcmp!(orbitalO, ip, Int.(idx), data, c; real_inverse=true)
    O = vcat(Float64.(x.counts), real.(orbitalO[1:2:end]))
    return real.(out), O
end

