using Test
using LinearAlgebra
using MVMCOptimizers
using MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData, InterAllTerm

# Independent four-electron Fock-basis oracle. No Green kernel is called here.
function normal_interall_fock_ratio(term, num, slater)
    overlap(occupation) = begin
        sites = findall(==(1), occupation) .- 1
        sum(((qp, weight),) -> begin
            a(i, j) = slater[qp * 64 + sites[i] * 8 + sites[j] + 1]
            weight * (a(1, 2) * a(3, 4) - a(1, 3) * a(2, 4) + a(1, 4) * a(2, 3))
        end, ((0, 1.0), (1, -0.375)))
    end
    moved = copy(num)
    sign = 1
    for (site, spin, create) in ((term.site3, term.spin3, false),
                               (term.site2, term.spin2, true),
                               (term.site1, term.spin1, false),
                               (term.site0, term.spin0, true))
        rs = site + 4 * spin
        moved[rs + 1] == Int(create) && return 0.0 + 0.0im
        isodd(sum(moved[1:rs])) && (sign = -sign)
        moved[rs + 1] = Int(create)
    end
    return conj(sign * overlap(moved) / overlap(num))
end

@testset "unit/normal InterAll explicit indices and spins" begin
    for complex in (false, true), idx in (Int[0, 2, 1, 3], Int[0, 3, 0, 3])
        data = ExpertModeData()
        data.modpara.nsite = 4
        data.modpara.nelec = 2
        data.modpara.nmp_trans = 2
        data.complex_flags = [Int(complex)]
        data.n_qp_trans = 2
        data.para_qp_trans = ComplexF64[1, -0.375]
        MVMCExpertModeParsers.init_qp_weight!(data)
        state = MVMCOptimizers.VMCOptimizationState(4, 2, 0, 0, 2, 1, complex, false)
        mat = state.slater_matrix
        for qp in 0:1, i in 0:7, j in (i + 1):7
            z = i ÷ 4 == j ÷ 4 ? 0.0 + 0.0im :
                ComplexF64(((17i + 13j + 7qp) % 31 - 15) / 7 + 0.125,
                           complex ? ((11i + 3j + qp) % 19 - 9) / 13 : 0.0)
            mat.slater_elm[qp * 64 + i * 8 + j + 1] = z
            mat.slater_elm[qp * 64 + j * 8 + i + 1] = -z
        end
        num = zeros(Int, 8)
        cfg = fill(-1, 8)
        for m in 0:3
            rs = idx[m + 1] + (m ÷ 2) * 4
            num[rs + 1] = 1
            cfg[rs + 1] = m % 2
        end
        if complex
            @test MVMCOptimizers.calculate_m_all_fcmp!(idx, 1, 3, data, state) == 0
            ip = MVMCOptimizers.calculate_ip_fcmp(mat.pf_m, 1, 3, data; reduce = :none)
        else
            mat.slater_elm_real .= real.(mat.slater_elm)
            @test MVMCOptimizers.calculate_m_all_real!(idx, 1, 3, data, state) == 0
            ip = ComplexF64(MVMCOptimizers.calculate_ip_real(mat.pf_m_real, 1, 3, data; reduce = :none))
        end
        original = (copy(idx), copy(cfg), copy(num), copy(mat.pf_m), copy(mat.pf_m_real),
                    copy(mat.inv_m), copy(mat.inv_m_real))
        terms = InterAllTerm[]
        for s in 0:1, t in 0:1, ri in 0:3, rj in 0:3, rk in 0:3, rl in 0:3
            term = InterAllTerm(ri, s, rj, s, rk, t, rl, t, -0.375 + 0.1875im, true)
            data.inter_all_terms = [term]
            expected = term.value * normal_interall_fock_ratio(term, num, mat.slater_elm)
            energy = MVMCOptimizers.calculate_hamiltonian(ip, idx, cfg, num, Int[], data, state;
                                                         all_complex = complex)
            @test isapprox(energy, expected; atol = 2e-12, rtol = 0)
            @test (idx, cfg, num, mat.pf_m, mat.pf_m_real, mat.inv_m, mat.inv_m_real) == original
            push!(terms, term)
        end
        # Duplicates and input order must survive accumulation. Compute this
        # expectation from separately tested Green calls, without sorting terms.
        append!(terms, terms[[1, 73, 1024]])
        data.inter_all_terms = terms
        expected = 0.0 + 0.0im
        for term in terms
            expected += term.value * MVMCOptimizers.green_func2(
                term.site0, term.site1, term.site2, term.site3, term.spin0, term.spin2,
                ip, idx, cfg, num, Int[], data, state; all_complex = complex)
        end
        actual = MVMCOptimizers.calculate_hamiltonian(ip, idx, cfg, num, Int[], data, state;
                                                     all_complex = complex)
        @test reinterpret(UInt64, [actual]) == reinterpret(UInt64, [expected])
        @test (idx, cfg, num, mat.pf_m, mat.pf_m_real, mat.inv_m, mat.inv_m_real) == original
        data.inter_all_terms = [InterAllTerm(0, 0, 1, 1, 2, 0, 3, 0, 1.0 + 0.0im, false)]
        @test_throws ArgumentError MVMCOptimizers.calculate_hamiltonian(
            ip, idx, cfg, num, Int[], data, state; all_complex = complex)
    end
end
