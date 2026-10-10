using Test, MVMCOptimizers, MVMCExpertModeParsers

@testset "Slater derivatives promote real inverse entries without a matrix copy" begin
    withenv("JULIA_MVMC_INNER_THREADS" => "1") do
        for size in (16, 32, 64)
            nqp, nsp, norb = 8, 4, 19
            nelec = size ÷ 2
            indices = [mod(i * 7 + i ÷ size, norb) for i in 1:(2 * size^2)]
            signs = [isodd(i) ? -1 : 1 for i in eachindex(indices)]
            inverse = [(i % 29 - 14) / 32 for i in 1:(nqp * size^2)]
            original = copy(inverse)
            pf = [ComplexF64(q / 16, (q - 5) / 32) for q in 1:nqp]
            cs = [ComplexF64(s / 16, (s - 2) / 32) for s in 1:nsp]
            cc = [ComplexF64((s + 2) / 8, (s - 1) / 16) for s in 1:nsp]
            ss = [ComplexF64((s - 3) / 8, (s + 1) / 16) for s in 1:nsp]
            expected = zeros(ComplexF64, nqp * norb)
            for q in 1:nqp
                spin = mod(q - 1, nsp) + 1
                trans = (q - 1) ÷ nsp
                for row in 1:size, column in 1:size
                    coefficient = row <= nelec ?
                        (column <= nelec ? pf[q] * cs[spin] : -pf[q] * cc[spin]) :
                        (column <= nelec ? pf[q] * ss[spin] : -pf[q] * cs[spin])
                    mapping = trans * size^2 + (row - 1) * size + column
                    orbital = indices[mapping] + 1
                    expected[(q - 1) * norb + orbital] +=
                        (ComplexF64(inverse[(q - 1) * size^2 + (row - 1) * size + column], 0.0) * coefficient) * signs[mapping]
                end
            end
            buffer = zeros(ComplexF64, nqp * norb)
            arguments = (indices, signs, inverse, pf, cs, cc, ss,
                         nelec, size, size^2, nqp, nsp, norb)
            budget = 8eps(Float64) * (1 + maximum(abs, expected))
            for repeat in 1:12
                fill!(buffer, 0)
                MVMCOptimizers._accumulate_slater_buffer_fcmp_fast!(buffer, arguments...)
                # Dyadic inputs/products are exactly representable at this
                # scale; the small explicit budget avoids bitwise FP comparison.
                @test isapprox(buffer, expected; atol=budget, rtol=8eps(Float64))
            end
            @test inverse == original
            observed = fetch(Threads.@spawn begin
                result = zeros(ComplexF64, length(buffer))
                MVMCOptimizers._accumulate_slater_buffer_fcmp_fast!(result, arguments...)
                result
            end)
            @test isapprox(observed, expected; atol=budget, rtol=8eps(Float64))
        end
    end

    data = MVMCExpertModeParsers.ExpertModeData()
    data.modpara = MVMCExpertModeParsers.ModParaParameters(
        nsite=2, nelec=1, nmp_trans=1, nsp_gauss_leg=1,
        n_orbital_idx=4, complex_flag=0,
    )
    data.qp_weights = MVMCExpertModeParsers.QuantumProjectionWeights()
    data.qp_weights.qp_full_weight = ComplexF64[1 + im/4]
    data.qp_weights.spgl_cos_sin = ComplexF64[1 + im/8]
    data.qp_weights.spgl_cos_cos = ComplexF64[1 - im/8]
    data.qp_weights.spgl_sin_sin = ComplexF64[1 + im/16]
    data.qp_trans = [[0, 1]]
    data.qp_trans_sgn = [[1, 1]]
    data.n_qp_opt_trans = 1
    data.qp_opt_trans = [[0, 1]]
    data.qp_opt_trans_sgn = [[1, 1]]
    data.orbital_idx_matrix = [0 1; 2 3]
    data.orbital_sgn = ones(Int, 2, 2)
    state = MVMCOptimizers.VMCOptimizationState(2, 1, 0, 4, 1, 1, false, false)
    state.slater_matrix.inv_m_real[1:4] .= [0.0, 0.5, -0.5, 0.0]
    state.slater_matrix.inv_m[1:4] .= ComplexF64.(state.slater_matrix.inv_m_real[1:4])
    state.slater_matrix.pf_m[1] = 2.0 + 0im
    expected = zeros(ComplexF64, 8)
    actual = similar(expected)
    scratch = MVMCOptimizers.VMCMainCalScratch(state)
    MVMCOptimizers.slater_elm_diff_fcmp!(expected, 1.5 + im/4, [0, 1], data, state, scratch)
    # Stale complex storage must never be consulted by the real-storage path.
    fill!(state.slater_matrix.inv_m, ComplexF64(NaN, NaN))
    MVMCOptimizers.slater_elm_diff_fcmp!(actual, 1.5 + im/4, [0, 1], data, state, scratch;
                                      real_inverse=true)
    @test all(isfinite, actual)
    @test isapprox(actual, expected; atol=8eps(Float64), rtol=8eps(Float64))
end
