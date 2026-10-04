using Test
using Random
using LinearAlgebra
using MVMCOptimizers
using MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData, ModParaParameters, GutzwillerTerm, JastrowTerm

@testset "direct SR failed Cholesky never substitutes into RHS" begin
    for (matrix, expected_info) in (
        (zeros(2, 2), 1), ([1.0 1.0; 1.0 1.0], 2), ([1.0 2.0; 2.0 1.0], 2),
    )
        _, info = LAPACK.potrf!('U', copy(matrix))
        @test info == expected_info
        rhs = [-0.5, -0.5]
        @test MVMCOptimizers._solve_direct_sr!(copy(matrix), rhs) == 1
        @test rhs == [-0.5, -0.5]
    end
    rhs = [-1.0, 2.0]
    @test MVMCOptimizers._solve_direct_sr!([2.0 0.0; 0.0 4.0], rhs) == 0
    @test rhs ≈ [-0.5, 0.5] atol=2eps(Float64) rtol=0
end

# C stcopt_dposv.c:45 returns DPOSV info; stcopt.c:174 updates Para only
# on success. These literal matrices have independently known leading-minor
# failures: zeros(2,2) -> info1; [1 1;1 1] and [1 2;2 1] -> info2.
# DPOSV must not perform POTRS or overwrite its RHS after POTRF fails.
@testset "direct SR preserves parameters after positive factorization info" begin
    for complex in (false, true), matrix in (
        zeros(2, 2), [1.0 1.0; 1.0 1.0], [1.0 2.0; 2.0 1.0],
    )
        data = ExpertModeData()
        data.modpara = ModParaParameters(
            nsite=2, nelec=1, nvmc_sample=1,
            dsr_opt_red_cut=0.0, dsr_opt_sta_del=0.0, dsr_opt_step_dt=0.25,
        )
        data.gutzwiller_terms = [GutzwillerTerm(0, 3.0 + 0.25im, complex)]
        data.jastrow_terms = [JastrowTerm(0, 1, -2.0 + 0.5im, false)]
        data.complex_flags = [Int(complex)]
        @test MVMCOptimizers.get_all_complex_flag(data) == complex
        # Real and imaginary components in complex mode exercise the complex
        # production input path; two real components exercise real conversion.
        active = complex ? [0, 1] : [0, 2]
        data.optimization_flags = falses(4)
        data.optimization_flags[active .+ 1] .= true
        state = MVMCOptimizers.VMCOptimizationState(2, 1, 2, 2, 1, 1, complex, false)
        sr = state.sr_opt
        if complex
            for (i, pi) in enumerate(active), (j, pj) in enumerate(active)
                sr.sr_opt_oo[(pi+2) * (2*sr.sr_opt_size) + pj+3] = matrix[i,j]
            end
            sr.sr_opt_ho[active .+ 3] .= 1.0
        else
            for i=1:2, j=1:2
                sr.sr_opt_oo_real[(i)*sr.sr_opt_size+j+1] = matrix[i,j]
            end
            sr.sr_opt_ho_real[2:3] .= 1.0
        end
        parameters = copy(MVMCOptimizers.pack_parameters(data))
        flags = copy(data.optimization_flags)
        source_rhs = copy(complex ? sr.sr_opt_ho : sr.sr_opt_ho_real)
        before_rng = rand(copy(Random.default_rng()), UInt32, 624)
        # Actual production entry, not an isolated LAPACK success mock.
        @test MVMCOptimizers.stochastic_opt!(data, state) == 1
        @test MVMCOptimizers.pack_parameters(data) == parameters
        @test data.optimization_flags == flags
        @test rand(copy(Random.default_rng()), UInt32, 624) == before_rng
        @test (complex ? sr.sr_opt_ho : sr.sr_opt_ho_real) == source_rhs
        # Verify that the actual production real-conversion/complex inputs
        # constructed the intended system, not a different accidental zero.
        actual_s = zeros(2, 2)
        actual_rhs = zeros(2)
        MVMCOptimizers.build_s_matrix_and_g_vector!(
            actual_s, actual_rhs, active, sr.sr_opt_oo, sr.sr_opt_ho,
            sr.sr_opt_size, 0.0, 0.25,
        )
        @test actual_s == matrix
        @test actual_rhs == [-0.5, -0.5]
    end
end
