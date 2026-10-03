using Test
using MVMCOptimizers
using MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData, GutzwillerTerm, JastrowTerm,
    OrbitalTerm, DoublonHolon2SiteIndex, DoublonHolon4SiteIndex,
    ChargeRBMPhysLayerTerm, SpinRBMPhysLayerTerm, GeneralRBMPhysLayerTerm,
    ChargeRBMHiddenLayerTerm, SpinRBMHiddenLayerTerm, GeneralRBMHiddenLayerTerm,
    ChargeRBMPhysHiddenTerm, SpinRBMPhysHiddenTerm, GeneralRBMPhysHiddenTerm
using MVMCOptimizers: VMCOptimizationState

@testset "unit/data_io: C full declared Para var output" begin
    coefficient(i) = ComplexF64((i + 1) * 0.125, -(i + 1) * 0.0625)
    data = ExpertModeData()
    data.n_gutzwiller_idx = 2
    data.n_jastrow_idx = 2
    data.gutzwiller_terms = [GutzwillerTerm(0, coefficient(0), true)]
    data.jastrow_terms = [JastrowTerm(0, 1, coefficient(2), true)]
    data.doublon_holon_2site_indices = [DoublonHolon2SiteIndex(zeros(Int, 0, 2))]
    data.doublon_holon_4site_indices = [DoublonHolon4SiteIndex(zeros(Int, 0, 4))]
    data.doublon_holon_2site_params = coefficient.(4:9)
    data.doublon_holon_4site_params = coefficient.(10:19)
    data.charge_rbm_phys_layer_terms = [ChargeRBMPhysLayerTerm(0, coefficient(20), true, 0)]
    data.spin_rbm_phys_layer_terms = [SpinRBMPhysLayerTerm(0, coefficient(21), true, 0)]
    data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, coefficient(22), true, 0)]
    data.charge_rbm_hidden_layer_terms = [ChargeRBMHiddenLayerTerm(0, coefficient(23), true, 0)]
    data.spin_rbm_hidden_layer_terms = [SpinRBMHiddenLayerTerm(0, coefficient(24), true, 0)]
    data.general_rbm_hidden_layer_terms = [GeneralRBMHiddenLayerTerm(0, coefficient(25), true, 0)]
    data.charge_rbm_phys_hidden_terms = [ChargeRBMPhysHiddenTerm(0, 0, coefficient(26), true, 0)]
    data.spin_rbm_phys_hidden_terms = [SpinRBMPhysHiddenTerm(0, 0, coefficient(27), true, 0)]
    data.general_rbm_phys_hidden_terms = [GeneralRBMPhysHiddenTerm(0, 0, 0, coefficient(28), true, 0)]
    data.modpara.n_orbital_idx = 2
    data.orbital_terms = [
        OrbitalTerm(0, 0, 0, coefficient(29), true),
        OrbitalTerm(0, 1, 1, coefficient(30), true),
        OrbitalTerm(1, 1, 0, coefficient(29), true), # Same stored slot, not another Para.
    ]
    data.opt_trans = coefficient.(31:32)
    state = VMCOptimizationState(2, 1, 20, 33, 1, 1, true, false)
    state.energy.etot = -3.0 + 0.0im
    state.energy.etot2 = 9.0 + 0.0im
    expected = read(joinpath(@__DIR__, "fixtures", "c_var_full.dat"), String)
    @test length(split(expected)) == 105 # C: 6 + 3*NPara, NPara=33.
    mktempdir() do output
        for (step, copies) in [(0, 1), (1, 2), (0, 1)]
            MVMCOptimizers.output_data!(data, state, step; output_dir = output)
            @test read(joinpath(output, "zvo_var.dat"), String) == repeat(expected, copies)
        end
        @test endswith(expected, "0.0 \n")
        @test !endswith(expected, "\n\n")
    end

    @testset "unmapped declared orbital slots, no duplicate-row expansion" begin
        sparse = ExpertModeData()
        sparse.n_gutzwiller_idx = 3
        sparse.n_jastrow_idx = 2
        sparse.gutzwiller_terms = [GutzwillerTerm(0, 1.25 - 0.5im, true)]
        sparse.jastrow_terms = [JastrowTerm(0, 1, 0.25 + 0.5im, true)]
        sparse.modpara.n_orbital_idx = 4
        sparse.orbital_terms = [
            OrbitalTerm(0, 1, 1, 0.75 - 0.125im, true),
            OrbitalTerm(1, 0, 1, 0.75 - 0.125im, true),
        ]
        sparse.opt_trans = ComplexF64[0.25, -0.5 + 0.25im]
        # Explicit C-offset contract, independent of pack_parameters/the writer.
        parameters = ComplexF64[1.25-0.5im, 0, 0, 0.25+0.5im, 0,
            0, 0.75-0.125im, 0, 0, 0.25, -0.5+0.25im]
        mktempdir() do output
            MVMCOptimizers.output_data!(sparse, state, 0; output_dir = output)
            values = parse.(Float64, split(read(joinpath(output, "zvo_var.dat"), String)))
            @test length(values) == 39 # C: 6+3*(Gutz3+Jast2+Slater4+OptTrans2).
            @test values[1:6] == [-3, 0, 0, 9, 0, 0]
            for (index, parameter) in enumerate(parameters)
                @test values[6+3index-2:6+3index] == [real(parameter), imag(parameter), 0]
            end
        end
    end
end
