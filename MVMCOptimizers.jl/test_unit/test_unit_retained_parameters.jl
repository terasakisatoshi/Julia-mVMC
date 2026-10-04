using Test
using MVMCOptimizers
using MVMCExpertModeParsers: ExpertModeData, OrbitalTerm, GeneralRBMPhysLayerTerm
using MVMCExpertModeParsers
using Random, SFMT
using MVMCExpertModeParsers: ChargeRBMPhysLayerTerm, SpinRBMPhysLayerTerm,
    ChargeRBMHiddenLayerTerm, SpinRBMHiddenLayerTerm, GeneralRBMHiddenLayerTerm,
    ChargeRBMPhysHiddenTerm, SpinRBMPhysHiddenTerm, GeneralRBMPhysHiddenTerm

@testset "declared unmapped slots survive actual SR update helper" begin
    data = ExpertModeData()
    data.n_gutzwiller_idx = 2
    data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, 0.0im, false, 2)]
    data.modpara.n_orbital_idx = 3
    data.orbital_terms = [OrbitalTerm(0, 1, 1, 0.0im, false), OrbitalTerm(1, 0, 1, 0.0im, false)]
    # Independent binary-exact assignment/delta expectations, not a replay oracle.
    initial = ComplexF64[1, 2, 3, 4, 5, 6, 7, 8]
    unpack_parameters!(data, initial)
    @test pack_parameters(data) == initial
    for i in [2, 3, 4, 6, 8]
        MVMCOptimizers.update_parameter_value(data, i, 0.125, -0.25)
    end
    expected = ComplexF64[1, 2.125-0.25im, 3.125-0.25im, 4.125-0.25im,
        5, 6.125-0.25im, 7, 8.125-0.25im]
    @test pack_parameters(data) == expected
    @test pack_parameters(deepcopy(data)) == expected
    set_parameter_value!(data, 7, 0.5 + 0.125im)
    @test all(t.value == 0.5 + 0.125im for t in data.orbital_terms)
    data.orbital_terms[end].value = 0.25im
    @test pack_parameters(data)[7] == 0.25im # mapped last-wins gather
    set_parameter_value!(data, 3, 0.0im)
    @test pack_parameters(data)[3] == 0
end

@testset "same-width active reinitialization matches independent C InitParameter" begin
    data = ExpertModeData()
    data.n_gutzwiller_idx = 2
    data.modpara.nneuron = 1
    data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, 0.0im, false, 2)]
    data.modpara.n_orbital_idx = 3
    data.orbital_terms = [OrbitalTerm(0, 1, 1, 0.0im, false)]
    data.optimization_flags = Bool[0,0,0,0,1,0,0,0,1,0,1,0,0,0,1,0]
    flags = copy(data.optimization_flags)
    lines = readlines(joinpath(@__DIR__, "fixtures", "c_retained_init.txt"))
    parameters = [ComplexF64(parse.(Float64, split(line))...) for line in lines[1:8]]
    next624 = parse.(UInt32, lines[9:end])
    @test length(next624) == 624
    for _ in 1:2
        unpack_parameters!(data, fill(12.5 + 0.25im, 8))
        rng = SFMT19937RNG(); Random.seed!(rng, 11272)
        MVMCExpertModeParsers.init_parameter!(data; rng=rng)
        @info "C InitParameter computed first divergence" maximum_absolute_difference=maximum(abs, pack_parameters(data) - parameters)
        # Real RBM scale <= .005: two ulps relative / eps*.01 absolute cover
        # this single scaled draw, not a Monte Carlo or solver discrepancy.
        @test pack_parameters(data) ≈ parameters atol=eps(Float64)*0.01 rtol=2eps(Float64)
        @test data.optimization_flags == flags
        @test [rand(rng, UInt32) for _ in 1:624] == next624
    end
end

@testset "short dense arrays retain declared DH slots; full projection gauge" begin
    data = ExpertModeData()
    data.n_gutzwiller_idx = 2
    data.n_jastrow_idx = 2
    data.doublon_holon_2site_indices = [MVMCExpertModeParsers.DoublonHolon2SiteIndex(zeros(Int, 0, 2))]
    data.doublon_holon_2site_params = ComplexF64[0]
    unpack_parameters!(data, ComplexF64[1, 3, 5, 7, 9, 10, 11, 12, 13, 14])
    data.optimization_flags = falses(20) # no gauge shift: test short DH prefix preservation
    MVMCOptimizers.sync_modified_parameter!(data)
    @test pack_parameters(data) == ComplexF64[1, 3, 5, 7, 9, 10, 11, 12, 13, 14]
    # Independently specified mean over ALL four declared projection slots is 4.
    data.optimization_flags[1:2:7] .= true
    MVMCOptimizers.sync_modified_parameter!(data)
    @test pack_parameters(data)[1:4] == ComplexF64[-3, -1, 1, 3]
    @test pack_parameters(data)[5:10] == ComplexF64[9, 10, 11, 12, 13, 14]
    # Same total NPara, different family widths must not reinterpret stale offsets.
    data.n_gutzwiller_idx = 3; data.n_jastrow_idx = 1
    @test pack_parameters(data)[1:4] == zeros(ComplexF64, 4)
    @test pack_parameters(data)[5:10] == ComplexF64[9, 10, 11, 12, 13, 14]
end

@testset "all nine actual RBM overlay keywords and Parallel offset" begin
    data = ExpertModeData()
    data.charge_rbm_phys_layer_terms = [ChargeRBMPhysLayerTerm(0, 0im, false, 2)]
    data.spin_rbm_phys_layer_terms = [SpinRBMPhysLayerTerm(0, 0im, false, 2)]
    data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, 0im, false, 2)]
    data.charge_rbm_hidden_layer_terms = [ChargeRBMHiddenLayerTerm(0, 0im, false, 2)]
    data.spin_rbm_hidden_layer_terms = [SpinRBMHiddenLayerTerm(0, 0im, false, 2)]
    data.general_rbm_hidden_layer_terms = [GeneralRBMHiddenLayerTerm(0, 0im, false, 2)]
    data.charge_rbm_phys_hidden_terms = [ChargeRBMPhysHiddenTerm(0, 0, 0im, false, 2)]
    data.spin_rbm_phys_hidden_terms = [SpinRBMPhysHiddenTerm(0, 0, 0im, false, 2)]
    data.general_rbm_phys_hidden_terms = [GeneralRBMPhysHiddenTerm(0, 0, 0, 0im, false, 2)]
    data.modpara.n_orbital_idx = 4
    data.n_orbital_anti_parallel = 2
    data.i_flg_orbital_anti_parallel = 1
    data.i_flg_orbital_parallel = 1
    data.orbital_terms = [OrbitalTerm(0, 1, 0, 0.0im, false), OrbitalTerm(1, 0, 3, 0.0im, false)]
    original = ComplexF64.(1:31)
    unpack_parameters!(data, original)
    keywords = ["InChargeRBM_PhysLayer", "InSpinRBM_PhysLayer", "InGeneralRBM_PhysLayer",
        "InChargeRBM_HiddenLayer", "InSpinRBM_HiddenLayer", "InGeneralRBM_HiddenLayer",
        "InChargeRBM_PhysHidden", "InSpinRBM_PhysHidden", "InGeneralRBM_PhysHidden"]
    expected = copy(original)
    mktempdir() do dir
        lines = String[]
        for (i, keyword) in enumerate(keywords)
            filename = "rbm$i.def"
            push!(lines, "$keyword $filename")
            write(joinpath(dir, filename), "===\nNParameter 2\n===\n===\n===\n0 $i 0.125\n1 0 0\n")
            expected[3i-2] = i + 0.125im
            expected[3i-1] = 0
        end
        push!(lines, "InOrbitalParallel parallel.def")
        write(joinpath(dir, "parallel.def"), "===\nNParameter 2\n===\n===\n===\n0 0.5 0\n1 0 0\n")
        expected[30] = 0.5; expected[31] = 0
        write(joinpath(dir, "namelist.def"), join(lines, "\n") * "\n")
        MVMCExpertModeParsers.read_input_parameters!(data, joinpath(dir, "namelist.def"))
        @test pack_parameters(data) == expected
        @test pack_parameters(data)[28:29] == original[28:29]
    end
    flags = copy(data.optimization_flags)
    unpack_parameters!(data, pack_parameters(deepcopy(data))) # MPI broadcast payload path
    @test pack_parameters(data) == expected
    @test data.optimization_flags == flags
    malformed = "0 0 0 0 0 0 " * join((i == 16 ? "bad 0 0" : "$i 0 0" for i in 1:31), " ")
    @test !first(MVMCOptimizers._load_para_triples!(data, malformed))
    @test pack_parameters(data) == expected
end

@testset "retained lifecycle, independent declared C order" begin
    data = ExpertModeData()
    data.n_gutzwiller_idx = 2
    data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, 0.0im, false, 2)]
    data.modpara.n_orbital_idx = 3
    data.orbital_terms = [OrbitalTerm(0, 1, 1, 0.0im, false)]
    unpack_parameters!(data, ComplexF64[1, 2, 3, 4, 5, 8, 2, 4])
    MVMCOptimizers.sync_modified_parameter!(data; shift_correlations=false)
    @test pack_parameters(data) == ComplexF64[1, 2, 3, 4, 5, 4, 1, 2]
    # Parser-side normalization must also see the unmapped maximum.
    unpack_parameters!(data, ComplexF64[1, 2, 3, 4, 5, 16, 2, 4])
    MVMCExpertModeParsers.sync_modified_parameter!(data)
    @test pack_parameters(data) == ComplexF64[1, 2, 3, 4, 5, 4, 0.5, 1]
    before = pack_parameters(data)
    @test !first(MVMCOptimizers._load_para_triples!(data, "0 0 0"))
    @test pack_parameters(data) == before
    # Full fixed record preserves holes and explicit zero values.
    record = "0 0 0 0 0 0 " * join(("$i 0 0" for i in 1:8), " ")
    @test first(MVMCOptimizers._load_para_triples!(data, record))
    @test pack_parameters(data) == ComplexF64.(1:8)
    mktempdir() do dir
        write(joinpath(dir, "namelist.def"), "InGutzwiller g.def\nInGeneralRBM_PhysLayer r.def\nInOrbital o.def\n")
        # parse_input_parameter_file accepts simple indexed records after header.
        header(n) = "================\nNParameter $n\n================\n================\n================\n"
        write(joinpath(dir, "g.def"), header(1) * "1 0.0 0.0\n")
        write(joinpath(dir, "r.def"), header(2) * "0 0.5 0.25\n1 0.0 0.0\n")
        write(joinpath(dir, "o.def"), header(2) * "0 0.25 0.0\n2 0.0 0.0\n")
        MVMCExpertModeParsers.read_input_parameters!(data, joinpath(dir, "namelist.def"))
        @test pack_parameters(data) == ComplexF64[1, 0, 0.5+0.25im, 0, 5, 0.25, 7, 0]
    end
    # Family-width invalidation is a Julia API behavior, not a C file extension.
    data.n_gutzwiller_idx = 3
    @test pack_parameters(data)[1:3] == zeros(ComplexF64, 3)
    @test pack_parameters(data)[4:6] == ComplexF64[0.5+0.25im, 0, 5]
    flags = falses(2 * count_total_parameters(data))
    data.optimization_flags = copy(flags)
    r1 = SFMT19937RNG(); Random.seed!(r1, 11272)
    MVMCExpertModeParsers.init_parameter!(data; rng=r1)
    @test pack_parameters(data) == zeros(ComplexF64, 9)
    @test data.optimization_flags == flags
    actual_rng = [rand(r1, UInt32) for _ in 1:624]
    # SFMT's C-backed state is global, so seed the independent stream AFTER capture.
    r2 = SFMT19937RNG(); Random.seed!(r2, 11272)
    @test actual_rng == [rand(r2, UInt32) for _ in 1:624]
end

@testset "actual direct and CG solver retain unmapped families" begin
    for cg in (false, true), target in (2, 3, 6)
        data = ExpertModeData()
        data.n_gutzwiller_idx = 2
        data.general_rbm_phys_layer_terms = [GeneralRBMPhysLayerTerm(0, 0, 0.0im, false, 2)]
        data.modpara.n_orbital_idx = 3
        data.orbital_terms = [OrbitalTerm(0, 1, 1, 0.0im, false)]
        data.modpara.dsr_opt_red_cut = 0.0
        data.modpara.dsr_opt_sta_del = 0.0
        data.modpara.dsr_opt_step_dt = 0.5
        data.optimization_flags = falses(16)
        data.optimization_flags[2 * target - 1] = true
        state = MVMCOptimizers.VMCOptimizationState(2, 1, 2, 8, 1, 2, !cg, false)
        size = state.sr_opt.sr_opt_size
        if cg
            state.energy.wc = 4.0 + 0im
            state.sr_opt.sr_opt_oo_real[size + target + 1] = 2.0
            state.sr_opt.sr_opt_ho_real[target + 1] = 1.0
            state.sr_opt.sr_opt_o_store_real[target + 1] = 2.0
            state.sr_opt.sr_opt_o_store_real[size + target + 1] = 2.0
            @test MVMCOptimizers.stochastic_opt_cg!(data, state) == 0
        else
            pi = 2 * (target - 1)
            state.sr_opt.sr_opt_oo[(pi + 2) * (2 * size) + pi + 3] = 2.0
            state.sr_opt.sr_opt_ho[pi + 3] = 1.0
            @test MVMCOptimizers.stochastic_opt!(data, state) == 0
        end
        # C S=2, g=-2*dt*HO=-1 => delta=-1/2; no oracle replay.
        expected = zeros(ComplexF64, 8); expected[target] = -0.5
        @test pack_parameters(data) ≈ expected atol=1e-14 rtol=0
        @test count(data.optimization_flags) == 1
    end
end
