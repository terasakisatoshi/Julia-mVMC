using Test
using MVMCOptimizers
using MVMCExpertModeParsers: ExpertModeData, OrbitalTerm
using MVMCExpertModeParsers: GutzwillerTerm, JastrowTerm, DoublonHolon2SiteIndex,
    DoublonHolon4SiteIndex, ChargeRBMPhysLayerTerm
using MVMCOptimizers: VMCOptimizationState, OptDataPoint

# Independent literal C avevar.c contract: n=2+NPara; NSROptItrSmp=1
# emits real/zero pairs, otherwise complex means and sample deviations.
@testset "unit/data_io: C optimization window" begin
    data = ExpertModeData()
    data.modpara.n_orbital_idx = 1
    data.modpara.nsr_opt_itr_smp = 1
    data.orbital_terms = [OrbitalTerm(0, 0, 0, 7.0 + 2.0im, true),
        OrbitalTerm(1, 1, 0, 7.0 + 2.0im, true)]
    state = VMCOptimizationState(2, 1, 0, 1, 1, 1, false, false)
    state.energy.etot = -3.0 + 0.25im
    state.energy.etot2 = 11.0 + 0.5im # Not E² inferred from E.
    MVMCOptimizers.store_opt_data!(data, state, 0)
    @test state.opt_data[1].parameters == ComplexF64[7 + 2im]
    @test state.opt_data[1].energy_squared == 11 + 0.5im
    unpack_parameters!(data, ComplexF64[99])
    @test state.opt_data[1].parameters == ComplexF64[7 + 2im]
    @test_throws ArgumentError MVMCOptimizers.store_opt_data!(data, state, 2)
    mktempdir() do output
        MVMCOptimizers.output_opt_data!(data, state; output_dir=output)
        lines = readlines(joinpath(output, "zqp_opt.dat"))
        @test length(lines) == 1
        @test parse.(Float64, split(only(lines))) == [-3, 0, 11, 0, 7, 0]
        @test readdir(output) == ["zqp_opt.dat"] # C creates no children for window1.
    end

    data.modpara.nsr_opt_itr_smp = 2
    state.opt_data = [OptDataPoint(-3 + 1im, 11 + 2im, ComplexF64[7 + 2im]),
        OptDataPoint(1 + 3im, 15 + 6im, ComplexF64[11 + 6im])]
    mktempdir() do output
        MVMCOptimizers.output_opt_data!(data, state; output_dir=output)
        lines = readlines(joinpath(output, "zqp_opt.dat"))
        @test length(lines) == 1
        # Literal independent means; deviations sqrt(sum |delta|²/(n-1)).
        # Exact binary sums here; permit only a few sqrt rounding ulps.
        @test isapprox(parse.(Float64, split(only(lines))),
            [-1, 2, sqrt(10), 13, 4, 4, 9, 4, 4]; atol=0.0, rtol=4eps(Float64))
        child = readlines(joinpath(output, "zqp_orbital_opt.dat"))
        @test child[1:5] == ["======================", "NOrbitalIdx  1",
            "======================", "======================", "======================"]
        @test parse.(Float64, split(child[6])) == [0, 9, 4]
    end
    @testset "invalid windows fail before opening files" begin
        for invalid in (OptDataPoint[], [state.opt_data[1]],
            [state.opt_data[1], OptDataPoint(0im, 0im, ComplexF64[])])
            state.opt_data = invalid
            mktempdir() do output
                @test_throws ArgumentError MVMCOptimizers.output_opt_data!(data, state; output_dir=output)
                @test isempty(readdir(output))
            end
        end
    end
    @testset "sparse declared/shared/reserved DH RBM OptTrans ranges" begin
        sparse = ExpertModeData()
        sparse.n_gutzwiller_idx = 3
        sparse.n_jastrow_idx = 2
        sparse.gutzwiller_terms = [GutzwillerTerm(0, 1 + 0im, true)]
        sparse.jastrow_terms = [JastrowTerm(0, 1, 4 + 0im, true)]
        sparse.doublon_holon_2site_indices = [DoublonHolon2SiteIndex(zeros(Int, 0, 2))]
        sparse.doublon_holon_4site_indices = [DoublonHolon4SiteIndex(zeros(Int, 0, 4))]
        sparse.doublon_holon_2site_params = ComplexF64.(6:11)
        sparse.doublon_holon_4site_params = ComplexF64.(12:21)
        sparse.charge_rbm_phys_layer_terms = [ChargeRBMPhysLayerTerm(0, 24 + 0im, true, 2)]
        sparse.modpara.n_orbital_idx = 4
        sparse.orbital_terms = [OrbitalTerm(0, 1, 1, 26.0 + 0.0im, true),
            OrbitalTerm(1, 0, 1, 26.0 + 0.0im, true)]
        sparse.opt_trans = ComplexF64[29, 30]
        sparse.modpara.nsr_opt_itr_smp = 2
        # Caller-owned reserved values, not assumptions about C malloc contents.
        sparse.retained_parameters[:gutzwiller] = ComplexF64.(1:3)
        sparse.retained_parameters[:jastrow] = ComplexF64.(4:5)
        sparse.retained_parameters[:charge_rbm_phys_layer_terms] = ComplexF64.(22:24)
        sparse.retained_parameters[:slater] = ComplexF64.(25:28)
        snapshot = VMCOptimizationState(2, 1, 21, 30, 1, 1, false, false)
        # Explicit offsets: G3 J2 DH6 DH10 RBM3 Slater4 Trans2.
        expected = ComplexF64.(1:30)
        MVMCOptimizers.store_opt_data!(sparse, snapshot, 0)
        MVMCOptimizers.store_opt_data!(sparse, snapshot, 1)
        @test snapshot.opt_data[1].parameters == expected
        mktempdir() do output
            MVMCOptimizers.output_opt_data!(sparse, snapshot; output_dir=output)
            values = parse.(Float64, split(read(joinpath(output, "zqp_opt.dat"), String)))
            @test length(values) == 96 # 3*(NPara30+2).
            @test values[7:3:end] == real.(expected)
            for (family, header, width, first) in [
                ("gutzwiller", "NGutzwillerIdx  3", 3, 1),
                ("jastrow", "NJastrowIdx  2", 2, 4),
                ("doublonHolon2site", "NDoublonHolon2siteIdx  1", 6, 6),
                ("doublonHolon4site", "NDoublonHolon4siteIdx  1", 10, 12),
                ("chargeRBM_physlayer", "NChargeRBM_PhysLayerIdx  3", 3, 22),
                ("orbital", "NOrbitalIdx  4", 4, 25),
                ("trans", "NQPOptTrans  2", 2, 29)]
                child = readlines(joinpath(output, "zqp_" * family * "_opt.dat"))
                @test child[2] == header
                @test length(child) == width + 5
                @test [parse(Float64, split(line)[2]) for line in child[6:end]] ==
                    real.(expected[first:first+width-1])
            end
        end
    end
    @testset "OrbitalAntiParallel plus doubled OrbitalParallel" begin
        split = ExpertModeData()
        split.i_flg_orbital_general = 1
        split.i_flg_orbital_anti_parallel = 1
        split.i_flg_orbital_parallel = 1
        split.n_orbital_anti_parallel = 3
        split.modpara.n_orbital_idx = 7 # Anti3 + 2*Parallel2.
        split.modpara.nsr_opt_itr_smp = 2
        history = VMCOptimizationState(2, 1, 0, 7, 1, 1, true, false)
        history.opt_data = [OptDataPoint(0, 0, ComplexF64.(1:7)),
            OptDataPoint(0, 0, ComplexF64.(1:7))]
        mktempdir() do output
            MVMCOptimizers.output_opt_data!(split, history; output_dir=output)
            anti = readlines(joinpath(output, "zqp_orbitalAntiParallel_opt.dat"))
            parallel = readlines(joinpath(output, "zqp_orbitalParallel_opt.dat"))
            @test anti[2] == "NOrbitalAntiParallelIdx  3"
            @test parallel[2] == "NOrbitalParallelIdx  4"
            @test length(anti) == 8
            @test length(parallel) == 9
            @test [parse(Float64, Base.split(line)[2]) for line in parallel[6:end]] == [4, 5, 6, 7]
        end
        split.n_orbital_anti_parallel = 2 # Leaves odd Parallel width5: invalid doubled layout.
        mktempdir() do output
            @test_throws ArgumentError MVMCOptimizers.output_opt_data!(split, history; output_dir=output)
            @test isempty(readdir(output))
        end
    end
    @testset "six-site declared layout is NPara+2, not 38 expanded rows" begin
        six = ExpertModeData()
        six.n_gutzwiller_idx = 1
        six.n_jastrow_idx = 1
        six.modpara.n_orbital_idx = 12
        six.modpara.nsr_opt_itr_smp = 1
        six.orbital_terms = [OrbitalTerm(i, j, mod(6i + j, 12), 0.0im, true)
            for i in 0:5 for j in 0:5]
        single = VMCOptimizationState(6, 3, 2, 14, 1, 1, false, false)
        single.opt_data = [OptDataPoint(-1.0, 2.0, ComplexF64.(1:14))]
        mktempdir() do output
            MVMCOptimizers.output_opt_data!(six, single; output_dir=output)
            lines = readlines(joinpath(output, "zqp_opt.dat"))
            @test length(lines) == 1
            @test parse.(Float64, split(only(lines))) ==
                collect(Iterators.flatten((value, 0.0) for value in [-1, 2, collect(1:14)...]))
            @test length(split(only(lines))) == 32
        end
    end
end
