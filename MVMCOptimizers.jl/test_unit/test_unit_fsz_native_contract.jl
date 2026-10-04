using Test, MVMCOptimizers, MVMCExpertModeParsers, LinearAlgebra
using MVMCExpertModeParsers: ExpertModeData
using Random, SFMT

# Same scaled component policy as the reviewed native child comparison.
# Avoid complex abs overflow and a unit absolute floor for subnormal inverses.
function fsz_component_agrees(actual::Real, expected::Real)
    isfinite(actual) && isfinite(expected) || return false
    scale = max(abs(actual), abs(expected))
    scale == 0 && return actual == expected
    error = abs(actual / scale - expected / scale)
    limit = 64eps(Float64) + 4 * (nextfloat(0.0) / scale)
    return error <= limit
end
fsz_component_agrees(a::Complex, b::Complex) =
    fsz_component_agrees(real(a),real(b)) && fsz_component_agrees(imag(a),imag(b))
fsz_component_agrees(a::Real, b::Complex) = fsz_component_agrees(complex(a),b)

@testset "FSZ saved planes accept real and complex log IP" begin
    data = ExpertModeData()
    data.modpara.nsite = 2; data.modpara.nelec = 2
    # Literal zero-based electron labels/site occupancy, independent of a sampler.
    idx, cfg, num, proj, spn = [0, 1, 0, 1], [0, 1, 2, 3], [1, 1, 1, 1], [2], [0, 0, 1, 1]
    inputs = deepcopy((idx, cfg, num, proj, spn))
    for log_ip in (2.0, 2.0 + 3.0im)
        state = MVMCOptimizers.VMCOptimizationState(2, 2, 1, 2, 1, 1, log_ip isa Complex, true)
        MVMCOptimizers.save_ele_config_fsz!(0, log_ip, idx, cfg, num, proj, spn, data, state)
        ec = state.electron_config
        @test ec.ele_idx == [0, 1, 0, 1]
        @test ec.ele_cfg == [0, 1, 2, 3]
        @test ec.ele_num == [1, 1, 1, 1]
        @test ec.ele_proj_cnt == [2]
        @test ec.ele_spn == [0, 0, 1, 1]
        @test (idx, cfg, num, proj, spn) == inputs
    end
end

# Public real-FSZ control: Julia Expert input, not a mode-name override.
# vmc_phys_cal! exposes both callback and caller-owned RNG; the namelist wrapper
# exposes neither. Preparation below follows that wrapper's fixed-load order.
@testset "public real FSZ full-occupancy analytic control" begin
    mktempdir() do root
        header(name, count; parameter=false) =
            "=====\n$name $count\n" * (parameter ? "ComplexType 0\n" : "=====\n") * "=====\n=====\n"
        files = Dict(
            "namelist.def" => "ModPara modpara.def\nLocSpin locspn.def\nGutzwiller gutz.def\nOrbitalGeneral orbital.def\nTransSym qp.def\nCoulombIntra coulomb.def\nOneBodyG green.def\n",
            "modpara.def" => "--------------------\nModel_Parameters 0\n--------------------\nVMC_Cal_Parameters\n--------------------\nCDataFileHead zvo\nCParaFileHead zqp\nNVMCCalMode 1\nNLanczosMode 0\nNDataIdxStart 1\nNDataQtySmp 1\nNsite 2\nNcond 4\n2Sz 0\nNSPGaussLeg 1\nNSPStot 0\nNMPTrans 1\nNSROptItrStep 1\nNSROptItrSmp 1\nNVMCWarmUp 1\nNVMCInterval 1\nNVMCSample 1\nNExUpdatePath 0\nRndSeed 1\nNSplitSize 1\nNStore 1\nNSRCG 0\n",
            "locspn.def" => header("NlocalSpin", 0) * "0 0\n1 0\n",
            "gutz.def" => header("NGutzwillerIdx", 1; parameter=true) * "0 0\n1 0\n0 0\n",
            # Complete General upper triangle of the four spin-orbitals.
            "orbital.def" => header("NOrbitalIdx", 1; parameter=true) *
                join(("$i $j 0 1\n" for i in 0:3 for j in i+1:3)) * "0 0\n",
            "qp.def" => header("NQPTrans", 1) * "0 1\n0 0 0 1\n0 1 1 1\n",
            "coulomb.def" => header("NCoulombIntra", 2) * "0 2\n1 2\n",
            "green.def" => header("NCisAjs", 16) *
                join(("$i $s $j $t\n" for i in 0:1 for s in 0:1 for j in 0:1 for t in 0:1)),
            # Two metadata triples, then declared (real, imag, stddev) triples.
            "fixed.dat" => "0 0 0 0 0 0\n0 0 0\n1 0 0\n",
        )
        for (name, contents) in files
            write(joinpath(root, name), contents)
        end
        snapshots = []
        for repeat in 1:2
            data = MVMCExpertModeParsers.parse_expert_mode_files(joinpath(root, "namelist.def"))
            @test data.i_flg_orbital_general == 1
            @test all(iszero, data.complex_flags)
            @test length(data.orbital_terms) == 6 && all(!t.is_complex for t in data.orbital_terms)
            @test length(data.gutzwiller_terms) == 1 && all(!t.is_complex for t in data.gutzwiller_terms)
            @test data.gutzwiller_idx == [0, 0]
            @test data.n_qp_trans == 1 && data.qp_trans == [[0, 1]]
            @test data.qp_trans_sgn == [[1, 1]] && data.para_qp_trans == ComplexF64[1]
            @test !MVMCOptimizers.get_all_complex_flag(data)
            @test (data.modpara.nelec, data.modpara.nvmc_sample, data.modpara.n_data_qty_smp,
                data.modpara.nvmc_warmup, data.modpara.nvmc_interval, data.modpara.nsr_opt_itr_step) == (2, 1, 1, 1, 1, 1)
            @test MVMCOptimizers.read_opt_para_file!(data, joinpath(root, "fixed.dat")) == 2
            MVMCOptimizers.read_input_parameters!(data, joinpath(root, "namelist.def"))
            MVMCOptimizers.sync_modified_parameter!(data; shift_correlations=false)
            fixed = copy(MVMCOptimizers.pack_parameters(data))
            flags = copy(data.optimization_flags)
            @test fixed == ComplexF64[0, 4] # C D_AmpMax / max(abs(Slater)).
            @test length(flags) == 4 && all(iszero, flags[1:2:end])
            rng = SFMT19937RNG(); Random.seed!(rng, 1)
            callbacks = []
            callback = (sample, actual, energy, info) -> begin
                @test sample == 0 && info == 0
                @test actual === data
                @test !MVMCOptimizers.get_all_complex_flag(actual) && actual.i_flg_orbital_general == 1
                @test fsz_component_agrees(energy, 4 + 0im)
                push!(callbacks, (sample, energy, info))
            end
            output = joinpath(root, "run-$repeat")
            mkdir(output) # Two distinct, test-owned output directories.
            @test MVMCOptimizers.vmc_phys_cal!(data; rng, callback, output_dir=output) == 0
            @test length(callbacks) == 1
            @test MVMCOptimizers.pack_parameters(data) == fixed
            @test data.optimization_flags == flags
            out = parse.(Float64, split(read(joinpath(output, "zvo_out.dat"), String)))
            @test length(out) == 6
            # H=2*n0up*n0down+2*n1up*n1down acts as 4 on full occupancy.
            # Normalized physical observables have unit scale, unlike the tiny
            # inverse entries above. Same 64eps dimension-four operation budget;
            # E2's scale is 16 and variance includes subtraction of E squared.
            @test all(isapprox(a,b; atol=64eps(Float64)*max(abs(b),1), rtol=0)
                for (a,b) in zip(out, (4, 0, 16, 0, 0, 0)))
            rows = filter(!isempty, strip.(readlines(joinpath(output, "zvo_cisajs_001.dat"))))
            @test length(rows) == 16
            for (row, term) in zip(rows, data.green_one_terms)
                fields = split(row)
                @test length(fields) == 6
                indices = parse.(Int, fields[1:4])
                @test indices == [term.site1, term.spin1 == :up ? 0 : 1,
                    term.site2, term.spin2 == :up ? 0 : 1]
                expected = indices[1:2] == indices[3:4] ? 1 : 0
                @test isapprox(complex(parse(Float64, fields[5]), parse(Float64, fields[6])),
                    complex(expected); atol=64eps(Float64), rtol=0)
            end
            # Observable repeatability only: SFMT wrapper exposes no raw/cursor/count.
            push!(snapshots, [rand(rng, UInt32) for _ in 1:624])
        end
        @test snapshots[1] == snapshots[2]
    end
end

@testset "FSZ native child contracts" begin
    data = ExpertModeData()
    data.modpara.nsite = 2
    data.modpara.nelec = 2
    idx, spn = [0, 1, 0, 1], [0, 0, 1, 1]
    for real_mode in (false, true)
        state = MVMCOptimizers.VMCOptimizationState(2, 2, 0, 0, 3, 1, !real_mode, true)
        sm = state.slater_matrix
        fill!(sm.pf_m, 7 + 2im); fill!(sm.inv_m, 9 + 3im)
        fill!(sm.pf_m_real, 7); fill!(sm.inv_m_real, 9)
        before = deepcopy(sm)
        f = real_mode ? MVMCOptimizers.calculate_m_all_fsz_real! : MVMCOptimizers.calculate_m_all_fsz!
        # Native 4x4 zero matrix: SKTRF INFO=3, no public publication.
        @test f(idx, spn, 2, 3, data, state) == 3
        @test sm.pf_m == before.pf_m
        @test sm.inv_m == before.inv_m
        @test sm.pf_m_real == before.pf_m_real
        @test sm.inv_m_real == before.inv_m_real
        @test f(idx, spn, 2, 2, data, state) == 0
    end
    state = MVMCOptimizers.VMCOptimizationState(2, 2, 0, 0, 3, 1, false, true)
    sm = state.slater_matrix
    # Independent native regular input. Pf = .5*1.25 - .125*.625 + .25*.375.
    for ((i,j), v) in zip(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)), (.5,.125,.25,.375,.625,1.25))
        sm.slater_elm_real[16 + i*4+j+1] = v
        sm.slater_elm_real[16 + j*4+i+1] = -v
    end
    complex_before = deepcopy((sm.pf_m, sm.inv_m, sm.slater_elm))
    @test MVMCOptimizers.calculate_m_all_fsz_real!(idx, spn, 2, 3, data, state) == 0
    @test sm.pf_m_real[2] == .640625 # Binary-exact operands and result.
    @test (sm.pf_m, sm.inv_m, sm.slater_elm) == complex_before
end

@testset "native initializers reject successful attempt 101" begin
  for real_mode in (true, false)
    file = real_mode ? "real-initializer-101.txt" : "complex-initializer-101.txt"
    lines = readlines(joinpath(@__DIR__, "fixtures", "c_fsz_contract", file))
    @test length(lines) == 7
    attempt, status, aborted, cursor, draws = parse.(Int, split(lines[1]))
    @test (attempt, status, aborted, cursor, draws) == (101, 0, 1, 202, 202)
    data = ExpertModeData()
    data.modpara.nsite = 128; data.modpara.nelec = 1; data.modpara.two_sz = -1
    state = MVMCOptimizers.VMCOptimizationState(128, 1, 0, 0, 1, 1, !real_mode, true)
    sm = state.slater_matrix
    # Native SFMT seed1 first reaches this finite sparse pair at attempt101.
    table = real_mode ? sm.slater_elm_real : sm.slater_elm
    table[83*256+220+1] = 1
    table[220*256+83+1] = -1
    ec = state.electron_config
    rng = SFMT19937RNG(); Random.seed!(rng, 1)
    f = real_mode ? MVMCOptimizers.make_initial_sample_fsz_real! : MVMCOptimizers.make_initial_sample_fsz!
    message = real_mode ? "makeInitialSample_fsz_real: Too many loops" : "makeInitialSample_fsz: Too many loops"
    @test_logs (:error, message) begin
        @test f(ec.tmp_ele_idx,
            ec.tmp_ele_cfg, ec.tmp_ele_num, ec.tmp_ele_proj_cnt, ec.tmp_ele_spn,
            1, 2, data, state, rng) == 1
    end
    for (actual, line) in zip((ec.tmp_ele_idx, ec.tmp_ele_cfg, ec.tmp_ele_num, ec.tmp_ele_spn), lines[2:5])
        @test actual == parse.(Int, split(line))
    end
    @test (real_mode ? sm.pf_m_real : sm.pf_m) == [1.0] # Kernel publishes before exhaustion.
    # The published SFMT wrapper does not expose cursor/raw state/count.
    # Assert its observable next624, not an invented cursor accessor.
    @test [rand(rng, UInt32) for _ in 1:624] == parse.(UInt32, split(lines[7]))
  end
end

@testset "FSZ independent native rows and local status" begin
    data = ExpertModeData()
    data.modpara.nsite = 2; data.modpara.nelec = 2
    idx, spn = [0, 1, 0, 1], [0, 0, 1, 1]
    for file in ("rows.txt", "sum-overflow.txt"), line in readlines(joinpath(@__DIR__, "fixtures", "c_fsz_contract", file))
        words = split(line)
        @test length(words) == 41
        real_mode = words[1] == "real"
        name = words[2]
        first, local_qp, expected_status = parse.(Int, words[3:5])
        state = MVMCOptimizers.VMCOptimizationState(2, 2, 0, 0, 3, 1, !real_mode, true)
        sm = state.slater_matrix
        table = real_mode ? sm.slater_elm_real : sm.slater_elm
        for qp in 0:2, ((i,j), value) in zip(((0,1),(0,2),(0,3),(1,2),(1,3),(2,3)), (.5,.125,.25,.375,.625,1.25))
            v = real_mode ? value : ComplexF64(value)
            if qp == first + local_qp
                if name == "zero"
                    v = zero(v)
                elseif name in ("infinite", "offset_infinite") && (i,j) == (0,1)
                    v = real_mode ? Inf : ComplexF64(Inf, 0)
                elseif name == "nan" && (i,j) == (0,1)
                    v = real_mode ? NaN : ComplexF64(NaN, 0)
                elseif name == "complex_regular" && !real_mode && (i,j) == (0,1)
                    v = ComplexF64(.5,.25)
                elseif name == "sum_overflow" && (i,j) == (0,1)
                    v = real_mode ? 1e308 : ComplexF64(1e308,1e308)
                end
            end
            table[16*qp+i*4+j+1] = v
            table[16*qp+j*4+i+1] = -v
        end
        pf = real_mode ? sm.pf_m_real : sm.pf_m
        inv = real_mode ? sm.inv_m_real : sm.inv_m
        fill!(pf, 23); fill!(inv, 29)
        before = deepcopy((pf, inv))
        f = real_mode ? MVMCOptimizers.calculate_m_all_fsz_real! : MVMCOptimizers.calculate_m_all_fsz!
        @test f(idx, spn, first+1, first+local_qp+2, data, state) == expected_status
        if expected_status != 0
            @test (pf, inv) == before
        else
            expected = complex(parse(Float64,words[8]), parse(Float64,words[9]))
            qp = first + local_qp + 1
            # Well-scaled 4x4 regular cases: fixed dimension operation budget,
            # not a computed-bit requirement. The overflow real case is scaled.
            @test fsz_component_agrees(pf[qp],expected)
            target = [complex(parse(Float64,words[i]),parse(Float64,words[i+1])) for i in 10:2:40]
            actual = inv[16*(qp-1)+1:16*qp]
            @test length(actual) == length(target) == 16
            @test all(fsz_component_agrees(a,b) for (a,b) in zip(actual,target))
            # Four products/three additions per entry. Scaled backward residual
            # tests the original real/complex matrix, independently of C values.
            # Use BigFloat only in this test to avoid overflow of the huge case.
            setprecision(BigFloat, 256) do
                a = [Complex{BigFloat}(table[16*(qp-1)+(i-1)*4+j]) for i in 1:4, j in 1:4]
                inverse = reshape(Complex{BigFloat}.(actual),4,4)
                for i in 1:4, j in 1:4
                    terms = [a[i,k]*inverse[k,j] for k in 1:4]
                    # C negates the inverse after factorizing the assembled
                    # column-major matrix: Slater * storedInv = -I.
                    target_identity = i == j ? -1 : 0
                    scale = abs(target_identity) + sum(abs,terms)
                    @test abs(sum(terms)-target_identity) <= 64eps(Float64)*scale
                end
            end
            # No preceding/following range or canonical inverse padding writes.
            @test inv[49:end] == before[2][49:end]
            @test pf[1:first] == before[1][1:first]
            @test pf[qp+1:end] == before[1][qp+1:end]
        end
    end
end
