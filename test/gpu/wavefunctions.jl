using Test, Random, CUDA, MVMCOptimizers, MVMCExpertModeParsers
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
@testset "Resident projected wavefunctions match independent CPU calculation" begin
    for model in ("hubbard_chain_real", "heisenberg_chain_real")
        data = parse_expert_mode_files(joinpath(@__DIR__, "..", "..", "examples", "inputs", model, "namelist.def"))
        rng = CPU.SFMT19937RNG()
        Random.seed!(rng, 79)
        MVMCExpertModeParsers.init_parameter!(data; rng)
        data.modpara.nmp_trans = max(1, abs(data.modpara.nmp_trans))
        data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) ÷ 2
        MVMCExpertModeParsers.init_qp_weight!(data)
        CPU.sync_modified_parameter!(data)
        ns, ne = data.modpara.nsite, data.modpara.nelec
        np = CPU.count_total_parameters(data)
        nproj = MVMCExpertModeParsers.projection_layout(data).n_proj
        nq = CPU.get_n_qp_full(data)
        cpu = CPU.VMCOptimizationState(ns, ne, nproj, np, nq, 1, false, false)
        CPU.update_slater_elm_fcmp!(data, cpu)
        CPU.convert_to_real_arrays!(cpu)
        cpu.slater_matrix.slater_elm_real .= real.(cpu.slater_matrix.slater_elm)
        cfg = hcat([vcat(randperm(rng, ns)[1:ne] .- 1, randperm(rng, ns)[1:ne] .- 1) for _ in 1:131]...)
        gpu = gpu_state(data; configurations=cfg)
        expected = zeros(131)
        expected_logs = zeros(131)
        expected_counts = zeros(Int32, nproj, 131)
        for b in 1:131
            idx = cfg[:, b]
            counts = zeros(Int, nproj)
            occ = zeros(Int, 2ns)
            for i in 1:2ne
                occ[idx[i] + (i > ne ? ns : 0) + 1] = 1
            end
            CPU.make_proj_cnt!(counts, occ, data)
            expected_logs[b] = CPU.log_proj_val(counts, data)
            expected_counts[:, b] .= counts
            @test CPU.calculate_m_all_real!(idx, 1, nq+1, data, cpu) == 0
            expected[b] = CPU.calculate_ip_real(cpu.slater_matrix.pf_m_real, 1, nq+1, data)
        end
        @test Array(gpu.counts) == expected_counts
        @test Array(gpu.logcorrelations) ≈ expected_logs atol=1e-12 rtol=1e-12
        @test Array(gpu.overlaps) ≈ expected atol=1e-12 rtol=1e-11
    end
end
