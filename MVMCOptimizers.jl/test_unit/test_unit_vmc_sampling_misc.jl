using Test
using MVMCOptimizers
using MVMCExpertModeParsers
using MVMCExpertModeParsers:
    ExpertModeData,
    ModParaParameters,
    GutzwillerTerm,
    JastrowTerm,
    DoublonHolon2SiteIndex,
    DoublonHolon4SiteIndex

if !isdefined(@__MODULE__, :make_ele_num)
    include(joinpath(@__DIR__, "helpers", "mock_state.jl"))
end

@testset "unit/vmc_sampling: live allocation-free projection coefficients" begin
    data = ExpertModeData()
    data.n_gutzwiller_idx = 3
    data.n_jastrow_idx = 2
    data.gutzwiller_terms = [GutzwillerTerm(0, 0.5 + 7im, true), GutzwillerTerm(1, -0.25 + 8im, true)]
    data.jastrow_terms = [JastrowTerm(0, 1, 1.75 + 9im, true)]
    data.doublon_holon_2site_indices = [DoublonHolon2SiteIndex(zeros(Int,4,2))]
    data.doublon_holon_4site_indices = [DoublonHolon4SiteIndex(zeros(Int,4,4))]
    data.doublon_holon_2site_params = ComplexF64[2+10im,3+11im,4+12im]
    data.doublon_holon_4site_params = ComplexF64[5+13im,6+14im]
    expected = [0.5,-0.25,0,1.75,0,2,3,4,0,0,0,5,6,0,0,0,0,0,0,0,0]
    # Independent explicit flattened layout, including missing header slots.
    for count in 0:24
        old = collect(1:count)
        new = reverse(old)
        z_old = 0.0; z_ratio = 0.0
        for i in 1:min(count,length(expected))
            z_old += expected[i]*old[i]
            z_ratio += expected[i]*(new[i]-old[i])
        end
        @test MVMCOptimizers.log_proj_val(old,data) ≈ z_old atol=1e-14 rtol=1e-14
        @test MVMCOptimizers.log_proj_ratio(new,old,data) ≈ z_ratio atol=1e-14 rtol=1e-14
    end
    old = collect(1:21); new = reverse(old)
    MVMCOptimizers.log_proj_val(old,data)
    MVMCOptimizers.log_proj_ratio(new,old,data)
    @test (@allocated MVMCOptimizers.log_proj_val(old,data)) == 0
    @test (@allocated MVMCOptimizers.log_proj_ratio(new,old,data)) == 0
    data.gutzwiller_terms[1] = GutzwillerTerm(0,1.5+7im,true)
    @test MVMCOptimizers.log_proj_val([1],data) ≈ 1.5
    @test MVMCOptimizers.log_proj_ratio([2],[1],data) ≈ 1.5
end

@testset "unit/vmc_sampling: accepted-hop scratch reuse" begin
    data = ExpertModeData()
    data.modpara = ModParaParameters(nsite=3, nelec=1)
    state = MVMCOptimizers.VMCOptimizationState(3,1,0,0,1,1,false,false)
    state.slater_matrix.slater_elm_real[2*6+4+1] = 4.0
    vec1, vec2 = state.workspace.hop_vec1_real, state.workspace.hop_vec2_real
    for poison in (NaN,17.0)
        state.slater_matrix.inv_m_real .= [0.0,-0.5,0.5,0.0,91.0]
        state.slater_matrix.pf_m_real[1] = 2.0
        fill!(vec1,poison); fill!(vec2,poison)
        MVMCOptimizers.update_m_all_real!(0,0,[2,1],1,2,data,state)
        @test state.slater_matrix.pf_m_real[1] ≈ 4.0
        @test state.slater_matrix.inv_m_real ≈ [0.0,-0.25,0.25,0.0,91.0]
        @test state.workspace.hop_vec1_real === vec1
        @test state.workspace.hop_vec2_real === vec2
    end
end

@testset "unit/vmc_sampling: log_proj_val / log_proj_ratio" begin
    data = ExpertModeData()
    data.modpara = ModParaParameters(nsite = 2)
    data.gutzwiller_terms = [
        GutzwillerTerm(0, 0.50 + 1.0im, true),  # imag should be ignored
    ]
    data.jastrow_terms = [
        JastrowTerm(0, 1, -0.25 + 2.0im, true),
        JastrowTerm(0, 2, 1.75 - 3.0im, true),
    ]

    proj_cnt_old = [2, 0, -1]
    proj_cnt_new = [3, 2, -4]

    v_old = MVMCOptimizers.log_proj_val(proj_cnt_old, data)
    v_new = MVMCOptimizers.log_proj_val(proj_cnt_new, data)
    ratio = MVMCOptimizers.log_proj_ratio(proj_cnt_new, proj_cnt_old, data)

    @test ratio ≈ (v_new - v_old) atol = 1e-14

    # Explicit manual check (real parts only).
    params = [0.50, -0.25, 1.75]
    manual_old = sum(params .* proj_cnt_old)
    manual_new = sum(params .* proj_cnt_new)
    @test v_old ≈ manual_old atol = 1e-14
    @test v_new ≈ manual_new atol = 1e-14
end

@testset "unit/vmc_sampling: update_ele_config! / revert_ele_config! round-trip" begin
    n_site = 4
    n_elec = 2

    # Two electrons total per spin-sector arrays: one up (mi=0), one down (mi=1)
    # Layout:
    # - ele_idx[mi + s*Ne] = site index (0-based)
    # - ele_cfg[ri + s*Nsite] = electron index mi or -1
    # - ele_num[ri + s*Nsite] = 1 if occupied else 0
    ele_idx = [0, 2, 1, 3]  # up: [0,2], down: [1,3]
    ele_cfg = fill(-1, 2 * n_site)
    ele_cfg[0 + 0 * n_site + 1] = 0
    ele_cfg[2 + 0 * n_site + 1] = 1
    ele_cfg[1 + 1 * n_site + 1] = 0
    ele_cfg[3 + 1 * n_site + 1] = 1

    ele_num = make_ele_num(n_site; up_sites = [0, 2], down_sites = [1, 3])

    mi, ri, rj, s = 0, 0, 1, 0
    @test ele_cfg[ri + s * n_site + 1] == mi
    @test ele_cfg[rj + s * n_site + 1] == -1
    @test ele_num[ri + s * n_site + 1] == 1
    @test ele_num[rj + s * n_site + 1] == 0

    ele_idx0 = copy(ele_idx)
    ele_cfg0 = copy(ele_cfg)
    ele_num0 = copy(ele_num)

    MVMCOptimizers.update_ele_config!(mi, ri, rj, s, ele_idx, ele_cfg, ele_num, n_site, n_elec)
    @test ele_idx[mi + s * n_elec + 1] == rj
    @test ele_cfg[ri + s * n_site + 1] == -1
    @test ele_cfg[rj + s * n_site + 1] == mi
    @test ele_num[ri + s * n_site + 1] == 0
    @test ele_num[rj + s * n_site + 1] == 1

    MVMCOptimizers.revert_ele_config!(mi, ri, rj, s, ele_idx, ele_cfg, ele_num, n_site, n_elec)
    @test ele_idx == ele_idx0
    @test ele_cfg == ele_cfg0
    @test ele_num == ele_num0
end

