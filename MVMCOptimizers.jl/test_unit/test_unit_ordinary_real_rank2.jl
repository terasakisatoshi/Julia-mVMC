using Test, MVMCOptimizers, PfaPack

@testset "Ordinary real C DSKR2 cancellation boundary" begin
    # Independent native fixture; inverse values are diagnostic, not a gate.
    line = first(readlines(joinpath(@__DIR__, "fixtures/c_ordinary_rank2/native.txt")))
    @test occursin("status=0 factor=0 inverse_calls=1", line)
    expected_pf = parse(Float64, split(line, "pf=")[2])
    A = zeros(4, 4)
    entries = [(1,2,2.0^54), (1,3,1.0), (1,4,-2.0^54),
               (2,3,2.0^55), (2,4,1.0), (3,4,2.0^55)]
    for (i,j,v) in entries
        A[i,j] = v
        A[j,i] = -v
    end
    original = copy(A)
    ipiv = zeros(Int, 4)
    # The former INFO1 negative control exposed the dependency's grouped update.
    # Its C-left association repair must now retain the native successful pivot.
    @test PfaPack.julia_dsktf2!(copy(A), copy(ipiv)) == 0
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(A, ipiv) == 0
    pf = PfaPack.utu2pfa(4, A, 4, ipiv)
    # Two dyadic factor products plus pivot sign: reviewed native 4EPS policy.
    @test abs(pf - expected_pf) <= 4eps(Float64) * abs(expected_pf)
    @test all(p -> 1 <= p <= 4, ipiv)

    # Actual ordinary child, including its existing reciprocal inverse route.
    # The native FSZ assembly and this full-occupancy ordinary assembly use the
    # same four spin-sites (0,1,2,3). No inverse forward comparison is asserted.
    slater = vec(permutedims(original))
    inverse = zeros(4,4)
    child_pf = Ref(0.0)
    @test MVMCOptimizers.calculate_m_all_child_real!(
        [0,1,0,1], slater, inverse, child_pf, zeros(4,4), zeros(Int,4),
        zeros(4), zeros(4,4), 2, 4) == 0
    @test abs(child_pf[] - expected_pf) <= 4eps(Float64) * abs(expected_pf)

    # View workspace, tied pivot (first wins), zero column, empty dimension.
    view_parent = zeros(6,6)
    view_parent[2:5,2:5] .= original
    view_piv = zeros(Int,4)
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(view(view_parent,2:5,2:5), view_piv) == 0
    @test view_piv == ipiv
    tied = [0.0 0 2; 0 0 -2; -2 2 0]
    tied_piv = zeros(Int,3)
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(tied, tied_piv) == 1
    @test tied_piv[2] == 1
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(zeros(4,4), zeros(Int,4)) == 3
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(zeros(0,0), Int[]) == 0
end
