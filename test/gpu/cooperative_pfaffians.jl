using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers
using PfaPack: utu2pfa
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers, :MVMCCUDAExt)

@testset "Cooperative Pfaffians preserve C-order factorization" begin
    rng = MersenneTwister(794)
    for n in (2, 6, 16, 64, 128)
        batch = 9
        matrices = randn(rng, n, n, batch)
        expected = zeros(Float64, batch)
        cpu_pivots = zeros(Int32, n, batch)
        cpu_factors = similar(matrices)
        for b in 1:batch
            matrices[:, :, b] -= transpose(matrices[:, :, b])
            f, p = copy(matrices[:, :, b]), zeros(Int32, n)
            @test MVMCOptimizers._ordinary_dsktf2_c_order!(f, p) == 0
            expected[b] = utu2pfa(n, f, n, p)
            cpu_pivots[:, b] .= p
            cpu_factors[:, :, b] .= f
        end
        factors = CuArray(matrices)
        pivots = CUDA.zeros(Int32, n, batch)
        values = CUDA.zeros(Float64, batch)
        GPU.cooperative_pfaffians!(values, factors, pivots)
        # The independent CPU normal factorization and product have the same
        # operation order; tolerances allow ordinary multiply/add roundoff.
        @test Array(values) ≈ expected atol=1e-10 rtol=1e-12
        @test Array(factors) ≈ cpu_factors atol=1e-10 rtol=1e-12
        @test Array(pivots) == cpu_pivots
        # Packed matrix buffers used by large walkers and walker-first buffers
        # used by the original sampler must give equivalent numeric results.
        walker_first = CuArray(permutedims(matrices, (3, 1, 2)))
        walker_pivots = CUDA.zeros(Int32, batch, n)
        GPU.cooperative_pfaffians!(values, GPU.factor_view(walker_first),
                                  GPU.pivot_view(walker_pivots))
        @test Array(values) ≈ expected atol=1e-10 rtol=1e-12
        @test Array(walker_pivots) == permutedims(cpu_pivots)
    end
    # A zero odd elimination column is legal. Closed-form nonsingular sparse
    # matrices test this without treating CPU full-normal factor info as an
    # oracle for those synthetic sparse inputs.
    for n in (6, 64, 128)
        matrices = zeros(Float64, n, n, 3)
        for i in 1:2:n-1
            matrices[i, i+1, 1] = 1.0
            matrices[i+1, i, 1] = -1.0
        end
        ne = n ÷ 2
        for i in 1:ne
            matrices[i, i+ne, 2] = 1.0
            matrices[i+ne, i, 2] = -1.0
        end
        values = CUDA.ones(Float64, 3)
        GPU.cooperative_pfaffians!(values, CuArray(matrices), CUDA.zeros(Int32, n, 3))
        @test Array(values) ≈ [1.0, (-1.0)^(ne*(ne-1)÷2), 0.0] atol=1e-14 rtol=1e-14
    end
end
