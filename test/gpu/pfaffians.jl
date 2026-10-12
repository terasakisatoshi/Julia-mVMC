using Test, Random, LinearAlgebra
using MVMCOptimizers, CUDA
using PfaPack: utu2pfa
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
@test CUDA.functional()

@testset "Device Pfaffians against CPU C-order factorization" begin
    rng = MersenneTwister(79)
    for T in (Float64, ComplexF64), n in (2, 4, 8, 16, 32), batch in (1, 131)
        a = randn(rng, T, n, n, batch)
        for b in 1:batch
            a[:, :, b] -= transpose(a[:, :, b])
        end
        expected = zeros(T, batch)
        for b in 1:batch
            f = copy(a[:, :, b])
            p = zeros(Int32, n)
            factor! = T === Float64 ? MVMCOptimizers._ordinary_dsktf2_c_order! :
                                     MVMCOptimizers._ordinary_zsktf2_c_order!
            @test factor!(f, p) == 0
            expected[b] = utu2pfa(n, f, n, p)
        end
        d = CuArray(a)
        values = CUDA.zeros(T, batch)
        pivots = CUDA.zeros(Int32, n, batch)
        gpu_pfaffians!(values, d, pivots)
        @test Array(values) ≈ expected atol=1e-10 rtol=1e-12
    end
    for n in (2, 8)
        values = CUDA.ones(Float64, 3)
        gpu_pfaffians!(values, CUDA.zeros(Float64, n, n, 3), CUDA.zeros(Int32, n, 3))
        @test Array(values) == zeros(3)
    end
    # Closed-form sparse nonsingular matrices have zero odd elimination
    # columns; they must not be mistaken for Pfaffian nodes.
    for n in (2,4,6,8), T in (Float64,ComplexF64)
        a = zeros(T,n,n,2)
        for i in 1:2:n-1
            a[i,i+1,1] = 1
            a[i+1,i,1] = -1
        end
        ne = n÷2
        for i in 1:ne
            a[i,i+ne,2] = 1
            a[i+ne,i,2] = -1
        end
        values = CUDA.zeros(T,2)
        gpu_pfaffians!(values,CuArray(a),CUDA.zeros(Int32,n,2))
        @test Array(values) ≈ T[1,(-1)^(ne*(ne-1)÷2)] atol=1e-14 rtol=1e-14
    end
end
