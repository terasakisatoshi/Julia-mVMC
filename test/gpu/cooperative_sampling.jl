using Test, Random, CUDA, MVMCOptimizers
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const GPU = Base.get_extension(MVMCOptimizers, :MVMCCUDAExt)

# Synthetic dense projected states isolate the block-wide sampler from Expert
# input preparation. Both paths receive identical physical matrices and tapes.
function sampling_state(ns, cfg, slater, localspins)
    ne, batch = size(cfg, 1) ÷ 2, size(cfg, 2)
    nq = size(slater, 3)
    s = GPU.CUDAState(ns, ne, 1, nq, 32, CuArray([0.1]),
        CUDA.zeros(Float64, 0, 1), CuArray(slater), CuArray([1.0, 0.4]),
        CuArray(cfg), CUDA.zeros(Int32, 2ns, batch), CUDA.zeros(Int32, 1, batch),
        CuArray(zeros(Int32, ns)), CuArray(fill(Int32(-1), ns, ns)), 1, 0,
        CUDA.zeros(Float64, batch, 2ne, 2ne), CUDA.zeros(Int32, batch, 2ne),
        CUDA.zeros(Float64, batch), CUDA.zeros(Float64, batch),
        CUDA.zeros(Float64, batch), CuArray(localspins), CUDA.zeros(Int32, batch),
        CUDA.zeros(Float64, 2ne, 2ne, batch),nothing)
    gpu_wavefunctions!(s)
    return s
end

function scalar_sample!(s, tape, exchange_fraction)
    batch = size(s.configurations, 2)
    @cuda threads=s.threads blocks=cld(batch, s.threads) fastmath=false GPU.sampling_kernel!(
        s.configurations, s.counts, s.occupancy, s.overlaps, s.logcorrelations,
        s.accepted, s.localspins, s.gutz, s.jastrow, s.parameters, s.slater,
        s.weights, GPU.factor_view(s.factors), GPU.pivot_view(s.pivots),
        s.pfaffians, tape, Float64(exchange_fraction), s.nsite, s.nelec,
        s.ngutz, s.njastrow, s.nqp, batch)
    return s
end

@testset "Cooperative sampler preserves fixed proposal and acceptance paths" begin
    for ns in (64, 128)
        ne, batch = ns ÷ 2, 4
        rng = MersenneTwister(731 + ns)
        cfg = repeat(reshape(Int32.(0:ns-1), ns, 1), 1, batch)
        # An initial doubly occupied site also exercises invalid exchanges
        # between electrons at the same position without duplicate spin sites.
        cfg[ne+1, :] .= 0
        slater = randn(rng, 2ns, 2ns, 2)
        for q in 1:2
            slater[:, :, q] -= transpose(slater[:, :, q])
        end
        for exchange_fraction in (0.0, 1.0, 0.37), pinned in (false, true)
            localspins = zeros(Int32, ns)
            pinned && (localspins[[1, ne + 2]] .= 1)
            scalar = sampling_state(ns, cfg, slater, localspins)
            cooperative = sampling_state(ns, cfg, slater, localspins)
            # Include occupied-target hopping self-loops,
            # pinned local-spin hops, and a vacant-target hopping proposal.
            tape = rand(rng, UInt32, 5, batch, 20)
            tape[:, :, 1] .= UInt32.([0, 0, 0, 0, 0])
            tape[:, :, 2] .= UInt32.([1, 0, ne + 1, 0, 0])
            tape[:, :, 3] .= UInt32.([1, 0, ne + 2, 0, 0])
            tape[:, :, 4] .= UInt32.([0, 0, ne + 3, typemax(UInt32), typemax(UInt32)])
            for step in 1:20
                packet = CuArray(tape[:, :, step:step])
                scalar_sample!(scalar, packet, exchange_fraction)
                gpu_sample!(cooperative; steps=1, exchange_fraction, tape=packet)
                # RNG words, integer state and discrete decisions are exact.
                @test Array(cooperative.configurations) == Array(scalar.configurations)
                @test Array(cooperative.accepted) == Array(scalar.accepted)
                @test Array(cooperative.counts) == Array(scalar.counts)
                @test Array(cooperative.occupancy) == Array(scalar.occupancy)
                # Shared update arithmetic may differ by normal GPU roundoff.
                @test Array(cooperative.overlaps) ≈ Array(scalar.overlaps) atol=1e-11 rtol=1e-11
                @test Array(cooperative.logcorrelations) ≈ Array(scalar.logcorrelations) atol=1e-13 rtol=1e-13
                if step == 1
                    @test Array(cooperative.configurations) == cfg
                    @test all(iszero, Array(cooperative.accepted))
                end
                if exchange_fraction == 0 && pinned && step == 2
                    @test Array(cooperative.configurations) == cfg
                    @test all(iszero, Array(cooperative.accepted))
                end
            end
        end
        # Production cuRAND tape generation consumes the same words in both
        # paths. Future words detect additional/conditional RNG consumption.
        localspins = zeros(Int32, ns)
        scalar = sampling_state(ns, cfg, slater, localspins)
        cooperative = sampling_state(ns, cfg, slater, localspins)
        CUDA.seed!(731)
        packet = CUDA.rand(UInt32, 5, batch, 20)
        scalar_sample!(scalar, packet, 0.37)
        future = Array(CUDA.rand(UInt32, 64))
        CUDA.seed!(731)
        gpu_sample!(cooperative; steps=20, exchange_fraction=0.37)
        @test Array(CUDA.rand(UInt32, 64)) == future
        @test Array(cooperative.configurations) == Array(scalar.configurations)
        @test Array(cooperative.accepted) == Array(scalar.accepted)
        @test Array(cooperative.overlaps) ≈ Array(scalar.overlaps) atol=1e-11 rtol=1e-11
    end
end
