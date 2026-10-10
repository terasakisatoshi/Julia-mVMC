using Test, MVMCOptimizers

withenv("JULIA_MVMC_INNER_THREADS" => "1") do
@testset "Slater derivative disjoint QP planes and repeated scratch reuse" begin
    for size in (16, 32, 64)
        nelec = size ÷ 2
        nqp, nsp, norb = 8, 4, 19
        indices = [mod(i * 7 + i ÷ size, norb) for i in 1:(2 * size^2)]
        signs = [isodd(i) ? -1 : 1 for i in eachindex(indices)]
        inv = [ComplexF64((i % 29 - 14) / 32, (i % 13 - 6) / 64) for i in 1:(nqp * size^2)]
        pf = [ComplexF64(q / 16, (q - 5) / 32) for q in 1:nqp]
        cs = [ComplexF64(s / 16, (s - 2) / 32) for s in 1:nsp]
        cc = [ComplexF64((s + 2) / 8, (s - 1) / 16) for s in 1:nsp]
        ss = [ComplexF64((s - 3) / 8, (s + 1) / 16) for s in 1:nsp]
        expected = zeros(ComplexF64, nqp * norb)
        # Independent direct spin-block contraction. Column/row traversal is C's
        # ascending scatter order; repeated orbital indices exercise accumulation.
        for q in 1:nqp
            spin = mod(q - 1, nsp) + 1
            trans = (q - 1) ÷ nsp
            for row in 1:size, column in 1:size
                coefficient = if row <= nelec
                    column <= nelec ? pf[q] * cs[spin] : -pf[q] * cc[spin]
                else
                    column <= nelec ? pf[q] * ss[spin] : -pf[q] * cs[spin]
                end
                mapping = trans * size^2 + (row - 1) * size + column
                orbital = indices[mapping] + 1
                expected[(q - 1) * norb + orbital] +=
                    (inv[(q - 1) * size^2 + (row - 1) * size + column] * coefficient) * signs[mapping]
            end
        end
        arguments = (indices, signs, inv, pf, cs, cc, ss,
                     nelec, size, size^2, nqp, nsp, norb)
        actual = fill(ComplexF64(NaN, NaN), nqp * norb)
        for repeat in 1:12
            fill!(actual, 0)
            MVMCOptimizers._accumulate_slater_buffer_fcmp_fast!(actual, arguments...)
            # Inputs are dyadic; error budget covers only complex arithmetic
            # grouping in the independent signed-coefficient reference.
            @test isapprox(actual, expected; atol=1e-13, rtol=1e-13)
        end
        worker = fetch(Threads.@spawn begin
            buffer = zeros(ComplexF64, nqp * norb)
            MVMCOptimizers._accumulate_slater_buffer_fcmp_fast!(buffer, arguments...)
            buffer
        end)
        @test isapprox(worker, expected; atol=1e-13, rtol=1e-13)
        nested = [zeros(ComplexF64, nqp * norb) for _ in 1:Threads.nthreads()]
        Threads.@threads :static for slot in eachindex(nested)
            MVMCOptimizers._accumulate_slater_buffer_fcmp_fast!(nested[slot], arguments...)
        end
        @test all(buffer -> isapprox(buffer, expected; atol=1e-13, rtol=1e-13), nested)
        @test all(isfinite, actual)
    end
end
end
