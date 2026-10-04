using Test, LinearAlgebra
using MVMCOptimizers

@testset "C CG solves defined small SPD without an absolute denominator cutoff" begin
    fields = split(read(joinpath(@__DIR__, "fixtures", "c_cg_small_spd.txt"), String))
    iterations = parse(Int, fields[1])
    solution, residual, direction = parse.(Float64, fields[2:4])
    # Binary-exact one-dimensional SPD: O=2^-54, S=2^-108, g=2^-54.
    # Condition number=1, delta=2^-108, d*S*d=2^-216 (nonzero, below 1e-30).
    # Exact analytical x=2^54. C's unchanged Main agrees; no sampling involved.
    for complex in (false, true)
        ws = MVMCOptimizers.CGWorkspace(1, 1, complex)
        ws.stcOs_real[1,1] = ldexp(1.0, -54)
        ws.g[1] = ldexp(1.0, -54)
        result = MVMCOptimizers.stochastic_opt_cg_main!(ws, 1, 1, 1.0, 0.0, 1e-30, 1, complex)
        @test result == iterations
        # One scalar update; one ulp at 2^54 allows portable computed arithmetic, not
        # the old zero solution. The observed C-Julia difference is recorded.
        @test ws.x[1] ≈ solution atol=4.0 rtol=0
        @test abs(ws.r[1] - residual) <= eps(Float64) * ldexp(1.0, -54)
        @test abs(ws.d[1] - direction) <= eps(Float64) * ldexp(1.0, -54)
        @info "C small-SPD CG first difference" complex solution_difference=ws.x[1]-solution residual_difference=ws.r[1]-residual
    end
end

@testset "verbatim C sampled CG across limits 1 through 41" begin
    decode(line) = [reinterpret(Float64, parse(UInt64, word; base=16)) for word in split(line)]
    for name in ("real", "complex", "sampled_complex")
        lines = filter(line -> !startswith(line, "#"), readlines(joinpath(@__DIR__, "fixtures", "c_cg_refresh", name * ".txt")))
        @test length(lines) == 7 + 3*41
        n, samples, complex_int = parse.(Int, split(lines[1]))
        @test n > 0
        @test samples > 0
        @test complex_int in (0, 1)
        complex = complex_int != 0
        mean, diagonal, real_samples, imag_samples, gradient, expected_product = decode.(lines[2:7])
        @test length(mean) == n
        @test length(diagonal) == n
        @test length(gradient) == n
        @test length(expected_product) == n
        @test length(real_samples) == n*samples
        @test length(imag_samples) == (complex ? n*samples : 0)
        function workspace()
            ws = MVMCOptimizers.CGWorkspace(n, samples, complex)
            ws.stcO .= mean; ws.sdiag .= diagonal; ws.g .= gradient
            ws.stcOs_real .= reshape(real_samples, n, samples)
            complex && (ws.stcOs_imag .= reshape(imag_samples, n, samples))
            return ws
        end
        ws = workspace()
        product = zeros(n)
        MVMCOptimizers.operate_by_s!(product, gradient, ws, n, samples, 1.0/samples, 1e-5, complex)
        operator_budget = 8 * (n + samples) * eps(Float64)
        @test all(abs(a-b) <= operator_budget + operator_budget*abs(b) for (a,b) in zip(product, expected_product))
        # Independent explicit covariance; does not invoke the sampled operator.
        covariance = zeros(n,n)
        for row in 1:n, col in 1:n
            gram = 0.0
            for sample in 1:samples
                r = (sample-1)*n + row; c = (sample-1)*n + col
                gram += real_samples[r]*real_samples[c] + (complex ? imag_samples[r]*imag_samples[c] : 0.0)
            end
            covariance[row,col] = gram/samples - mean[row]*mean[col] + (row==col ? 1e-5*diagonal[row] : 0.0)
        end
        first_difference = nothing
        for limit in 1:41
            base = 8 + 3*(limit-1)
            words = split(lines[base]; limit=3)
            @test parse(Int,words[1]) == limit
            expected_iter = parse(Int,words[2])
            @test 0 <= expected_iter <= limit
            expected = (decode(words[3]), decode(lines[base+1]), decode(lines[base+2]))
            @test all(length(vector) == n for vector in expected)
            ws = workspace()
            iterations = MVMCOptimizers.stochastic_opt_cg_main!(ws,n,samples,1.0/samples,1e-5,0.0,limit,complex)
            @test iterations == expected_iter
            budget = 8*(n+samples)*limit*eps(Float64) # Unchanged Rust C-refresh component policy.
            for (stage,actual,target) in zip(("solution","residual","direction"),(ws.x,ws.r,ws.d),expected)
                @test length(actual) == length(target)
                for i in 1:n
                    if first_difference === nothing && actual[i] != target[i]
                        first_difference = (limit, stage, i, actual[i]-target[i])
                    end
                    @test abs(actual[i]-target[i]) <= budget + budget*abs(target[i])
                end
            end
            for row in 1:n
                ax = 0.0; scale = abs(gradient[row])
                for col in 1:n
                    term = covariance[row,col]*ws.x[col]
                    ax += term; scale += abs(term)
                end
                residual_budget = 16*(n+samples)*limit*eps(Float64)*scale
                @test abs(ws.r[row]-(gradient[row]-ax)) <= residual_budget
            end
        end
        @test length(lines) == 7 + 3*41
        @info "C refresh first computed difference (not a tolerance failure)" name first_difference
    end
end
