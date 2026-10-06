using Test
using MVMCExpertModeParsers

@testset "contract/InterAll scientific coefficients" begin
    # C GetInfoInterAll reads the last two fields with %lf: exponent letters
    # belong to coefficient tokens and do not make a data row a header.
    content = """
    ================================
    NInterAll 6
    ComplexType 1
    ================================
    site0 spin0 site1 spin1 site2 spin2 site3 spin3 real imag
    # scientific real and imaginary components
    0 0 1 0 2 1 3 1 -3.75e-1 +1.875E-1
    3 1 2 1 1 0 0 0 +2.5E+1 -4e-2 # inline comment
    // Signed zero is meaningful even though the coefficient is real.
    1 0 1 0 2 1 2 1 -0e+0 +0E-0
    2 0 3 0 0 1 1 1 1.25 -0.5
    0 0 1 0 2 1 3 1 -3.75e-1 +1.875E-1 // duplicate
    0 1 1 1 2 0 3 0 2e-1 -0e+10
    """
    indices = ((0, 0, 1, 0, 2, 1, 3, 1),
               (3, 1, 2, 1, 1, 0, 0, 0),
               (1, 0, 1, 0, 2, 1, 2, 1),
               (2, 0, 3, 0, 0, 1, 1, 1),
               (0, 0, 1, 0, 2, 1, 3, 1),
               (0, 1, 1, 1, 2, 0, 3, 0))
    values = ComplexF64[ComplexF64(-0.375, 0.1875), ComplexF64(25.0, -0.04),
                        ComplexF64(-0.0, 0.0), ComplexF64(1.25, -0.5),
                        ComplexF64(-0.375, 0.1875), ComplexF64(0.2, -0.0)]
    complex_flags = (true, true, false, true, true, false)

    function check_terms(result)
        @test result.success
        @test length(result.data) == length(indices)
        for (term, idx, value, complex_flag) in zip(result.data, indices, values, complex_flags)
            @test (term.site0, term.spin0, term.site1, term.spin1,
                   term.site2, term.spin2, term.site3, term.spin3) == idx
            @test reinterpret(UInt64, [term.value]) == reinterpret(UInt64, [value])
            @test term.is_complex == complex_flag
        end
    end

    check_terms(MVMCExpertModeParsers.parse_interall_content(content))
    mktemp() do path, io
        write(io, content)
        flush(io)
        check_terms(MVMCExpertModeParsers.parse_interall_def(path))
    end
end
