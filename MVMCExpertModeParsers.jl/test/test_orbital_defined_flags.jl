using Test, MVMCExpertModeParsers

@testset "orbital flags follow the independent C AP/P complex sum" begin
    # Literal defined-slot expectations from readdef.c GetInfoOpt and
    # GetInfoOptOrbitalParalell. False in unwritten AP slots is a safe extension.
    for (ap_complex, p_complex) in ((false, false), (true, false),
                                    (false, true), (true, true))
        d = MVMCExpertModeParsers.ExpertModeData()
        d.modpara.complex_flag = 1 # must not determine orbital masks
        d.i_flg_orbital_anti_parallel = 1
        d.i_flg_orbital_parallel = 1
        d.n_orbital_anti_parallel = 2
        d.modpara.n_orbital_idx = 4
        d.orbital_terms = [
            MVMCExpertModeParsers.OrbitalTerm(0, 1, 0, 0.0im, ap_complex),
            MVMCExpertModeParsers.OrbitalTerm(1, 0, 1, 0.0im, ap_complex),
            MVMCExpertModeParsers.OrbitalTerm(0, 1, 2, 0.0im, p_complex),
            MVMCExpertModeParsers.OrbitalTerm(0, 1, 3, 0.0im, p_complex),
        ]
        d.optimization_flags = trues(8) # expose stale imaginary flags
        MVMCExpertModeParsers.set_orbital_opt_flags!(d, Dict(0=>1, 1=>0, 2=>0, 3=>0);
            orbital_complex = ap_complex || p_complex)
        c = ap_complex || p_complex
        @test d.optimization_flags == Bool[1,c,0,0,0,c,0,c]
    end
end

@testset "actual AP/P file declarations determine masks" begin
    for (ap, p) in ((0,0), (1,0), (0,1), (1,1), (2,0), (0,2))
        mktempdir() do dir
            for (name, declaration, count, body) in (
                ("ap.def", ap, 2, "0 1 0\n1 0 1\n0 1\n1 0\n"),
                ("p.def", p, 1, "0 1 0\n0 0\n"))
                write(joinpath(dir,name), "=====\nNOrbitalIdx $count\nComplexType $declaration\n=====\n=====\n$body")
            end
            d = MVMCExpertModeParsers.ExpertModeData()
            d.modpara.complex_flag = 1
            d.i_flg_orbital_parallel = 1
            d.n_orbital_anti_parallel = 2
            d.modpara.n_orbital_idx = 4
            d.optimization_flags = trues(8)
            # Deliberately empty term storage: masks must use actual headers.
            MVMCExpertModeParsers.apply_orbital_opt_flags_from_files!(d,
                [("OrbitalAntiParallel","ap.def"),("OrbitalParallel","p.def")], dir)
            c = ap + p > 0
            @test d.optimization_flags == Bool[1,c,0,0,0,c,0,c]
        end
    end
end
