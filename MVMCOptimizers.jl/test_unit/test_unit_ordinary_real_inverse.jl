using Test, MVMCOptimizers, LinearAlgebra

function ordinary_real_fixture_pairs(path, count)
    words = split(read(path,String))
    length(words) == 2count || error("complete real pair fixture required")
    values = parse.(Float64, words)
    all(iszero, values[2:2:end]) || error("real fixture imaginary fields must be zero")
    return values[1:2:end]
end

@testset "Ordinary real private C direct inverse" begin
    # Independent solve/residual checks exercise small even dimensions,
    # strided views and repeated scratch use without a Rust-generated oracle.
    for n in (2,4,6,8,16)
        vt = fill(2.0,n-1)
        T = diagm(-1 => vt, 1 => -vt)
        B = [Float64(i-j)/8 for i in 1:n, j in 1:n]
        parent = fill(-91.0,n+2,n+2)
        C = view(parent,2:n+1,2:n+1)
        reference = T \ B
        for reuse in 1:2
            MVMCOptimizers._ordinary_sktdsmx_real_c_order!(n,vt,B,C)
            @test all(abs(a-e) <= 256eps(Float64)*(1+abs(e)) for (a,e) in zip(C,reference))
            @test opnorm(T*C-B,Inf) <= 256eps(Float64)*(opnorm(T,Inf)*opnorm(C,Inf)+opnorm(B,Inf))
            @test all(==(-91.0),parent[[1,n+2],:]) && all(==(-91.0),parent[:,[1,n+2]])
        end
    end
    alias = ones(2,2)
    MVMCOptimizers._ordinary_sktdsmx_real_c_order!(2,[2.0],alias,alias)
    @test alias ≈ [-0.25 -0.25; -0.5 -0.5] atol=1e-14 rtol=1e-14

    # Analytic tridiagonal system: all four C branches, exact dyadic operands.
    vt = [2.0,1.0,4.0]
    B = Matrix{Float64}(I,4,4)
    C = fill(17.0,4,4)
    MVMCOptimizers._ordinary_sktdsmx_real_c_order!(4,vt,B,C)
    expected = [0.0 .5 0 .125; -.5 0 0 0; 0 0 0 .25; -.125 0 -.25 0]
    @test C == expected
    T = [0.0 -2 0 0; 2 0 -1 0; 0 1 0 -4; 0 0 4 0]
    @test T*C == B

    # Power-of-two/IEEE classifications agree with reviewed native probe.
    # Subnormal/Inf cases are internal diagnostics, not finite public models.
    for (denominator,numerator,quotient) in (
        (2.0^1023,1.0,2.0^-1023), (nextfloat(0.0),0.0,0.0),
        (nextfloat(0.0),nextfloat(0.0),1.0), (Inf,1.0,0.0), (0.0,1.0,Inf))
        C = fill(17.0,2,2)
        MVMCOptimizers._ordinary_sktdsmx_real_c_order!(2,[denominator],fill(numerator,2,2),C)
        @test C[1,:] == fill(quotient,2)
        @test C[2,:] == fill(-quotient,2)
    end
    C = zeros(2,2)
    MVMCOptimizers._ordinary_sktdsmx_real_c_order!(2,[Inf],fill(Inf,2,2),C)
    @test all(isnan,C)

    for divisor in (2.0^1023,2.0^-1023)
        A = [0.0 divisor; -divisor 0.0]
        M = fill(17.0,2,2); vt = fill(17.0,2); piv = [1,2]
        MVMCOptimizers._ordinary_utu2inv_real_c_order!(2,A,2,piv,vt,M,2)
        @test all(isfinite,A)
        @test A == [0.0 -1/divisor; 1/divisor 0.0]
        @test M == Matrix{Float64}(I,2,2)
        @test vt == [-divisor,17.0]
        @test piv == [1,2]
    end

    root = joinpath(@__DIR__,"fixtures/c_ordinary_real_inverse")
    input = split(read(joinpath(root,"real4_identity.input.txt"),String))
    @test input[1:2] == ["r","4"]
    @test length(input) == 38
    factor = reshape(parse.(Float64,input[3:2:34]),4,4)
    @test all(iszero,parse.(Float64,input[4:2:34]))
    piv = parse.(Int,input[35:38]); @test piv == [1,2,3,4]
    native = split(read(joinpath(root,"real4_identity.c.txt"),String))
    @test length(native) == 74
    values = parse.(Float64,native[1:70])
    @test all(iszero,values[2:2:end])
    reference = values[1:2:end]
    @test parse.(Int,native[71:74]) == piv
    operator = reshape(ordinary_real_fixture_pairs(joinpath(root,"real4_identity.operator.txt"),16),4,4)
    for poison in (0.0,17.0), reuse in 1:2
        parent = fill(-91.0,6,6); parent[2:5,2:5] .= factor
        A = view(parent,2:5,2:5); M = fill(poison,4,4); vt = fill(poison,4)
        MVMCOptimizers._ordinary_utu2inv_real_c_order!(4,A,6,piv,vt,M,4)
        actual = vcat(vec(A),vec(M),vt[1:3])
        # Existing independent real4 budget: four <=6-wide stages, 256eps
        # abs+rel; condition estimate14.2 and separate scaled residual guard.
        @test all(abs(a-e) <= 256eps(Float64)*(1+abs(e)) for (a,e) in zip(actual,reference))
        @test length(actual) == length(reference)
        scale = opnorm(operator,Inf)*opnorm(A,Inf)+1
        @test opnorm(operator*A-I,Inf)/scale <= 256eps(Float64)
        @test all(==(-91.0),parent[[1,6],:]) && all(==(-91.0),parent[:,[1,6]])
        @test piv == [1,2,3,4]
    end
    # Exercise the actual ordinary child with independently reconstructed
    # full operator, not a factor input injected into the public child.
    inverse = zeros(4,4); pf = Ref(0.0)
    @test MVMCOptimizers.calculate_m_all_child_real!([0,1,0,1],vec(permutedims(operator)),
        inverse,pf,zeros(4,4),zeros(Int,4),zeros(4),zeros(4,4),2,4) == 0
    @test isfinite(pf[]) && all(isfinite,inverse)
    # Child applies final minus scaling for its C row-major storage contract.
    @test opnorm(operator*transpose(inverse)-I,Inf)/(opnorm(operator,Inf)*opnorm(inverse,Inf)+1) <= 256eps(Float64)

    # Independently analytic dyadic matrix with a genuine net (1,3) swap.
    # T has upper entries2,1,4. Its literal inverse solves T*X=I exactly.
    # Permuting both axes by q gives S=P*T*P^T and Xs=P*X*P^T.
    tridiagonal = [0.0 2 0 0; -2 0 1 0; 0 -1 0 4; 0 0 -4 0]
    tridiagonal_inverse = [0.0 -.5 0 -.125; .5 0 0 0; 0 0 0 -.25; .125 0 .25 0]
    @test tridiagonal*tridiagonal_inverse == Matrix{Float64}(I,4,4)
    q = [3,2,1,4]
    skew = tridiagonal[q,q]
    analytic_inverse = tridiagonal_inverse[q,q]
    factor = copy(skew); piv = zeros(Int,4)
    @test MVMCOptimizers._ordinary_dsktf2_c_order!(factor,piv) == 0
    @test piv == [1,2,1,4]
    @test any(piv .!= collect(1:4))
    @test factor[1,2] == 2.0 && factor[2,3] == 1.0 && factor[3,4] == 4.0
    work = fill(17.0,4,4); vt = fill(17.0,4)
    MVMCOptimizers._ordinary_utu2inv_real_c_order!(4,factor,4,piv,vt,work,4)
    # All operands/products are dyadic and exactly representable here;
    # literal analytical equality and zero residual, no new N6 tolerance.
    @test factor == analytic_inverse
    @test skew*factor == Matrix{Float64}(I,4,4)
    @test piv == [1,2,1,4]
    @test all(iszero,diag(factor))
    child_inverse = zeros(4,4); child_pf = Ref(0.0); child_piv = zeros(Int,4)
    @test MVMCOptimizers.calculate_m_all_child_real!([0,1,0,1],vec(permutedims(skew)),
        child_inverse,child_pf,zeros(4,4),child_piv,zeros(4),zeros(4,4),2,4) == 0
    @test child_piv == [1,2,1,4]
    @test transpose(child_inverse) == analytic_inverse
    @test skew*transpose(child_inverse) == Matrix{Float64}(I,4,4)
end
