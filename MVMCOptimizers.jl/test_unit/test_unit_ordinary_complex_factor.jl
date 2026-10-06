using Test, MVMCOptimizers, LinearAlgebra
using PfaPack: utu2pfa

function factor_native_pairs(line, n)
    values = parse.(Float64, split(line)[2:end])
    length(values) == 2n*n || error("complete native matrix required")
    all(isfinite,values) || error("nonfinite native fixture")
    return reshape(complex.(values[1:2:end],values[2:2:end]),n,n)
end

function factor_child(A, sites, nsite)
    n = size(A,1); ne = n÷2; nsite2 = 2nsite
    slater = zeros(ComplexF64,nsite2*nsite2)
    for col in 1:n, row in 1:n
        rsi = sites[col] + ((col-1)÷ne)*nsite
        rsj = sites[row] + ((row-1)÷ne)*nsite
        slater[rsi*nsite2+rsj+1] = -A[row,col]
    end
    inverse = fill(17.0+0.0im,n,n); pf = Ref(17.0+0.0im)
    piv = zeros(Int,n); buf = zeros(ComplexF64,n,n)
    vt = zeros(ComplexF64,n-1); work = zeros(ComplexF64,n,n)
    status = MVMCOptimizers.calculate_m_all_child_fcmp!(sites,slater,inverse,pf,buf,piv,vt,work,nsite,n)
    return status,pf[],inverse,piv
end

@testset "Ordinary complex native C factor order and public dispatch" begin
    root=joinpath(@__DIR__,"fixtures/c_ordinary_complex_factor")
    lines=readlines(joinpath(root,"dh4_qp1.native.txt"))
    A=factor_native_pairs(only(filter(s->startswith(s,"assembled "),lines)),6)
    native=factor_native_pairs(only(filter(s->startswith(s,"factor "),lines)),6)
    expected_piv=parse.(Int,split(only(filter(s->startswith(s,"pivots "),lines)))[2:end])
    @test expected_piv==[1,1,1,4,3,6]
    factor=copy(A); piv=zeros(Int,6)
    @test MVMCOptimizers._ordinary_zsktf2_c_order!(factor,piv)==0
    @test piv==expected_piv
    # Retained non-cancelling scalar factor entries; native zero entries remain
    # structural zeros. No matrix-wide unit-floor or CG parameter allowance.
    for i in eachindex(native)
        for (actual,reference) in ((real(factor[i]),real(native[i])),(imag(factor[i]),imag(native[i])))
            @test abs(actual-reference)<=4eps(Float64)*abs(reference)
        end
    end
    status,pf,inverse,piv=factor_child(A,[0,2,1,3,1,2],6)
    @test status==0 && piv==expected_piv
    expected_pf=62.402730079191478-68.047650211337611im
    @test abs(real(pf)-real(expected_pf))<=4eps(Float64)*abs(real(expected_pf))
    @test abs(imag(pf)-imag(expected_pf))<=4eps(Float64)*abs(imag(expected_pf))
    @test all(isfinite,inverse)
    native_inverse=factor_native_pairs(only(filter(s->startswith(s,"published-negative-inverse "),lines)),6)
    # Public child publishes the transpose of the mathematical inverse.
    # Retained native6 reference-quality gate, not a CG forward allowance.
    # Six complex products per residual entry: <=48 elementary operations.
    gamma48=(48eps(Float64)/2)/(1-48eps(Float64)/2)
    X=transpose(inverse); C=transpose(native_inverse)
    scale=opnorm(A,Inf)*opnorm(X,Inf)+1
    native_scale=opnorm(A,Inf)*opnorm(C,Inf)+1
    rho=opnorm(A*X-I,Inf); native_rho=opnorm(A*C-I,Inf)
    @test rho/scale<=gamma48
    @test native_rho/native_scale<=gamma48
    # Neumann reference-inverse identity; include residual evaluation error.
    uncertainty=gamma48*(scale+native_scale)
    @test native_rho+gamma48*native_scale<1
    forward=opnorm(C,Inf)*(rho+native_rho+uncertainty)/(1-native_rho-gamma48*native_scale)
    @test opnorm(X-C,Inf)<=forward

    # Native dyadic cancellation: grouped subtraction returned INFO1 on the
    # historical Julia path; C left association retains INFO0 and PF=-1.
    B=zeros(ComplexF64,4,4)
    for (i,j,value) in ((1,2,2.0^54),(1,3,1.0),(1,4,-2.0^54),
                         (2,3,2.0^55),(2,4,1.0),(3,4,2.0^55))
        B[i,j]=value;B[j,i]=-value
    end
    record=readlines(joinpath(root,"rank2.native.txt"))
    @test record==["complex-rank2-boundary status=0 factor=0 pfa_calls=1 inverse_calls=1 pf_re=-1 pf_im=0","pivots 1 2 3 4"]
    status,pf,inverse,piv=factor_child(B,[0,1,0,1],2)
    @test status==0
    @test piv==[1,2,3,4]
    @test abs(real(pf)+1.0)<=4eps(Float64)
    @test abs(imag(pf))<=4eps(Float64)
    @test all(isfinite,inverse)

    # Defined first zero-pivot status, before public PF/inverse publication.
    zero=zeros(ComplexF64,4,4);piv=zeros(Int,4)
    @test MVMCOptimizers._ordinary_zsktf2_c_order!(zero,piv)==3
    @test piv==[1,2,3,4]
    @test MVMCOptimizers._ordinary_zsktf2_c_order!(zeros(ComplexF64,0,0),Int[])==0
    @test_throws DimensionMismatch MVMCOptimizers._ordinary_zsktf2_c_order!(zeros(ComplexF64,4,4),zeros(Int,3))
    @test_throws ArgumentError MVMCOptimizers._ordinary_zsktf2_c_order!(zeros(ComplexF64,2,3),zeros(Int,2))
end
