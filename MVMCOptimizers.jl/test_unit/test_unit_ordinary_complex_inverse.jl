using Test, MVMCOptimizers, LinearAlgebra

function ordinary_complex_component_agrees(a,e)
    isnan(e) && return isnan(a)
    isinf(e) && return a==e
    return isfinite(a) && abs(a-e) <= 4nextfloat(0.0)+16eps(Float64)*abs(e)
end
function ordinary_complex_pairs(words,count)
    length(words)==2count || error("complete complex fixture required")
    v=parse.(Float64,words)
    return complex.(v[1:2:end],v[2:2:end])
end

@testset "Ordinary complex private C quotient and full inverse" begin
    @test MVMCOptimizers._ordinary_complex_uses_gnu("linux","glibc")
    @test !MVMCOptimizers._ordinary_complex_uses_gnu("linux","musl")
    @test !MVMCOptimizers._ordinary_complex_uses_gnu("macos",nothing)
    root=joinpath(@__DIR__,"fixtures/c_ordinary_complex_inverse")
    for (file,divide,count) in (
        ("quotients-linux-gnu.txt",MVMCOptimizers._ordinary_complex_divide_gnu,373),
        ("quotients-llvm.txt",(z,w)->MVMCOptimizers._ordinary_complex_divide_llvm(z,w),373),
        ("quotients-macos-arm.txt",(z,w)->MVMCOptimizers._ordinary_complex_divide_llvm(z,w;arm_fma=true),375))
        rows=filter(l->!startswith(l,"#")&&!isempty(l),readlines(joinpath(root,file)))
        @test length(rows)==count
        for line in rows
            words=split(line); @test length(words)==6
            x=reinterpret.(Float64,parse.(UInt64,words;base=16))
            actual=divide(complex(x[1],x[2]),complex(x[3],x[4]))
            @test ordinary_complex_component_agrees(real(actual),x[5])
            @test ordinary_complex_component_agrees(imag(actual),x[6])
        end
    end
    host=Base.BinaryPlatforms.HostPlatform()
    z=1.0+2.0im; w=3.0-4.0im
    reference=MVMCOptimizers._ordinary_complex_uses_gnu(Base.BinaryPlatforms.os(host),Base.BinaryPlatforms.libc(host)) ?
        MVMCOptimizers._ordinary_complex_divide_gnu(z,w) :
        MVMCOptimizers._ordinary_complex_divide_llvm(z,w;arm_fma=Sys.isapple()&&Sys.ARCH==:aarch64)
    @test MVMCOptimizers._ordinary_complex_divide(z,w)==reference

    # Analytic dyadic4 covers all four solver branches, imaginary operands.
    vt=ComplexF64[2+2im,1+1im,4+4im]; B=Matrix{ComplexF64}(I,4,4); C=zeros(ComplexF64,4,4)
    MVMCOptimizers._ordinary_sktdsmx_complex_c_order!(4,vt,B,C)
    analytic=ComplexF64[0 .5 0 .125; -.5 0 0 0; 0 0 0 .25; -.125 0 -.25 0].*(.5-.5im)
    @test C==analytic
    T=ComplexF64[0 -2 0 0; 2 0 -1 0; 0 1 0 -4; 0 0 4 0].*(1+1im)
    @test T*C==B
    for (d,component) in ((2.0^1023,2.0^-1024),(2.0^-1023,2.0^1022))
        a=complex(d,d); A=ComplexF64[0 a; -a 0]
        M=fill(17.0+0.0im,2,2); vt=fill(17.0+0.0im,2)
        MVMCOptimizers._ordinary_utu2inv_complex_c_order!(2,A,2,[1,2],vt,M,2)
        expected=complex(component,-component)
        @test all(isfinite,A)
        @test A==ComplexF64[0 -expected; expected 0]
        @test M==Matrix{ComplexF64}(I,2,2)
        @test vt==[-a,17.0+0.0im]
    end
    # Genuine net nonidentity permutation, no new tolerance: all dyadic.
    q=[3,2,1,4]
    tridiagonal=-T
    skew=tridiagonal[q,q]; literal_inverse=(-analytic)[q,q]
    factor=copy(tridiagonal); piv=[1,2,1,4]
    M=fill(17.0+0.0im,4,4); vt=fill(17.0+0.0im,4)
    MVMCOptimizers._ordinary_utu2inv_complex_c_order!(4,factor,4,piv,vt,M,4)
    @test factor==literal_inverse
    @test skew*factor==Matrix{ComplexF64}(I,4,4)
    @test piv==[1,2,1,4]

    # Independent native6, existing budget/residual, all A/M/vT planes.
    input=split(read(joinpath(root,"complex6.input.txt"),String))
    @test input[1:2]==["c","6"] && length(input)==80
    f=reshape(ordinary_complex_pairs(input[3:74],36),6,6)
    piv=parse.(Int,input[75:80]); @test piv==[2,1,4,3,6,5]
    words=split(read(joinpath(root,"complex6.c.txt"),String))
    @test length(words)==160
    native=ordinary_complex_pairs(words[1:154],77)
    @test parse.(Int,words[155:160])==piv
    op=reshape(ordinary_complex_pairs(split(read(joinpath(root,"complex6.operator.txt"),String)),36),6,6)
    for poison in (0.0,17.0)
        M=fill(complex(poison,0.0),6,6); vt=fill(complex(poison,0.0),6)
        for reuse in 1:2
            parent=fill(-91.0+0.0im,8,8); parent[2:7,2:7].=f
            A=view(parent,2:7,2:7)
            MVMCOptimizers._ordinary_utu2inv_complex_c_order!(6,A,8,piv,vt,M,6)
            actual=vcat(vec(A),vec(M),vt[1:5]); @test length(actual)==length(native)
            # Retained native6 bound256eps abs+rel, condition estimate11.745;
            # four <=6-wide stages, separate backward residual. Not SR budget.
            @test all(abs(a-e)<=256eps(Float64)*(1+abs(e)) for (a,e) in zip(actual,native))
            @test opnorm(op*A-I,Inf)/(opnorm(op,Inf)*opnorm(A,Inf)+1)<=256eps(Float64)
            @test all(==(-91.0+0.0im),parent[[1,8],:])&&all(==(-91.0+0.0im),parent[:,[1,8]])
            @test piv==[2,1,4,3,6,5]
        end
    end
    # Actual ordinary complex child; existing turbo factor remains unchanged.
    inverse=zeros(ComplexF64,4,4); pf=Ref(0.0+0.0im)
    @test MVMCOptimizers.calculate_m_all_child_fcmp!([0,1,0,1],vec(permutedims(skew)),
        inverse,pf,zeros(ComplexF64,4,4),zeros(Int,4),zeros(ComplexF64,4),zeros(ComplexF64,4,4),2,4)==0
    @test isfinite(pf[])&&all(isfinite,inverse)
    @test opnorm(skew*transpose(inverse)-I,Inf)/(opnorm(skew,Inf)*opnorm(inverse,Inf)+1)<=256eps(Float64)
end
