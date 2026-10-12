using Test, CUDA, MVMCOptimizers, LinearAlgebra
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
@testset "Resident SR moments retain generic and CPU arithmetic" begin
    path=joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,32,rng)
    s=gpu_state(data;configurations=cfg)
    sr=GPU.CUDASR(s,data)
    O=[sin(0.13i+0.17b)+0.03i for i=1:size(sr.derivatives,1),b=1:32]
    energies=[cos(0.19b)-0.02b for b=1:32]
    copyto!(sr.derivatives,O)
    e=CuArray(energies)
    actual=GPU.sr_moments(sr,e)
    reference=GPU.sr_moments(sr.derivatives,e)
    meanO=vec(sum(O;dims=2))./32
    meanE=sum(energies)/32
    OO=(O*transpose(O))./32
    HO=(O*energies)./32
    expected=(;meanO,meanE=[meanE],OO,HO,covariance=OO-meanO*transpose(meanO))
    # Float64 BLAS reductions can differ across GPU and CPU providers. This
    # modest-scale fixture uses 32 terms; 1e-12 covers rounding, not model error.
    for field in propertynames(expected)
        @test Array(getproperty(actual,field))≈getproperty(expected,field) atol=1e-12 rtol=1e-12
        @test Array(getproperty(actual,field))≈Array(getproperty(reference,field)) atol=1e-12 rtol=1e-12
    end
end
@testset "Resident direct SR solves and retains failure status" begin
    path=joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng=GPU.prepared_gpu_data(path;seed=79)
    cfg=GPU.initial_configurations(data,2,rng)
    s=gpu_state(data;configurations=cfg)
    sr=GPU.CUDASR(s,data)
    np=length(s.parameters)
    solver=GPU.CUDASRSolver(np)
    meanO=[0.03sin(Float64(i)) for i=1:np]
    meanE=0.3
    HO=[0.05cos(Float64(i)) for i=1:np]
    covariance=Matrix(Diagonal(Float64.(1:np))) .+ 0.01
    moments=(;meanO=CuArray(meanO),meanE=CuArray([meanE]),
        HO=CuArray(HO),covariance=CuArray(covariance))
    flags=Array(sr.flags)
    threshold=maximum(abs,diag(covariance))*data.modpara.dsr_opt_red_cut
    active=flags .& (diag(covariance) .> threshold)
    rhs=-2data.modpara.dsr_opt_step_dt .* (HO .- meanE.*meanO)
    system=copy(covariance)
    for i=1:np
        if active[i]
            system[i,i]*=1+data.modpara.dsr_opt_sta_del
        else
            system[i,:].=0;system[:,i].=0;system[i,i]=1;rhs[i]=0
        end
    end
    expected=cholesky(Hermitian(system,:U))\rhs
    before=Array(s.parameters)
    update=GPU.resident_sr_update!(s,sr,moments,data.modpara,solver)
    GPU.check_sr_solver(solver)
    @test Array(update)≈expected atol=1e-12 rtol=1e-12
    @test norm(system*Array(update)-rhs)<=1e-12*(1+norm(rhs))
    @test Array(s.parameters)≈before+expected atol=1e-12 rtol=1e-12
    copyto!(sr.flags,fill(true,np))
    bad=Matrix{Float64}(I,np,np);bad[1,2]=bad[2,1]=2
    before=Array(s.parameters)
    failed=merge(moments,(;covariance=CuArray(bad)))
    GPU.resident_sr_update!(s,sr,failed,data.modpara,solver)
    @test_throws ErrorException GPU.check_sr_solver(solver)
    @test Array(s.parameters)≈before atol=1e-13 rtol=1e-13
end
