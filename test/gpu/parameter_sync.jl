using Test, CUDA, MVMCOptimizers, MVMCExpertModeParsers
const GPU = Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
@testset "Resident parameter normalization follows CPU C-order gauge" begin
    path = joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng = GPU.prepared_gpu_data(path;seed=79)
    cfg = GPU.initial_configurations(data,2,rng)
    s = gpu_state(data;configurations=cfg)
    np = length(s.parameters)
    nproj = MVMCExpertModeParsers.projection_layout(data).n_proj
    for fixed in (false,true), kind in (:ordinary,:zero_slater,:small_slater)
        model = deepcopy(data)
        fixed && (model.optimization_flags[1]=false)
        params = [sin(Float64(i))*13 for i in 1:np]
        kind == :zero_slater && fill!(view(params,nproj+1:np),0.0)
        kind == :small_slater && (params[nproj+1:end] .*= 1e-200)
        MVMCOptimizers.unpack_parameters!(model,ComplexF64.(params))
        gauge = GPU.CUDAParameterGauge(model)
        MVMCOptimizers.sync_modified_parameter!(model)
        expected = real.(MVMCOptimizers.pack_parameters(model))
        copyto!(s.parameters,params)
        GPU.sync_gpu_parameters!(s,gauge)
        # Only ordered scalar sum/divide/multiply/subtract operations participate.
        @test Array(s.parameters) ≈ expected atol=1e-13 rtol=1e-13
    end
end
