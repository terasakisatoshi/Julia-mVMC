using Test, Random, LinearAlgebra, CUDA, MVMCOptimizers, MVMCExpertModeParsers
CUDA.device!(parse(Int, get(ENV, "JULIA_MVMC_GPU_DEVICE", "0")))
CUDA.allowscalar(false)
const CPU = MVMCOptimizers
const P = MVMCExpertModeParsers
const GPU = Base.get_extension(CPU, :MVMCCUDAExt)

include("cpu_oracle.jl")

@testset "Resident energies, Green functions and analytic SR derivatives" begin
    for model in ("hubbard_chain_real", "heisenberg_chain_real")
        path = joinpath(@__DIR__, "..", "..", "examples", "inputs", model, "namelist.def")
        data, rng = GPU.prepared_gpu_data(path; seed=79)
        cfg = GPU.initial_configurations(data, 32, rng)
        s = gpu_state(data; configurations=cfg)
        c = cpu_state(data)
        o = GPU.CUDAObservables(s, data)
        sr = GPU.CUDASR(s, data)
        gpu_observables!(s,o)
        GPU.derivatives!(s,sr)
        expected = hcat([cpu_observables(data,c,Int.(cfg[:,b]))[1] for b in 1:32]...)
        expectedO = hcat([cpu_observables(data,c,Int.(cfg[:,b]))[2] for b in 1:32]...)
        # Recomputed Pfaffians versus CPU inverse-update Green functions.
        @test Array(o.values) ≈ expected atol=1e-9 rtol=1e-9
        @test Array(sr.derivatives) ≈ expectedO atol=1e-9 rtol=1e-9
        GPU.derivatives!(s,sr,o)
        @test Array(sr.derivatives) ≈ expectedO atol=1e-9 rtol=1e-9
        energies = copy(vec(view(o.values,1,:)))
        moments = GPU.sr_moments(sr.derivatives,energies)
        @test Array(moments.meanO) ≈ vec(sum(expectedO;dims=2))/32 atol=1e-9 rtol=1e-9
        @test Array(moments.OO) ≈ expectedO*expectedO'/32 atol=1e-9 rtol=1e-9
        @test Array(moments.HO) ≈ expectedO*expected[1,:]/32 atol=1e-9 rtol=1e-9
        # Independently solve the same C covariance/stabilization equation.
        mo = vec(sum(expectedO;dims=2))/32
        me = sum(expected[1,:])/32
        S = expectedO*expectedO'/32-mo*mo'
        v = diag(S)
        cutoff = maximum(abs,v)*data.modpara.dsr_opt_red_cut
        flags = [CPU.get_opt_flag_for_parameter(data,2(p-1)) != 0 for p in eachindex(mo)]
        active = flags .& (v .> cutoff)
        rhs = -2data.modpara.dsr_opt_step_dt .* (expectedO*expected[1,:]/32 .- me.*mo)
        for i in eachindex(mo)
            if active[i]
                S[i,i] *= 1+data.modpara.dsr_opt_sta_del
            else
                S[i,:] .= 0
                S[:,i] .= 0
                S[i,i] = 1
                rhs[i] = 0
            end
        end
        expected_update = cholesky(Hermitian(S,:U)) \ rhs
        update = GPU.sr_update!(s,sr,moments,data.modpara)
        @info "CPU/GPU numerical evidence" model derivative_error=maximum(abs,Array(sr.derivatives)-expectedO) observable_error=maximum(abs,Array(o.values)-expected) sr_condition=cond(S) update_error=maximum(abs,Array(update)-expected_update) sr_residual=norm(S*Array(update)-rhs)
        # Measured worst conditioning ~5.3e6, O differences <3e-14, update
        # difference ~2e-11. Allow solver roundoff with a residual gate below.
        @test Array(update) ≈ expected_update atol=1e-9 rtol=1e-9
        @test norm(S*Array(update)-rhs) <= 1e-8*(1+norm(rhs))
    end
end

@testset "Shared UInt32 tape: proposals, rejection draws, and accept decisions" begin
    hostrng = MersenneTwister(79)
    for model in ("hubbard_chain_real", "heisenberg_chain_real"), ex in (0.0, 0.5, 1.0)
        path = joinpath(@__DIR__, "..", "..", "examples", "inputs", model, "namelist.def")
        data,rng = GPU.prepared_gpu_data(path; seed=79)
        cfg = Int.(GPU.initial_configurations(data, 8, rng))
        s = gpu_state(data; configurations=cfg)
        c = cpu_state(data)
        loc = CPU.get_loc_spn_array(data)
        ne,ns = data.modpara.nelec,data.modpara.nsite
        for step in 1:20
            tape = rand(hostrng,UInt32,5,8,1)
            expected_accepted = zeros(Int32,8)
            for b in 1:8
                original = copy(cfg[:,b])
                i = Int(tape[1,b,1] % UInt32(ne))+1
                spin = tape[2,b,1] < 0x80000000 ? 0 : 1
                i += spin*ne
                ri = cfg[i,b]
                exchange = Float64(tape[4,b,1])/4294967296.0 < ex
                j = exchange ? Int(tape[3,b,1] % UInt32(ne))+1+(1-spin)*ne : 0
                rj = exchange ? cfg[j,b] : Int(tape[3,b,1] % UInt32(ns))
                valid = if exchange
                    ri != rj && !(rj in cfg[spin*ne+1:(spin+1)*ne,b]) &&
                        !(ri in cfg[(1-spin)*ne+1:(2-spin)*ne,b])
                else
                    loc[ri+1] == 0 && loc[rj+1] == 0 && !(rj in cfg[spin*ne+1:(spin+1)*ne,b])
                end
                valid || continue
                old = cpu_overlap(data,c,original)
                cfg[i,b] = rj
                exchange && (cfg[j,b] = ri)
                new = cpu_overlap(data,c,cfg[:,b])
                logratio = 2*(new.logcorr-old.logcorr+log(abs(new.ip)+1e-100)-log(abs(old.ip)+1e-100))
                u = Float64(tape[5,b,1])/4294967296.0
                if new.ip != 0 && log(u) < logratio
                    expected_accepted[b] = 1
                else
                    cfg[:,b] .= original
                end
            end
            gpu_sample!(s;steps=1,exchange_fraction=ex,tape=CuArray(tape))
            @test Array(s.configurations) == cfg
            @test Array(s.accepted) == expected_accepted
            expected_ip = [cpu_overlap(data,c,cfg[:,b]).ip for b in 1:8]
            @test Array(s.overlaps) ≈ expected_ip atol=1e-11 rtol=1e-11
        end
    end
end

@testset "CUDA production RNG and 20-step same-implementation repeatability" begin
    path = joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    a = run_gpu_para_opt_from_namelist(path;nsteps=20,walkers=32,warmup=4,interval=4,seed=79)
    future_a = Array(CUDA.rand(UInt32,64))
    b = run_gpu_para_opt_from_namelist(path;nsteps=20,walkers=32,warmup=4,interval=4,seed=79)
    future_b = Array(CUDA.rand(UInt32,64))
    @test all(isfinite,a.energies)
    @test a.energies ≈ b.energies atol=1e-12 rtol=1e-12
    @test a.parameters ≈ b.parameters atol=1e-12 rtol=1e-12
    @test Array(a.state.configurations) == Array(b.state.configurations)
    @test future_a == future_b
    p1 = run_gpu_phys_cal_from_namelist(path;nsamples=20,walkers=32,warmup=4,interval=4,seed=79)
    future_p1 = Array(CUDA.rand(UInt32,64))
    p2 = run_gpu_phys_cal_from_namelist(path;nsamples=20,walkers=32,warmup=4,interval=4,seed=79)
    future_p2 = Array(CUDA.rand(UInt32,64))
    @test isfinite(p1.energy)
    @test p1.energy ≈ p2.energy atol=1e-12 rtol=1e-12
    @test p1.green_one ≈ p2.green_one atol=1e-12 rtol=1e-12
    @test p1.green_two ≈ p2.green_two atol=1e-12 rtol=1e-12
    @test Array(p1.state.configurations) == Array(p2.state.configurations)
    @test future_p1 == future_p2
end

@testset "CUDA input rejection before sampling" begin
    path = joinpath(@__DIR__,"..","..","examples","inputs","hubbard_chain_real","namelist.def")
    data,rng = GPU.prepared_gpu_data(path;seed=79)
    cfg = GPU.initial_configurations(data,8,rng)
    for fieldvalue in ((:lanczos_mode,1),(:nsplit_size,2),(:two_sz,2),(:nex_update_path,3))
        bad = deepcopy(data)
        setfield!(bad.modpara,fieldvalue...)
        @test_throws ArgumentError gpu_state(bad;configurations=cfg)
    end
    bad = deepcopy(data)
    bad.i_flg_orbital_general = 1
    @test_throws ArgumentError gpu_state(bad;configurations=cfg)
    bad = deepcopy(data)
    bad.modpara.nsrcg = 1
    @test_throws ArgumentError gpu_para_opt!(bad;configurations=cfg,nsteps=1)
    @test_throws ArgumentError gpu_state(data;configurations=cfg,threads=0)
    @test_throws ArgumentError gpu_state(data;configurations=fill(Int32(0),size(cfg)))
    s = gpu_state(data;configurations=cfg)
    @test_throws DimensionMismatch gpu_sample!(s;steps=1,tape=CUDA.zeros(UInt32,4,8,1))
    @test_throws ArgumentError gpu_sample!(s;steps=1,exchange_fraction=2)
end
