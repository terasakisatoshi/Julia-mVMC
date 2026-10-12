# CPU setup follows existing Julia lifecycle; all Monte Carlo and per-sample
# numerical work after setup executes on the selected CUDA device.
function initial_configurations(data, walkers, rng)
    ns = data.modpara.nsite
    ne = data.modpara.nelec
    np = CPU.count_total_parameters(data)
    nproj = Parsers.projection_layout(data).n_proj
    nq = CPU.get_n_qp_full(data)
    cpu = CPU.VMCOptimizationState(ns, ne, nproj, np, nq, 1, false, false)
    CPU.update_slater_elm_fcmp!(data, cpu)
    cpu.slater_matrix.slater_elm_real .= real.(cpu.slater_matrix.slater_elm)
    cfg = Matrix{Int32}(undef, 2ne, walkers)
    for b in 1:walkers
        e = cpu.electron_config
        info = CPU.make_initial_sample!(e.tmp_ele_idx, e.tmp_ele_cfg, e.tmp_ele_num,
                                       e.tmp_ele_proj_cnt, data, cpu, rng)
        info == 0 || error("initial configuration failed (status $info)")
        cfg[:,b] .= e.tmp_ele_idx
    end
    return cfg
end

function prepared_gpu_data(namelist; seed=nothing, initial_def=:auto, opt_para=nothing, physcal=false)
    data = Parsers.parse_expert_mode_files(String(namelist))
    validate_cuda_data(data)
    operator_tables(data)
    physcal || data.modpara.nsrcg == 0 ||
        throw(ArgumentError("CUDA optimization requires NSRCG=0"))
    data.modpara.nmp_trans = max(1, abs(data.modpara.nmp_trans))
    if data.modpara.nelec == 0
        data.modpara.nelec = (data.modpara.nlocspin + data.modpara.ncond) ÷ 2
    end
    rng = CPU.SFMT19937RNG()
    actual_seed = CPU.resolve_rnd_seed(CPU.serial_context(), data.modpara.rnd_seed, seed)
    data.modpara.rnd_seed = actual_seed
    CPU.Random.seed!(rng, actual_seed)
    Parsers.init_parameter!(data; rng)
    initial = initial_def === :auto ? joinpath(dirname(namelist), "initial.def") : initial_def
    if initial !== nothing && initial !== :none && (initial_def !== :auto || isfile(initial))
        CPU.read_initial_def!(data, String(initial)) || error("could not read initial parameters")
    end
    Parsers.read_input_parameters!(data, String(namelist))
    if opt_para !== nothing
        CPU.read_opt_para_file!(data, String(opt_para)) == CPU.count_total_parameters(data) ||
            error("incomplete optimized parameter file")
    end
    CPU.sync_modified_parameter!(data; shift_correlations=!physcal)
    Parsers.init_qp_weight!(data)
    return data, rng
end

function exchange_probability(data)
    path = data.modpara.nex_update_path
    path == 0 && return 0.0
    path == 1 && return 0.5
    path == 2 && return 1.0
    throw(ArgumentError("CUDA requires NExUpdatePath in 0:2"))
end

function validate_initial_walkers(s::CUDAState, configurations)
    # A single setup-time scalar check, before seeding or sampling.
    all(x -> isfinite(x) && x != 0, s.overlaps) ||
        throw(ArgumentError("initial walkers must have finite nonzero overlap"))
    all(isfinite, s.logcorrelations) || throw(ArgumentError("initial correlation factors must be finite"))
    loc = Array(s.localspins)
    for b in axes(configurations,2), site in eachindex(loc)
        if loc[site] == 1
            count(==(site-1), view(configurations,:,b)) == 1 ||
                throw(ArgumentError("local-spin sites must be singly occupied"))
        end
    end
    return nothing
end

function CPU.gpu_phys_cal!(data::ExpertModeData; configurations, nsamples::Integer=1,
                          warmup::Integer=10, interval::Integer=1, seed::Integer=79,
                          callback=nothing)
    nsamples > 0 && warmup >= 0 && interval >= 0 || throw(ArgumentError("invalid sampling lengths"))
    # Validate operator tables before RNG use or launch.
    operator_tables(data)
    s = CPU.gpu_state(data; configurations)
    validate_initial_walkers(s, configurations)
    o = CUDAObservables(s, data)
    CUDA.seed!(seed)
    ex = exchange_probability(data)
    CPU.gpu_sample!(s; steps=warmup, exchange_fraction=ex)
    accumulator = CUDA.zeros(Float64, size(o.values,1))
    for sample in 1:nsamples
        CPU.gpu_sample!(s; steps=interval, exchange_fraction=ex)
        CPU.gpu_observables!(s, o)
        accumulator .+= vec(sum(o.values; dims=2)) ./ size(o.values,2)
        callback !== nothing && callback(sample, s, o)
    end
    # Single final summary transfer; no saved configurations cross the bus.
    means = Array(accumulator) ./ nsamples
    return (; status=0, energy=means[1], green_one=means[2:1+length(o.one)],
            green_two=means[2+length(o.one):1+length(o.one)+length(o.two)],
            green_two_ex=means[2+length(o.one)+length(o.two):end], state=s)
end

function CPU.gpu_para_opt!(data::ExpertModeData; configurations, nsteps::Integer,
                          warmup::Integer=10, interval::Integer=1, seed::Integer=79,
                          callback=nothing)
    nsteps > 0 && warmup >= 0 && interval >= 0 || throw(ArgumentError("invalid optimization lengths"))
    data.modpara.nsrcg == 0 || throw(ArgumentError("CUDA currently uses direct SR (NSRCG=0)"))
    operator_tables(data)
    s = CPU.gpu_state(data; configurations)
    validate_initial_walkers(s, configurations)
    o = CUDAObservables(s, data; measure_greens=false)
    sr = CUDASR(s, data)
    CUDA.seed!(seed)
    ex = exchange_probability(data)
    CPU.gpu_sample!(s; steps=warmup, exchange_fraction=ex)
    energies = Float64[]
    for step in 1:nsteps
        CPU.gpu_sample!(s; steps=interval, exchange_fraction=ex)
        CPU.gpu_observables!(s, o)
        derivatives!(s, sr, o)
        e = copy(vec(view(o.values,1,:)))
        moments = sr_moments(sr.derivatives, e)
        sr_update!(s, sr, moments, data.modpara)
        # C's gauge shifts and Slater normalization at a step boundary. Only
        # O(NPara) scalars cross the bus, never configurations/OO/O/Green data.
        CPU.unpack_parameters!(data, ComplexF64.(Array(s.parameters)))
        CPU.sync_modified_parameter!(data)
        copyto!(s.parameters, real.(CPU.pack_parameters(data)))
        refresh_slater!(s)
        CPU.gpu_wavefunctions!(s)
        push!(energies, only(Array(moments.meanE)))
        callback !== nothing && callback(step, s, moments)
    end
    return (; status=0, energies, parameters=Array(s.parameters), state=s)
end

function CPU.run_gpu_para_opt_from_namelist(namelist::AbstractString;
        nsteps::Integer, walkers::Integer=256, device::Integer=0, seed::Union{Integer,Nothing}=nothing,
        initial_def=:auto, warmup::Integer=10, interval::Integer=1, callback=nothing)
    walkers > 0 || throw(ArgumentError("walkers must be positive"))
    CUDA.device!(device)
    data, rng = prepared_gpu_data(namelist; seed, initial_def)
    cfg = initial_configurations(data, walkers, rng)
    result = CPU.gpu_para_opt!(data; configurations=cfg, nsteps, warmup, interval,
                               seed=data.modpara.rnd_seed, callback)
    return merge(result, (; data))
end

function CPU.run_gpu_phys_cal_from_namelist(namelist::AbstractString;
        opt_para=nothing, nsamples::Integer=1, walkers::Integer=256, device::Integer=0,
        seed::Union{Integer,Nothing}=nothing, initial_def=:auto, warmup::Integer=10, interval::Integer=1,
        callback=nothing)
    walkers > 0 || throw(ArgumentError("walkers must be positive"))
    CUDA.device!(device)
    data, rng = prepared_gpu_data(namelist; seed, initial_def, opt_para, physcal=true)
    cfg = initial_configurations(data, walkers, rng)
    result = CPU.gpu_phys_cal!(data; configurations=cfg, nsamples, warmup, interval,
                               seed=data.modpara.rnd_seed, callback)
    return merge(result, (; data))
end
