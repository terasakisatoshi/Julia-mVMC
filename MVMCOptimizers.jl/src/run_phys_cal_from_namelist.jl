"""
    run_phys_cal_from_namelist(namelist_path; opt_para, mode, seed=nothing,
                               output_dir=tempname()) -> NamedTuple

Drive `vmc_phys_cal!` (VMCPhysCal mode) from a C-mVMC `namelist.def`, loading a
committed *fixed* optimized-parameter file (`zqp_opt.dat`) so the Green-function
output is deterministic and decoupled from optimization. This is the entry point
used by the PhysCal reference gate.

# Init order (PhysCal — note the difference from `run_para_opt_from_namelist`)

    parse → seed RNG → read_opt_para_file! → read_input_parameters!
          → sync_modified_parameter! → vmc_phys_cal!(data; rng, output_dir)

Unlike `run_para_opt_from_namelist`, this runner deliberately does **not** call
`init_parameter!` or `init_qp_weight!`. `vmc_phys_cal!` performs both internally:
its save → `init_parameter!` → restore dance consumes the SFMT RNG exactly once
to match C's single `InitParameter()` (`vmc_phys_cal.jl`), and it calls
`init_qp_weight!` itself. Calling either here as well would advance the RNG a
second time before sampling and desynchronise the Monte-Carlo stream from the C
reference — a bit-level failure that looks like a BLAS/platform issue but is not.
The runner only loads + overlays + syncs the parameters; `vmc_phys_cal!`'s
internal save/restore preserves those values across its single internal
`init_parameter!`.

`read_opt_para_file!` runs right after parsing: it has no `init_parameter!`
precondition (the parser already created the Gutzwiller/Jastrow/orbital term
arrays and set `modpara.n_orbital_idx`; the loader only sets each term's
`.value`).

# Arguments
- `namelist_path::AbstractString`: path to `namelist.def`; sibling `.def` files
  resolve relative to it.
- `opt_para::AbstractString` (**required**): path to the committed fixed
  `zqp_opt.dat`. Explicit and required — the gate must be deterministic about
  which parameters it consumed.
- `mode::Symbol`: `:real` / `:cmp` / `:fsz` sanity label (validated for the
  documented set; the real execution path is determined by the parsed `.def`
  files, as in `run_para_opt_from_namelist`, so it is not passed downstream).
- `seed::Union{Integer,Nothing} = nothing`: SFMT19937 seed. `nothing` resolves
  `modpara.RndSeed` with the same C-parity `resolve_rnd_seed` rule as
  `run_para_opt_from_namelist`; under MPI the per-group `+ group1` offset is
  applied so each `NSplitSize=1` rank runs an independent chain.
- `output_dir::AbstractString = tempname()`: directory for the `zvo_*` outputs
  (created if absent).

# Timer
With `MVMC_C_TIMER=1` (or the deprecated `MVMC_TIMER`), writes
`zvo_CalcTimer.dat` in C's `OutputTimerPhysCal` format; the `MVMC_*_DIAG`
variables additionally write `zvo_CalcTimerDiag.dat` for the sections they
instrument. The timer is off by default and does not change sampling, output or
the RNG stream.

Sampling counts (`NDataQtySmp`, `NVMCSample`) come from `modpara`; there is no
`nsmp` override — in PhysCal those counts are distinct and easy to confuse, so
`modpara` (from the namelist) is the single source of truth.

# Returns
A `NamedTuple` `(; status, output_dir, n_para_consumed)`:
- `status::Int`: exit code from `vmc_phys_cal!` (0 = OK).
- `output_dir::String`: absolute path to the run outputs.
- `n_para_consumed::Int`: number of fixed parameters applied
  (`n_proj + n_slater`); the gate can assert this is `> 0`.
"""
function run_phys_cal_from_namelist(
    namelist_path::AbstractString;
    opt_para::AbstractString,
    mode::Symbol,
    seed::Union{Integer,Nothing} = nothing,
    output_dir::AbstractString = tempname(),
)
    mode in (:real, :cmp, :fsz) ||
        throw(ArgumentError("mode must be :real, :cmp, or :fsz; got :$mode"))

    namelist_str = String(namelist_path)

    # C-compatible section timer. Enabled by MVMC_C_TIMER (MVMC_TIMER kept as a
    # deprecated alias). This is the single place that reads the env var and
    # constructs the concrete CTimer; it flows into vmc_phys_cal! through a
    # function barrier, exactly like run_para_opt_from_namelist.
    c_timer_env = ctimer_env_enabled("MVMC_C_TIMER")
    legacy_timer_env = ctimer_env_enabled("MVMC_TIMER")
    diag_envs = ctimer_diag_envs()
    if legacy_timer_env && !c_timer_env
        @warn "MVMC_TIMER is deprecated; use MVMC_C_TIMER=1 for the C-compatible zvo_CalcTimer.dat timer."
    end
    timer_enabled = c_timer_env || legacy_timer_env || ctimer_any_diag_enabled(diag_envs)
    c_timer = CTimer(timer_enabled)
    ctimer_reset!(c_timer)

    ctimer_start!(c_timer, 0)   # [0] All
    ctimer_start!(c_timer, 1)   # [1] Initialization

    # 1. Parse expert-mode .def files (relative paths resolved from namelist_path).
    ctimer_start!(c_timer, 11)  # [11] ReadDefFile
    data = MVMCExpertModeParsers.parse_expert_mode_files(namelist_str)
    ctimer_stop!(c_timer, 11)
    validate_supported_modpara(data.modpara)
    validate_supported_phys_cal_modpara(data.modpara)
    validate_supported_phys_cal_data(data)
    ctx = build_parallel_context(data.modpara.nsplit_size)

    # 2. Seed the SFMT19937 RNG (C-compatible convention) and pass it to
    #    vmc_phys_cal! so its single internal RNG consumption is reproducible.
    rng = SFMT19937RNG()
    actual_seed = resolve_rnd_seed(ctx, data.modpara.rnd_seed, seed)
    Random.seed!(rng, actual_seed)

    # 3-5. Fixed-parameter load, In*.def overlays and sync. C bundles
    #      InitParameter + ReadInputParameters + SyncModifiedParameter under
    #      [13] InitParameter (vmcmain.c:261-277). vmc_phys_cal!'s internal
    #      init_parameter! (the single RNG-consuming call) stays inside [2] to
    #      preserve PhysCal's save/restore ordering.
    ctimer_start!(c_timer, 13)  # [13] InitParameter

    # 3. Load the fixed optimized parameters (strict; returns the consumed count).
    n_para_consumed = read_opt_para_file!(data, String(opt_para))

    # 4. In*.def overlays (C ReadInputParameters), after the fixed load so any
    #    overlay wins — matching C's order.
    read_input_parameters!(data, namelist_str)

    # 5. Slater rescale before PhysCal. C enables DH/GJ shift flags only in Opt mode.
    #    MPI R1: rank0 fixed parameters are broadcast before local sync, matching
    #    C's SyncModifiedParameter(comm_parent) shape while keeping PhysCal's
    #    shift_correlations=false contract.
    sync_modified_parameter!(ctx, data; shift_correlations = false)
    ctimer_stop!(c_timer, 13)
    ctimer_stop!(c_timer, 1)   # end [1] Initialization

    # 6. PhysCal. vmc_phys_cal! owns the single init_parameter! (RNG match) and
    #    init_qp_weight!; its save/restore preserves the params loaded above.
    out_dir = abspath(String(output_dir))  # absolute, matching run_para_opt_from_namelist
    mkpath(out_dir)
    ctimer_start!(c_timer, 2)   # [2] VMCPhysCal
    status = vmc_phys_cal!(
        data;
        rng = rng,
        output_dir = out_dir,
        ctx = ctx,
        c_timer = c_timer,
    )
    ctimer_stop!(c_timer, 2)
    ctimer_stop!(c_timer, 0)    # end [0] All

    if timer_enabled && is_output_rank(ctx)
        write_ctimer_phys_cal(c_timer, out_dir)
        if ctimer_any_diag_enabled(diag_envs)
            write_ctimer_diag(c_timer, out_dir)
        end
    end

    return (; status = status, output_dir = out_dir, n_para_consumed = n_para_consumed)
end
