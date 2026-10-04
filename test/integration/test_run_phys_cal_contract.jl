using Test
using MVMCOptimizers

# Cheap contract test for run_phys_cal_from_namelist against a committed fixture:
# the namelist must PARSE, then the strict fixed-parameter load must FAIL *before*
# any sampling. This pins the corrected init order's early stages (parse → load)
# without running a full PhysCal sweep — the numeric PhysCal-vs-C comparison is the
# Plan 3b e2e gate's job. A loader error here proves sampling was never reached
# (read_opt_para_file! is step 3; vmc_phys_cal! / mkpath is step 6).

@testset "run_phys_cal_from_namelist: errors at the loader, before sampling" begin
    namelist = joinpath(
        @__DIR__, "reference", "heisenberg_chain_real", "inputs", "namelist.def",
    )
    @test isfile(namelist)

    mktempdir() do dir
        # (a) Missing fixed-parameter file: parse succeeds, loader errors, and no
        #     output directory is created (we never reach mkpath/sampling).
        out_a = joinpath(dir, "out_a")
        err = try
            run_phys_cal_from_namelist(
                namelist; opt_para = joinpath(dir, "nope.dat"), mode = :real,
                output_dir = out_a,
            )
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("read_opt_para_file!", msg)
        @test occursin("file not found", msg)
        @test !isdir(out_a)

        # (b) Malformed (too-short) fixed-parameter file: parse succeeds, loader
        #     errors before sampling.
        bad = joinpath(dir, "bad_zqp_opt.dat")
        write(bad, "1.0 2.0 3.0\n")  # far fewer than 6 + 3*NPara floats
        out_b = joinpath(dir, "out_b")
        err2 = try
            run_phys_cal_from_namelist(
                namelist; opt_para = bad, mode = :real, output_dir = out_b,
            )
            nothing
        catch e
            e
        end
        @test err2 !== nothing
        msg2 = sprint(showerror, err2)
        @test occursin("read_opt_para_file!", msg2)
        @test occursin("too short", msg2)
        @test !isdir(out_b)
    end
end

# Regression: `MVMC_C_TIMER=1` must emit the C-format PhysCal section report
# (`OutputTimerPhysCal`) and must not perturb the sampling/output path. The
# fixture is the committed fixed-parameter PhysCal set (NDataQtySmp=1), so one
# sample per run keeps this cheap.
@testset "run_phys_cal_from_namelist: MVMC_C_TIMER report without numerical change" begin
    refdir = joinpath(@__DIR__, "reference", "heisenberg_chain_real", "physcal_ref")
    namelist = joinpath(refdir, "inputs", "namelist.def")
    opt_para = joinpath(refdir, "zqp_opt.dat")
    @test isfile(namelist)
    @test isfile(opt_para)

    timer_labels = [
        "All                         [0] ",
        "Initialization              [1] ",
        "  read options             [10] ",
        "  ReadDefFile              [11] ",
        "  SetMemory                [12] ",
        "  InitParameter            [13] ",
        "VMCPhysCal                  [2] ",
        "  VMCMakeSample             [3] ",
        "    makeInitialSample      [30] ",
        "    make candidate         [31] ",
        "    hopping update         [32] ",
        "      UpdateProjCnt        [60] ",
        "      CalculateNewPfM2     [61] ",
        "      CalculateLogIP       [62] ",
        "      UpdateMAll           [63] ",
        "    exchange update        [33] ",
        "      UpdateProjCnt        [65] ",
        "      CalculateNewPfMTwo2  [66] ",
        "      CalculateLogIP       [67] ",
        "      UpdateMAllTwo        [68] ",
        "    lspinflip update       [36] ",
        "      UpdateProjCnt       [600] ",
        "      CalculateNewPfMTwo2 [601] ",
        "      CalculateLogIP      [602] ",
        "      UpdateMAllTwo       [603] ",
        "    recal PfM and InvM     [34] ",
        "    save electron config   [35] ",
        "  VMCMainCal                [4] ",
        "    CalculateMAll          [40] ",
        "    LocEnergyCal           [41] ",
        "      CalHamiltonian0      [70] ",
        "      CalHamiltonian1      [71] ",
        "      CalHamiltonian2      [72] ",
        "    CalculateGreenFunc     [42] ",
        "      GreenFunc1           [50] ",
        "      GreenFunc2           [51] ",
        "      addPhysCA            [52] ",
        "      addPhysCACA          [53] ",
        "    Lanczos1               [43] ",
        "    Lanczos2               [44] ",
        "  UpdateSlaterElm          [20] ",
        "  WeightAverage            [21] ",
        "  outputData               [22] ",
    ]

    mktempdir() do dir
        off_dir = joinpath(dir, "timer_off")
        on_dir = joinpath(dir, "timer_on")

        withenv(
            "MVMC_C_TIMER" => nothing,
            "MVMC_TIMER" => nothing,
            "MVMC_MAINCAL_DIAG" => nothing,
            "MVMC_CALHAM1_DIAG" => nothing,
            "MVMC_SLATER_DIAG" => nothing,
            "MVMC_WEIGHTAVG_DIAG" => nothing,
        ) do
            run_phys_cal_from_namelist(
                namelist; opt_para = opt_para, mode = :real, output_dir = off_dir,
            )
        end
        withenv("MVMC_C_TIMER" => "1") do
            run_phys_cal_from_namelist(
                namelist; opt_para = opt_para, mode = :real, output_dir = on_dir,
            )
        end

        @test !isfile(joinpath(off_dir, "zvo_CalcTimer.dat"))
        timer_path = joinpath(on_dir, "zvo_CalcTimer.dat")
        @test isfile(timer_path)

        lines = readlines(timer_path)
        @test length(lines) == length(timer_labels)
        for (line, label) in zip(lines, timer_labels)
            @test startswith(line, label)
            @test !isnothing(tryparse(Float64, strip(line[length(label)+1:end])))
        end

        # The report is additive: every other output must be byte-identical, so
        # the timer does not perturb RNG, sampling or accumulated quantities.
        off_files = sort(filter(f -> f != "zvo_CalcTimer.dat", readdir(off_dir)))
        on_files = sort(filter(f -> f != "zvo_CalcTimer.dat", readdir(on_dir)))
        @test off_files == on_files
        @test !isempty(off_files)
        for name in off_files
            @test read(joinpath(off_dir, name)) == read(joinpath(on_dir, name))
        end

        # A diagnostic family enables the parent timer and writes the diag file
        # (vmc_main_cal! derives the diag timers from the same CTimer).
        diag_dir = joinpath(dir, "timer_diag")
        withenv("MVMC_C_TIMER" => "1", "MVMC_CALHAM1_DIAG" => "1") do
            run_phys_cal_from_namelist(
                namelist; opt_para = opt_para, mode = :real, output_dir = diag_dir,
            )
        end
        diag_path = joinpath(diag_dir, "zvo_CalcTimerDiag.dat")
        @test isfile(diag_path)
        diag_text = read(diag_path, String)
        @test occursin("CalH1 GreenFunc1Real       [920] ", diag_text)
        @test occursin("WeightAverage diagnostic   [960] ", diag_text)
    end
end
