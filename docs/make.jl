using Documenter

const CUDA_BENCHMARK_REPORT = read(joinpath(@__DIR__, "..", "benchmark", "cuda", "results",
    "linux-x86_64-rtx3060-2026-10-12.md"), String) * "\n\n---\n\n" *
    read(joinpath(@__DIR__, "..", "benchmark", "cuda", "results",
    "linux-x86_64-rtx3060-L128-2026-10-12.md"), String) * "\n\n---\n\n" *
    read(joinpath(@__DIR__, "..", "benchmark", "cuda", "results",
    "linux-x86_64-rtx3060-L128-whole-2026-10-12.md"), String)

const PAGES = [
    "Home" => "index.md",
    "English" => [
        "Overview" => "en/index.md",
        "Installation" => "en/installation.md",
        "Tutorial" => "en/tutorial.md",
        "Input files" => "en/input_files.md",
        "Optimization" => "en/optimization.md",
        "Physics calculation" => "en/physics_calc.md",
        "CUDA execution" => "en/gpu.md",
        "CUDA measurements" => "en/gpu_benchmark.md",
        "Output files" => "en/output_files.md",
        "Compatibility" => "en/compatibility.md",
    ],
    "日本語" => [
        "概要" => "ja/index.md",
        "インストール" => "ja/installation.md",
        "チュートリアル" => "ja/tutorial.md",
        "入力ファイル" => "ja/input_files.md",
        "最適化" => "ja/optimization.md",
        "物理量計算" => "ja/physics_calc.md",
        "CUDA 実行" => "ja/gpu.md",
        "出力ファイル" => "ja/output_files.md",
        "互換性" => "ja/compatibility.md",
    ],
]

makedocs(;
    sitename = "Julia-mVMC",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        edit_link = "main",
        canonical = "https://tmisawa.github.io/Julia-mVMC/stable/",
        mathengine = Documenter.KaTeX(),
    ),
    pages = PAGES,
    pagesonly = true,
    doctest = false,
    checkdocs = :none,
)

if get(ENV, "CI", "false") == "true"
    deploydocs(;
        repo = "github.com/tmisawa/Julia-mVMC.git",
        devbranch = "main",
    )
end
