# CUDA による実行

変分最適化と PhysCal に、実験的な CUDA.jl 6 バックエンドを追加しました。
CPU/SFMT の既存ランナーは CUDA に依存せず、そのまま利用できます。
GPU では複数の独立 walker をまとめて処理し、配置・Pfaffian・逆行列・
観測量・変分微分・SR 行列をデバイスに保持します。

```sh
julia +1.13.1 --project=test/gpu -e 'using Pkg; Pkg.instantiate(); Pkg.build(["SFMT", "PfaPack"])'
```

```julia
using CUDA, MVMCOptimizers
CUDA.allowscalar(false)
input = "examples/inputs/hubbard_chain_real/namelist.def"
opt = run_gpu_para_opt_from_namelist(input;
    nsteps=20, walkers=256, device=0, seed=79, warmup=100, interval=20)
phys = run_gpu_phys_cal_from_namelist(input;
    nsamples=20, walkers=256, device=0, seed=79, warmup=100, interval=20)
```

`device` は 0 始まりです。`warmup` と `interval` は walker ごとの提案回数で、
明示指定時は CPU 入力の sweep 数とは異なります。省略時は
`walkers=NVMCSample`、`interval=NVMCInterval*Nsite`、
`warmup=NVMCWarmUp*NVMCInterval*Nsite` を使います。PhysCal の反復数は
`NDataQtySmp` に従います。各 walker を個別に熱平衡化します。
seed を省略すると入力の `RndSeed` に従います。
`initial_def=:auto` は隣接する `initial.def` を読み込みます。
PhysCal の `opt_para="zqp_opt.dat"` で最適化済みパラメータを指定できます。
最適化は energy の列と最終パラメータ、PhysCal は energy と入力順の Green 配列を返します。
GPU ランナーは CPU の C 形式ファイルを出力せず、計算結果を返す API です。

現時点では実数・Sz 保存・`2Sz=0`、Gutzwiller/Jastrow、固定されたスピン・並進射影、
通常の Hamiltonian、1体・直接2体・積形式2体 Green 関数、直接 SR (`NSRCG=0`) に対応します。
FSZ、複素数ランナー、RBM、doublon-holon、変分 OptTrans、Lanczos、MPI split は拒否します。
詳細な対応範囲と転送境界は [英語版](../en/gpu.md) に記載しています。

乱数は検証環境の CUDA.jl 6.4.2 では cuRAND の XORWOW を使います。
提案ごとに UInt32 を5語 GPU 上で生成し、無効・棄却された提案でも固定の5語を消費します。
CPU の SFMT と同じ seed でも軌跡は一致しません。
共通の UInt32 列を渡す CPU oracle との比較、観測量・微分・SR 更新の比較、
20ステップの同一実装再現性と将来の整数乱数列の一致で検証します。

```sh
JULIA_MVMC_GPU_DEVICE=0 julia +1.13.1 --project=test/gpu test/gpu/runtests.jl
JULIA_MVMC_GPU_DEVICE=1 JULIA_MVMC_GPU_PROFILE=1 \
  julia +1.13.1 --project=test/gpu test/gpu/benchmark.jl
```

2台ある場合は GPU 0 をデバッグ、GPU 1 をベンチマークに分けられます。
最適化・PhysCal ランナー自体は1台の GPU で実行します。
`test/gpu/benchmark_multi_gpu.jl` は同じ固定配置バッチを2台に分割し、
永続的な Julia worker で並列に観測量・微分を計算します。
この計測は分散 SR や Monte Carlo 全体の高速化率ではありません。
ウォームアップ後の同期済み計測と
CUPTI によるループ内転送の検査を行い、[測定結果](../en/gpu_benchmark.md) に記録します。
小さなバッチでは起動・計算のコストが効くため、GPU を使えば常に高速になるとは限りません。

L128 の Hubbard 入力を `benchmark/cuda/inputs/hubbard_chain_L128` に追加しました。
大行列では Pfaffian の協調並列化、GPU 上の batched LU 逆行列、
Slater 列のキャッシュ、演算子ごとの並列化、疎な解析微分の直接縮約を使います。
特異な射影行列には既存の minor 計算を残します。
大きな格子では配置を GPU 上で初期化し、すべての射影行列が非特異であることを
検証します。Slater 微分基底は直接 CSC 形式で構築し、L128 では約 12.44 MB
（密行列では約 3.24 GB）です。サンプリングは Pfaffian・逆行列の更新を使い、
パラメータ更新の境界で batched LU により再計算します。観測量もこのキャッシュを使います。
SR のしきい値、パラメータのゲージ補正、ソルバー状態、エネルギー履歴は GPU に
保持し、最終結果の返却時に転送します。

全体の性能は `test/gpu/benchmark_end_to_end.jl` で計測します。
入力読込・初期化・サンプリング・SR・結果返却を含み、初回コンパイルを分けます。
CPU と GPU の保存サンプル総数は揃えますが、独立 walker と単一 CPU チェーンでは
熱平衡化の提案数や軌跡が異なります。詳細は [測定結果](../en/gpu_benchmark.md) を参照してください。
`JULIA_MVMC_GPU_INVERSE=cooperative` で Gauss-Jordan の比較経路を選択できます。
