using Test,Random,CUDA,MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)

@testset "Fused cofactors preserve sparse contraction and invalid-walker output" begin
    rng=MersenneTwister(32128)
    ns,ne,nq,batch=64,32,2,7
    n=2ne
    nrows=4ns*ns*nq
    counts=(0,1,31,32,33,127,1024,8192)
    np=length(counts)
    basis=zeros(Float64,nrows,np)
    for p in 1:np
        selected=randperm(rng,nrows)[1:counts[p]]
        basis[selected,p] .= randn(rng,counts[p])
    end
    mapping=ones(Int32,2ns,batch)
    for b in 1:batch,spin in 0:1
        sites=randperm(rng,ns)[1:ne]
        for electron in 1:ne
            mapping[sites[electron]+spin*ns,b]=-Int32(electron+spin*ne)
        end
    end
    inverse=randn(rng,n,n,nq,batch)
    pf=randn(rng,nq,batch)
    weights=[0.7,-0.4]
    overlaps=rand(rng,batch).+0.5
    sensitivity=zeros(Float64,nrows,batch)
    # Independent host construction explicitly scatters the same analytic
    # cofactors into physical Slater rows before a dense BLAS contraction.
    for b in 1:batch,q in 1:nq,rj in 1:2ns,ri in 1:2ns
        i,j=-mapping[ri,b],-mapping[rj,b]
        if i>0 && j>0 && i<j
            cofactor=pf[q,b]*inverse[j,i,q,b]
            row=ri+(rj-1)*2ns+(q-1)*4ns*ns
            sensitivity[row,b]=-weights[q]*cofactor/overlaps[b]
        end
    end
    expected=transpose(basis)*sensitivity
    sparse=CUDA.cuSPARSE.CuSparseMatrixCSC(CuArray(basis))
    coordinates=GPU.decoded_sparse_basis(sparse,ns)
    source_rows=Int.(Array(sparse.rowVal)).-1
    @test Array(coordinates.ri) == Int32.(source_rows.%(2ns).+1)
    @test Array(coordinates.rj) == Int32.((source_rows.÷(2ns)).%(2ns).+1)
    @test Array(coordinates.q) == Int32.(source_rows.÷(4ns*ns).+1)
    valid=trues(nq,batch)
    valid[2,2]=false
    dvalid=CuArray(valid)
    output=CUDA.fill(9.75,np,batch)
    @cuda threads=256 blocks=cld(np*batch,8) fastmath=false GPU.fused_sparse_derivative_kernel!(
        output,sparse.colPtr,coordinates,sparse.nzVal,CuArray(mapping),
        CuArray(inverse),CuArray(pf),dvalid,CuArray(overlaps),CuArray(weights),ns,nq,np,batch)
    actual=Array(output)
    @test actual[:,[1;3:batch]] ≈ expected[:,[1;3:batch]] atol=1e-11 rtol=1e-11
    @test all(==(9.75),actual[:,2]) # invalid walker retains fallback-owned output
    GPU.warp_sparse_contraction!(output,CuArray(sensitivity),sparse;valid=dvalid,only_invalid=true)
    @test Array(output) ≈ expected atol=1e-11 rtol=1e-11
    @test all(iszero,Array(output)[1,:])
end

@testset "Fused derivative reciprocal preserves extreme finite ratios" begin
    # Subnormal overlap overflows its reciprocal; a huge overlap has a
    # subnormal reciprocal. Both must retain division of the finite numerator.
    overlaps=[nextfloat(0.0),1e308,0.5,-0.5]
    pf=reshape(copy(overlaps),1,:)
    inverse=zeros(2,2,1,4);inverse[2,1,:,:].=1
    mapping=repeat(reshape(Int32[-1,-2],2,1),1,4)
    coordinates=(;ri=CuArray(Int32[1]),rj=CuArray(Int32[2]),q=CuArray(Int32[1]))
    output=CUDA.zeros(Float64,1,4)
    @cuda threads=128 blocks=1 fastmath=false GPU.fused_sparse_derivative_kernel!(
        output,CuArray(Int32[1,2]),coordinates,CuArray([1.0]),CuArray(mapping),
        CuArray(inverse),CuArray(pf),CUDA.fill(true,1,4),CuArray(overlaps),
        CuArray([1.0]),1,1,1,4)
    @test all(isapprox.(Array(output),-1.0;atol=1e-15,rtol=1e-15))
end
