using Test,Random,CUDA,MVMCOptimizers
CUDA.device!(parse(Int,get(ENV,"JULIA_MVMC_GPU_DEVICE","0")))
CUDA.allowscalar(false)
const GPU=Base.get_extension(MVMCOptimizers,:MVMCCUDAExt)
function destination_selector_kernel!(output,occ,localspins,spins,words,ns)
    b=(blockIdx().x-1)*blockDim().x+threadIdx().x
    b>length(output) && return nothing
    @inbounds output[b]=GPU.empty_hopping_destination(occ,view(localspins,:,b),
        Int(spins[b]),words[b],ns,b)
    return nothing
end
@testset "Empty destination ranks match independent host selection" begin
    rng=MersenneTwister(178)
    ns,batch=8,64
    occ=Int32.(rand(rng,Bool,2ns,batch))
    localspins=Int32.(rand(rng,Bool,ns,batch))
    spins=Int32.(rand(rng,0:1,batch))
    words=rand(rng,UInt32,batch)
    occ[:,1].=1 # full spin sectors
    localspins[:,2].=1 # no allowed itinerant destination
    occ[:,3].=0;localspins[:,3].=0;words[3]=typemax(UInt32)
    expected=Int32[]
    for b in 1:batch
        choices=[r-1 for r in 1:ns if occ[r+spins[b]*ns,b]==0 && localspins[r,b]==0]
        push!(expected,isempty(choices) ? -1 : choices[Int(words[b]%UInt32(length(choices)))+1])
    end
    output=CUDA.zeros(Int32,batch)
    @cuda threads=128 blocks=1 destination_selector_kernel!(output,CuArray(occ),
        CuArray(localspins),CuArray(spins),CuArray(words),ns)
    @test Array(output)==expected
    @test expected[1:2]==Int32[-1,-1]
end
