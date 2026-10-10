using MVMCOptimizers, LinearAlgebra, Random, Statistics, InteractiveUtils
BLAS.set_num_threads(1)
function probe(n)
 rng=MersenneTwister(19); raw=randn(rng,2n,2n); source=vec(permutedims(raw-transpose(raw)))
 # full occupancy of n/2 sites per spin, selected from the 2n site/spin operator
 ele=vcat(collect(0:div(n,2)-1),collect(0:div(n,2)-1))
 invparent=zeros(n,n,1); inv=view(invparent,:,:,1); slater=view(source,:); pf=Ref(0.0)
 ws=MVMCOptimizers.PfaPackWorkspace(n)
 f=MVMCOptimizers.calculate_m_all_child_real!
 args=(ele,slater,inv,pf,ws.buf_m_real,ws.iwork,ws.v_t_real,ws.m_work_real,n,n)
 @assert f(args...)==0
 bytes=@allocated f(args...)
 times=[@elapsed(for _ in 1:10000; f(args...); end) for rep in 1:3]
 println("CHILD $n $(median(times)/10000) $bytes $(typeof(ws.iwork))")
 n==32 && open(ARGS[1],"w") do io; code_warntype(io,f,Tuple{map(typeof,args)...});end
end
for n in (16,24,32); probe(n); end
