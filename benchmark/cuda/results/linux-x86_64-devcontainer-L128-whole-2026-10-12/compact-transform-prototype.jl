# Isolated prototype: the original observable implementation remains default.
struct PartialTransform{V,M,C,I,S,T} <: AbstractArray{Float64,4}
    values::V
    columns::M
    original::C
    inverse::I
    slater::S
    targets::T
    ns::Int
    ne::Int
    nq::Int
    batch::Int
end
Base.size(t::PartialTransform)=(2t.ne,2t.ns,t.nq,t.batch)
@inline Base.size(t::PartialTransform,d::Int)=size(t)[d]
@inline function Base.getindex(t::PartialTransform,i::Int,r::Int,q::Int,b::Int)
    @inbounds begin
        occupied=t.targets[r,b]
        occupied<0 && return i == -occupied ? -1.0 : 0.0
        column=t.columns[r]
        column>0 && return t.values[i,column,(q-1)*t.batch+b]
        result=0.0
        for j in 1:2t.ne
            rj=t.original[j,b]+(j>t.ne ? t.ns : 0)+1
            result += t.inverse[i,j,q,b]*t.slater[rj,r,q]
        end
        return result
    end
end

struct CompactTransform
    sites::CuVector{Int32}
    columns::CuVector{Int32}
    original::CuMatrix{Int32}
    rows::CuArray{Float64,3}
    values::CuArray{Float64,3}
end
function compact_transform_sites(data)
    ops,_,one,two,ex1,_=operator_tables(data)
    sites=Int32[]
    for op in Iterators.flatten((ops,two))
        if op[7]==2 && (op[1]!=op[2] || op[4]!=op[5])
            push!(sites,op[1]+op[3]*data.modpara.nsite+1)
        end
    end
    return sort!(unique!(sites))
end
function CompactTransform(s,data)
    sites=compact_transform_sites(data)
    width=2s.nsite-2s.nelec
    s.nelec>=32 && !isempty(sites) && length(sites)<=width÷4 ||
        throw(ArgumentError("compact transform not eligible"))
    columns=zeros(Int32,2s.nsite)
    for (k,r) in enumerate(sites);columns[r]=k;end
    n,batch=2s.nelec,size(s.configurations,2)
    rows=CUDA.zeros(Float64,n,length(sites),batch*s.nqp)
    return CompactTransform(CuArray(sites),CuArray(columns),similar(s.configurations),rows,similar(rows))
end
function fill_compact_transform_rows!(rows,sites,cfg,slater,ns,ne,batch,count)
    index=(blockIdx().x-1)*blockDim().x+threadIdx().x
    n=2ne;k=size(rows,2)
    index>n*k*count && return nothing
    j=(index-1)%n+1
    column=((index-1)÷n)%k+1
    m=(index-1)÷(n*k)+1
    b=(m-1)%batch+1;q=(m-1)÷batch+1
    @inbounds rows[j,column,m]=slater[cfg[j,b]+(j>ne ? ns : 0)+1,sites[column],q]
    return nothing
end
function prepare_compact_transform!(s,o,c::CompactTransform)
    batch=size(s.configurations,2);work=o.batched_inverse
    copyto!(c.original,s.configurations)
    @cuda threads=128 blocks=cld(batch,128) target_sites_kernel!(work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
    @cuda threads=256 blocks=cld(length(c.rows),256) fastmath=false fill_compact_transform_rows!(c.rows,c.sites,c.original,s.slater,s.nsite,s.nelec,batch,batch*s.nqp)
    CUDA.cuBLAS.gemm_strided_batched!('N','N',work.alpha,work.inverse,c.rows,work.beta,c.values)
    return (c.values,c.columns,c.original,cache_view(o.inverse),s.slater,work.target_map,s.nsite,s.nelec,s.nqp,batch)
end
function compact_parallel_operator_kernel!(prefix,suffix,partial)
    transformed=PartialTransform(partial...)
    parallel_operator_kernel!(prefix...,transformed,suffix...)
    return nothing
end
function compact_scalar_observable_kernel!(prefix,suffix,partial)
    transformed=PartialTransform(partial...)
    observable_kernel!(prefix...,transformed,suffix...)
    return nothing
end
function compact_observables!(s,o,c::CompactTransform;sampling_cache=nothing)
    if sampling_cache===nothing
        batched_inverse_cache!(s,o,o.batched_inverse)
    else
        observable_sampling_cache!(s,o,sampling_cache)
    end
    partial=prepare_compact_transform!(s,o,c)
    batch=size(s.configurations,2)
    suffix=(s.nsite,s.nelec,s.ngutz,s.njastrow,s.nqp,batch)
    prefix=(o.termvalues,o.ops,o.one,o.two,o.ex1,o.ex2,s.configurations,
        s.occupancy,s.overlaps,s.logcorrelations,s.gutz,s.jastrow,s.parameters,
        s.slater,s.weights,factor_view(s.factors),pivot_view(s.pivots),s.pfaffians,
        cache_view(o.inverse),o.pfaffians,o.valid)
    count=size(o.termvalues,1)
    if count>0
        @cuda threads=256 blocks=cld(count*batch,256) fastmath=false compact_parallel_operator_kernel!(prefix,suffix,partial)
    end
    @cuda threads=128 blocks=cld(batch,128) fastmath=false ordered_observable_reduce_kernel!(o.values,o.termvalues,o.coeffs,o.valid,s.nqp,batch)
    scalarprefix=(o.values,o.ops,o.coeffs,o.one,o.two,o.ex1,o.ex2,
        s.configurations,s.counts,s.occupancy,s.overlaps,s.logcorrelations,
        s.gutz,s.jastrow,s.parameters,s.slater,s.weights,factor_view(s.factors),
        pivot_view(s.pivots),s.pfaffians,cache_view(o.inverse),o.pfaffians,o.valid)
    @cuda threads=s.threads blocks=cld(batch,s.threads) fastmath=false compact_scalar_observable_kernel!(scalarprefix,(suffix...,true),partial)
    return o.values
end
