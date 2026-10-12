# A warp owns one CSC column/walker dot product. Contiguous lanes read
# neighboring sparse coefficients, while each lane retains its local sum order.
# The final fixed tree changes only floating-point addition order; no RNG or
# control state is involved, and CPU derivative tolerances remain unchanged.
function warp_sparse_derivative_contraction!(O, sensitivity, colptr, rows, values, np, batch,
                                            valid=nothing, only_invalid=false)
    thread = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    output = thread ÷ 32
    lane = thread % 32
    output >= np*batch && return nothing # whole inactive warps leave together
    p = output % np+1
    b = output ÷ np+1
    if valid !== nothing && only_invalid
        all_valid = true
        @inbounds for q in 1:size(valid,1)
            all_valid &= valid[q,b]
        end
        all_valid && return nothing
    end
    result = 0.0
    @inbounds for k in colptr[p]+lane:32:colptr[p+1]-1
        result += values[k]*sensitivity[rows[k],b]
    end
    for offset in (16,8,4,2,1)
        result += shfl_down_sync(UInt32(0xffffffff),result,offset)
    end
    if lane == 0
        @inbounds O[p,b] = result
    end
    return nothing
end

function warp_sparse_contraction!(output, sensitivity, basis; valid=nothing, only_invalid=false)
    np,batch=size(output)
    np*batch == 0 && return output
    @cuda threads=256 blocks=cld(np*batch,8) fastmath=false warp_sparse_derivative_contraction!(
        output,sensitivity,basis.colPtr,basis.rowVal,basis.nzVal,np,batch,valid,only_invalid)
    return output
end

# Compose sparse basis coefficients directly with cached Pfaffian cofactors.
# No full (4*Nsite^2*Nqp, walkers) sensitivity array is needed for valid walkers.
function fused_sparse_derivative_kernel!(O, colptr, coordinates, coefficients,
                                        mapping, inv, pf, valid, overlaps, weights,
                                        ns, nq, np, batch)
    thread = (blockIdx().x-1)*blockDim().x+threadIdx().x-1
    output = thread ÷ 32
    lane = thread % 32
    output >= np*batch && return nothing
    # Adjacent warps reuse a column's coefficient/coordinate stream across
    # walkers. Output ownership and each dot-product's arithmetic stay fixed.
    b = output % batch+1
    p = output ÷ batch+1
    @inbounds for q in 1:nq
        !valid[q,b] && return nothing
    end
    result = 0.0
    @inbounds for k in colptr[p]+lane:32:colptr[p+1]-1
        ri = Int(coordinates.ri[k])
        rj = Int(coordinates.rj[k])
        q = Int(coordinates.q[k])
        i = -Int(mapping[ri,b])
        j = -Int(mapping[rj,b])
        if i > 0 && j > 0 && i < j
            cofactor = pf[q,b]*inv[j,i,q,b]
            sensitivity = -weights[q]*cofactor/overlaps[b]
            result += coefficients[k]*sensitivity
        end
    end
    for offset in (16,8,4,2,1)
        result += shfl_down_sync(UInt32(0xffffffff),result,offset)
    end
    if lane == 0
        @inbounds O[p,b] = result
    end
    return nothing
end

function fused_sparse_derivatives!(s, sr, o)
    work = o.batched_inverse
    basis = sr.sparse_basis
    batch = size(s.configurations,2)
    np = size(sr.derivatives,1)
    # Cache-only callers may not have built the observable transform. Refresh
    # the resident occupied-index lookup from the current configurations.
    @cuda threads=128 blocks=cld(batch,128) target_sites_kernel!(
        work.target_sites,work.target_map,s.configurations,s.nsite,s.nelec,batch)
    @cuda threads=256 blocks=cld(np*batch,8) fastmath=false fused_sparse_derivative_kernel!(
        sr.derivatives,basis.colPtr,sr.decoded_basis,basis.nzVal,work.target_map,
        cache_view(o.inverse),o.pfaffians,o.valid,s.overlaps,s.weights,
        s.nsite,s.nqp,np,batch)
    return sr.derivatives
end

# Slater row coordinates are invariant throughout an optimization. Decode them
# once at setup instead of repeating 64-bit division for every walker and step.
function decode_sparse_coordinates_kernel!(coordinates, rows, ns)
    k = (blockIdx().x-1)*blockDim().x+threadIdx().x
    k > length(rows) && return nothing
    @inbounds begin
        row = Int(rows[k])-1
        coordinates.ri[k] = Int32(row%(2ns)+1)
        coordinates.rj[k] = Int32((row÷(2ns))%(2ns)+1)
        coordinates.q[k] = Int32(row÷(4ns*ns)+1)
    end
    return nothing
end

function decoded_sparse_basis(basis, ns)
    count = length(basis.rowVal)
    coordinates = (;ri=CUDA.zeros(Int32,count),rj=CUDA.zeros(Int32,count),q=CUDA.zeros(Int32,count))
    if count != 0
        @cuda threads=256 blocks=cld(count,256) decode_sparse_coordinates_kernel!(
            coordinates,basis.rowVal,ns)
    end
    return coordinates
end
