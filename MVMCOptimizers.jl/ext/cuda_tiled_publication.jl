# Transport-only kernels: source values are copied, never summed.
# Julia arrays are column-major, so pad the FIRST shared dimension to 33.
function tiled_publish_sampling_inverse_kernel!(output,packed,pf,valid,source,source_pf,source_valid,n,batch,count)
    tile=CuStaticSharedArray(Float64,(33,32))
    x,y=threadIdx().x,threadIdx().y
    first_e=(blockIdx().x-1)*32
    first_b=(blockIdx().y-1)*32
    q=blockIdx().z
    @inbounds for offset in 0:8:24
        e=first_e+x
        b=first_b+y+offset
        value=0.0
        if e<=n*n && b<=batch
            i=(e-1)%n+1;j=(e-1)÷n+1
            m=(q-1)*batch+b
            value=source_valid[q,b] ? source[i,j,q,b] : 0.0
            packed[i,j,m]=value
            if e==1
                pf[q,b]=source_pf[q,b]
                valid[q,b]=source_valid[q,b]
            end
        end
        tile[x,y+offset]=value
    end
    sync_threads()
    @inbounds for offset in 0:8:24
        e=first_e+y+offset
        b=first_b+x
        if e<=n*n && b<=batch
            i=(e-1)%n+1;j=(e-1)÷n+1
            output[i,j,q,b]=tile[y+offset,x]
        end
    end
    return nothing
end
function tiled_publish_batched_transform!(transformed,packed,target_map,n,ns,batch,count)
    tile=CuStaticSharedArray(Float64,(33,32))
    x,y=threadIdx().x,threadIdx().y
    first_e=(blockIdx().x-1)*32
    first_b=(blockIdx().y-1)*32
    q=blockIdx().z
    @inbounds for offset in 0:8:24
        e=first_e+x
        b=first_b+y+offset
        value=0.0
        if e<=n*2ns && b<=batch
            i=(e-1)%n+1;r=(e-1)÷n+1
            m=(q-1)*batch+b
            column=target_map[r,b]
            value=column>0 ? packed[i,column,m] : (i==-column ? -1.0 : 0.0)
        end
        tile[x,y+offset]=value
    end
    sync_threads()
    @inbounds for offset in 0:8:24
        e=first_e+y+offset
        b=first_b+x
        if e<=n*2ns && b<=batch
            i=(e-1)%n+1;r=(e-1)÷n+1
            transformed[i,r,q,b]=tile[y+offset,x]
        end
    end
    return nothing
end
