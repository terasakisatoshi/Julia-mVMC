# C parameter.c gauge/Slater normalization for the supported real Gutz/Jastrow
# runner. Keep scalar reduction order; apply independent outputs in parallel.
struct CUDAParameterGauge
    ngutz::Int
    njastrow::Int
    nproj::Int
    shift_gj::Bool
end
function CUDAParameterGauge(data)
    layout = Parsers.projection_layout(data)
    ng,nj = layout.n_gutzwiller,layout.n_jastrow
    shift = ng > 0 && nj > 0 && (isempty(data.optimization_flags) ||
        (all(i->CPU._real_opt_flag(data,layout.gutzwiller_offset+i-1),1:ng) &&
         all(i->CPU._real_opt_flag(data,layout.jastrow_offset+i-1),1:nj)))
    return CUDAParameterGauge(ng,nj,layout.n_proj,shift)
end
function parameter_gauge_kernel!(parameters,ng,nj,nproj,shift_gj,np)
    factors = CuStaticSharedArray(Float64,2)
    if threadIdx().x == 1
        shift = 0.0
        if shift_gj
            @inbounds for i in 1:ng+nj
                shift = CUDA.add_rn(shift,parameters[i])
            end
            shift = CUDA.div_rn(shift,Float64(ng+nj))
        end
        largest = 0.0
        @inbounds for i in nproj+1:np
            x = abs(parameters[i])
            x > largest && (largest = x)
        end
        factors[1] = shift
        factors[2] = largest > 0 ? CUDA.div_rn(CPU.D_AMP_MAX,largest) : 1.0
    end
    sync_threads()
    @inbounds for i in threadIdx().x:blockDim().x:np
        if i <= ng+nj && shift_gj
            parameters[i] = CUDA.add_rn(parameters[i],-factors[1])
        elseif i > nproj
            parameters[i] = CUDA.mul_rn(parameters[i],factors[2])
        end
    end
    return nothing
end
function sync_gpu_parameters!(s::CUDAState,gauge::CUDAParameterGauge)
    @cuda threads=128 blocks=1 fastmath=false parameter_gauge_kernel!(s.parameters,
        gauge.ngutz,gauge.njastrow,gauge.nproj,gauge.shift_gj,length(s.parameters))
    return s.parameters
end
