# Native cuSOLVER calls keep status words resident until final reporting.
# The checked high-level Cholesky path remains the standalone reference path.
struct CUDASRSolver
    system::CuMatrix{Float64}
    rhs::CuVector{Float64}
    workspace::CuVector{Float64}
    factor_info::CuVector{Int32}
    solve_info::CuVector{Int32}
    failure::CuVector{Int32}
end
function CUDASRSolver(np::Integer)
    matrix = CUDA.zeros(Float64,np,np)
    count = Ref{Cint}(0)
    CUDA.cuSOLVER.cusolverDnDpotrf_bufferSize(CUDA.cuSOLVER.dense_handle(),
        'U',np,matrix,np,count)
    return CUDASRSolver(matrix,CUDA.zeros(Float64,np),CUDA.zeros(Float64,Int(count[])),
        CUDA.zeros(Int32,1),CUDA.zeros(Int32,1),CUDA.zeros(Int32,1))
end
function apply_sr_solution!(parameters,update,factor_info,solve_info,failure,np)
    i=(blockIdx().x-1)*blockDim().x+threadIdx().x
    i>np && return nothing
    @inbounds if factor_info[1] != 0 || solve_info[1] != 0
        if i==1 && failure[1]==0
            failure[1]=factor_info[1] != 0 ? factor_info[1] : solve_info[1]
        end
    elseif failure[1] == 0
        @inbounds parameters[i] += update[i]
    end
    return nothing
end
function resident_sr_update!(s,sr,moments,modpara,solver::CUDASRSolver)
    np=length(s.parameters)
    size(solver.system)==(np,np) || throw(DimensionMismatch("SR solver parameter count"))
    copyto!(solver.system,moments.covariance)
    variance=diag(solver.system)
    cutoff=maximum(abs,variance;dims=1) .* modpara.dsr_opt_red_cut
    @cuda threads=128 blocks=cld(np,128) fastmath=false sr_system_kernel!(solver.system,
        solver.rhs,variance,sr.flags,moments.meanO,moments.meanE,moments.HO,cutoff,
        modpara.dsr_opt_sta_del,modpara.dsr_opt_step_dt,np)
    handle=CUDA.cuSOLVER.dense_handle()
    CUDA.cuSOLVER.cusolverDnDpotrf(handle,'U',np,solver.system,np,
        solver.workspace,length(solver.workspace),solver.factor_info)
    CUDA.cuSOLVER.cusolverDnDpotrs(handle,'U',np,1,solver.system,np,
        solver.rhs,np,solver.solve_info)
    @cuda threads=128 blocks=cld(np,128) fastmath=false apply_sr_solution!(s.parameters,
        solver.rhs,solver.factor_info,solver.solve_info,solver.failure,np)
    return solver.rhs
end
function check_sr_solver(solver::CUDASRSolver)
    info=only(Array(solver.failure))
    info==0 || error("CUDA direct SR failed (cuSOLVER status $info)")
    return nothing
end
