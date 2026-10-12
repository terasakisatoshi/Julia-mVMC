"""
    gpu_pfaffians!(values, factors, pivots)

Factor a batch of skew matrices on the device. `factors` has dimensions
`(n, n, batch)` and is overwritten; `pivots` is `(n, batch)`.
Load CUDA to enable this optional method. No host transfers are performed.
"""
function gpu_pfaffians! end

"""Optional device-resident parameter optimization entry point (CUDA extension)."""
function gpu_para_opt! end

"""Optional device-resident physical quantity entry point (CUDA extension)."""
function gpu_phys_cal! end

"""Construct a CUDA workspace from initialized ExpertModeData and fixed configurations."""
function gpu_state end

"""Recompute projected overlaps and logarithmic correlation factors on the device."""
function gpu_wavefunctions! end

"""Advance resident walkers using a CUDA UInt32 tape or CUDA-generated draws."""
function gpu_sample! end

"""Evaluate local energies and requested Green functions in resident buffers."""
function gpu_observables! end

"""Allocate reusable model tables and observable buffers for a resident CUDA state."""
function gpu_observables_workspace end

"""Run device-resident optimization from Expert-mode inputs; requires CUDA."""
function run_gpu_para_opt_from_namelist end

"""Run device-resident PhysCal from Expert-mode inputs; requires CUDA."""
function run_gpu_phys_cal_from_namelist end
