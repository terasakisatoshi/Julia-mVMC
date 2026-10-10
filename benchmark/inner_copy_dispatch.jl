using MVMCOptimizers, Statistics, LinearAlgebra
BLAS.set_num_threads(1)
ENV["JULIA_MVMC_INNER_THREADS"] = "1"
println("threads,n,kernel,serial_ns,threaded_ns,ratio")
function measure(f, repetitions)
    f()
    values = Float64[]
    for _ in 1:5
        start = time_ns()
        for _ in 1:repetitions
            f()
        end
        push!(values, (time_ns() - start) / repetitions)
    end
    median(values)
end
for n in (128, 2048, 8192, 32768, 65536, 262144, 1048576)
    real_src = [sin(Float64(i)) for i in 1:n]
    complex_src = complex.(real_src, real_src)
    real_dst = zeros(n)
    complex_dst = zeros(ComplexF64, n)
    repetitions = max(20, min(10000, cld(10^7, n)))
    for (name, f) in (("real_to_complex", threaded -> MVMCOptimizers.copy_real_to_complex!(complex_dst, real_src; threaded)),
                      ("complex_to_real", threaded -> MVMCOptimizers.copy_complex_realpart!(real_dst, complex_src; threaded)))
        serial = measure(() -> f(false), repetitions)
        parallel = measure(() -> f(true), repetitions)
        println("$(Threads.nthreads()),$n,$name,$serial,$parallel,$(parallel/serial)")
    end
end
