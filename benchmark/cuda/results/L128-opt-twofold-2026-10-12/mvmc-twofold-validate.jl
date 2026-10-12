using Serialization, Test, Statistics
const DIR=ARGS[1]
reference=deserialize(joinpath(DIR,"baseline-1.bin"))
times=Dict(v=>Float64[] for v in ("baseline","candidate"))
@testset "Independent public Opt20: same control and numerical results" begin
    for rep in 1:5, variant in ("baseline","candidate")
        r=deserialize(joinpath(DIR,"$(variant)-$(rep).bin"))
        push!(times[variant],r.seconds)
        for field in (:cfg,:occupancy,:counts,:accepted,:future)
            equal=getproperty(r,field)==getproperty(reference,field)
            @test equal
        end
        energy_ok=all(isapprox.(r.values,reference.values;atol=1e-9,rtol=1e-9))
        parameters_ok=all(isapprox.(r.parameters,reference.parameters;atol=1e-9,rtol=1e-9))
        @test energy_ok
        @test parameters_ok
        println("rep=",rep," variant=",variant,
                " max_abs_energy_diff=",maximum(abs,r.values-reference.values),
                " max_abs_parameter_diff=",maximum(abs,r.parameters-reference.parameters))
    end
end
println("baseline_seconds=",times["baseline"]," candidate_seconds=",times["candidate"])
println("baseline_median=",median(times["baseline"]),
        " candidate_median=",median(times["candidate"]),
        " speedup=",median(times["baseline"])/median(times["candidate"]),
        " frozen_original_speedup=",10.462263521/median(times["candidate"]))
