using Test

# Independent C contract: vmcmake.c Counter[2]++ at exchange entry and
# Counter[3]++ only after acceptance; vmcmake_real.c has the same contract.
# This source-placement regression does not claim full sampling/RNG parity.
@testset "ordinary exchange counters follow C branch placement" begin
    source = read(joinpath(@__DIR__, "..", "src", "vmc_sampling.jl"), String)
    for name in ("vmc_make_sample!", "vmc_make_sample_real!")
        start = findfirst("function " * name * "(", source)
        @test start !== nothing
        tail = SubString(source, first(start))
        stop = findfirst("\nend", tail)
        body = String(SubString(tail, 1, last(stop)))
        exchange = split(body, "elseif update_type == EXCHANGE"; limit = 2)[2]
        attempt = findfirst("state.electron_config.counter[3] += 1", exchange)
        candidate = findfirst("make_candidate_exchange(", exchange)
        reject = findfirst("if reject_flag != 0", exchange)
        accepted = findfirst("state.electron_config.counter[4] += 1", exchange)
        accepted_count = findfirst("n_accept += 1", exchange)
        rejection = findnext("\n                else", exchange, last(accepted_count))
        @test first(attempt) < first(candidate) < first(reject)
        @test last(accepted_count) < first(accepted) < first(rejection)
        @test length(findall("state.electron_config.counter[3] += 1", exchange)) == 1
        @test length(findall("state.electron_config.counter[4] += 1", exchange)) == 1
    end
end
