using Test
using MVMCOptimizers
using MVMCExpertModeParsers
using MVMCExpertModeParsers: ExpertModeData, ModParaParameters,
    ChargeRBMHiddenLayerTerm, SpinRBMHiddenLayerTerm, GeneralRBMHiddenLayerTerm,
    ChargeRBMPhysHiddenTerm, SpinRBMPhysHiddenTerm, GeneralRBMPhysHiddenTerm

@testset "RBM counter: sum couplings before adding hidden bias" begin
    # C MakeRBMCnt (rbm.c:251-283) starts ctmp at zero, sums each
    # hidden neuron's couplings, then adds ctmp to its bias exactly once.
    # These binary-exact inputs distinguish that algorithm from accumulating
    # into the bias: (1 + 2^54) - 2^54 loses the bias; 1 + (2^54 - 2^54) does not.
    for family in (:charge, :spin, :general), complex in (false, true)
        data = ExpertModeData()
        data.modpara = ModParaParameters(nsite = 2)
        bias = ComplexF64(1, complex ? 2 : 0)
        large = ComplexF64(2.0^54, complex ? 2.0^55 : 0)
        if family == :charge
            data.charge_rbm_hidden_layer_terms = [ChargeRBMHiddenLayerTerm(0, bias, complex, 0)]
            data.charge_rbm_phys_hidden_terms = [
                ChargeRBMPhysHiddenTerm(0, 0, large, complex, 0),
                ChargeRBMPhysHiddenTerm(1, 0, -large, complex, 1),
            ]
            occupation = [1, 1, 1, 1]
        elseif family == :spin
            data.spin_rbm_hidden_layer_terms = [SpinRBMHiddenLayerTerm(0, bias, complex, 0)]
            data.spin_rbm_phys_hidden_terms = [
                SpinRBMPhysHiddenTerm(0, 0, large, complex, 0),
                SpinRBMPhysHiddenTerm(1, 0, -large, complex, 1),
            ]
            occupation = [1, 1, 0, 0]
        else
            data.general_rbm_hidden_layer_terms = [GeneralRBMHiddenLayerTerm(0, bias, complex, 0)]
            data.general_rbm_phys_hidden_terms = [
                GeneralRBMPhysHiddenTerm(0, 0, 0, large, complex, 0),
                GeneralRBMPhysHiddenTerm(1, 0, 0, -large, complex, 1),
                GeneralRBMPhysHiddenTerm(0, 1, 0, 0.0 + 0.0im, complex, 2),
                GeneralRBMPhysHiddenTerm(1, 1, 0, 0.0 + 0.0im, complex, 3),
            ]
            occupation = [1, 1, 0, 0]
        end
        before = deepcopy(data)
        @test MVMCOptimizers.make_rbm_cnt(occupation, data) == [bias]
        # Counter construction must not rewrite parameter inputs.
        hidden = Symbol(family, :_rbm_hidden_layer_terms)
        coupling = Symbol(family, :_rbm_phys_hidden_terms)
        @test getproperty(data, hidden)[1].value == getproperty(before, hidden)[1].value
        @test [t.value for t in getproperty(data, coupling)] ==
              [t.value for t in getproperty(before, coupling)]
    end
end
