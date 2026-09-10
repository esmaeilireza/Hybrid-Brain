"""Week 4 tests - NeuralNetwork assembly."""
import numpy as np

from core.neural_network import NeuralNetwork

from tests.test_plasticity import PARAMS

# Homeostatic scaling would compensate the perturbations these tests
# impose - they verify the inhibitory wiring, so scaling must be off.
PARAMS = {**PARAMS, "homeostasis": {"enabled": False}}


def test_network_ticks_and_produces_activity() -> None:
    net = NeuralNetwork(PARAMS, n_exc=100, n_inh=25, seed=1)
    total = 0
    for _ in range(500):
        s_e, s_i = net.step(np.full(100, 15.0))
        total += s_e.sum() + s_i.sum()
    assert total > 100, "network produced almost no activity"


def test_inhibition_gates_excitation() -> None:
    """Removing the I population's drive must change E dynamics -
    evidence the inhibitory loop is actually wired in."""
    net = NeuralNetwork(PARAMS, n_exc=100, n_inh=25, seed=1)
    rates = []
    for inh_drive in (0.0, 8.0):
        net.pop_e.reset(); net.pop_i.reset()
        n = 0
        for _ in range(500):
            s_e, _ = net.step(np.full(100, 15.0), np.full(25, inh_drive))
            n += s_e.sum()
        rates.append(n)
    assert rates[1] < rates[0], "extra inhibition did not reduce E firing"
