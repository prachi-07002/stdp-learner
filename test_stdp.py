"""Unit tests for STDP, the LIF neuron, and the pairing/pattern demos.

Run:  python3 -m unittest discover -v   (from this directory)
"""

import math
import unittest

from neuron import LIFNeuron
from simulation import lif_response, pairing_protocol, pattern_learning
from stdp import StdPSynapse, stdp_update


class TestStdPRule(unittest.TestCase):
    def test_ltp_direction_small_positive_dt(self):
        # Pre 10 ms before post -> weight increases.
        self.assertGreater(stdp_update(10.0), 0.0)
        syn = StdPSynapse(weight=0.5)
        syn.apply_pair(10.0)
        self.assertGreater(syn.weight, 0.5)

    def test_ltd_direction_small_negative_dt(self):
        # Post 10 ms before pre -> weight decreases.
        self.assertLess(stdp_update(-10.0), 0.0)
        syn = StdPSynapse(weight=0.5)
        syn.apply_pair(-10.0)
        self.assertLess(syn.weight, 0.5)

    def test_magnitude_decays_with_abs_dt(self):
        # Exponential window: closer spikes change the weight more.
        self.assertGreater(abs(stdp_update(5.0)), abs(stdp_update(20.0)))
        self.assertGreater(abs(stdp_update(20.0)), abs(stdp_update(40.0)))
        self.assertGreater(abs(stdp_update(-5.0)), abs(stdp_update(-20.0)))
        self.assertGreater(abs(stdp_update(-20.0)), abs(stdp_update(-40.0)))
        # Exact exponential form at the textbook parameters.
        self.assertAlmostEqual(stdp_update(10.0), 0.01 * math.exp(-0.5))
        self.assertAlmostEqual(stdp_update(-10.0), -0.01 * math.exp(-0.5))

    def test_no_change_for_large_abs_dt(self):
        # Spikes 200 ms apart (10 time constants) are effectively unpaired.
        self.assertLess(abs(stdp_update(200.0)), 1e-6)
        self.assertLess(abs(stdp_update(-200.0)), 1e-6)
        syn = StdPSynapse(weight=0.5)
        syn.apply_pair(200.0)
        self.assertAlmostEqual(syn.weight, 0.5, places=5)

    def test_zero_dt_no_change(self):
        self.assertEqual(stdp_update(0.0), 0.0)

    def test_weight_stays_bounded(self):
        syn = StdPSynapse(weight=0.99, w_max=1.0)
        for _ in range(500):
            syn.apply_pair(5.0)  # strong repeated LTP
        self.assertLessEqual(syn.weight, 1.0)
        self.assertEqual(syn.weight, 1.0)
        syn = StdPSynapse(weight=0.01, w_max=1.0)
        for _ in range(500):
            syn.apply_pair(-5.0)  # strong repeated LTD
        self.assertGreaterEqual(syn.weight, 0.0)
        self.assertEqual(syn.weight, 0.0)

    def test_repeated_pairings_monotonic(self):
        ltp = pairing_protocol(10.0, n_pairs=60, w0=0.5)
        self.assertTrue(all(b >= a for a, b in zip(ltp, ltp[1:])),
                        "LTP weights must not decrease")
        self.assertGreater(ltp[-1], ltp[0])
        ltd = pairing_protocol(-10.0, n_pairs=60, w0=0.5)
        self.assertTrue(all(b <= a for a, b in zip(ltd, ltd[1:])),
                        "LTD weights must not increase")
        self.assertLess(ltd[-1], ltd[0])

    def test_event_driven_pairing(self):
        # Nearest-neighbour pairing through the spike API agrees in sign.
        syn = StdPSynapse(weight=0.5)
        syn.pre_spike(100.0)
        syn.post_spike(110.0)  # dt = +10 -> LTP
        self.assertGreater(syn.weight, 0.5)
        syn = StdPSynapse(weight=0.5)
        syn.post_spike(100.0)
        syn.pre_spike(110.0)  # dt = -10 -> LTD
        self.assertLess(syn.weight, 0.5)


class TestPatternLearning(unittest.TestCase):
    def test_causal_potentiates_anticausal_depresses(self):
        res = pattern_learning(n_trials=40, w0=0.5)
        self.assertGreater(res["w_causal"][-1], 0.5)
        self.assertLess(res["w_anticausal"][-1], 0.5)
        self.assertGreater(res["selectivity"][-1], 0.3)
        # Selectivity grows monotonically over trials.
        sel = res["selectivity"]
        self.assertTrue(all(b >= a for a, b in zip(sel, sel[1:])))


class TestLIFNeuron(unittest.TestCase):
    def test_no_input_no_spike(self):
        neuron = LIFNeuron()
        self.assertEqual(neuron.run(200.0, lambda t: 0.0), [])

    def test_strong_current_spikes(self):
        neuron = LIFNeuron()
        spikes = neuron.run(200.0, lambda t: 10.0)
        self.assertTrue(len(spikes) > 0)

    def test_weak_synapse_no_post_spike_strong_synapse_spikes(self):
        # Same presynaptic drive; only the potentiated synapse fires the
        # postsynaptic neuron -- learning changes behaviour, not just a number.
        self.assertEqual(lif_response(0.3), [])
        strong = lif_response(0.9)
        self.assertTrue(len(strong) > 0)
        # Post spikes follow presynaptic spikes (at 0, 100, 200 ms) closely.
        for s in strong:
            self.assertLess(min(abs(s - p) for p in (0.0, 100.0, 200.0)), 10.0)


if __name__ == "__main__":
    unittest.main()
