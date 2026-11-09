"""Experiments built on the STDP synapse and the LIF neuron.

1. ``pairing_protocol`` -- the classic Bi & Poo pairing experiment: repeat
   pre/post spike pairs at a fixed dt and watch the weight walk up (LTP,
   dt > 0) or down (LTD, dt < 0).
2. ``pattern_learning`` -- two inputs compete for one postsynaptic neuron.
   Input A always fires 10 ms *before* the (teacher-driven) postsynaptic
   spike, input B always fires 10 ms *after* it. STDP potentiates A and
   depresses B, so the synapse population becomes selective for the causal
   input -- the timing pattern, not the firing rate, decides the winner.
   On the analog board this is the potentiometer story: the causal
   synapse's conductance is turned up, the anti-causal one's turned down.
3. ``lif_response`` -- closed-loop check: presynaptic spikes drive a LIF
   neuron through a synapse of a given weight. A weak synapse cannot make
   the postsynaptic neuron fire; a potentiated one can, a few ms after
   each presynaptic spike.
"""

from __future__ import annotations

from neuron import LIFNeuron
from stdp import StdPSynapse


def pairing_protocol(dt_ms: float, n_pairs: int = 60, w0: float = 0.5,
                     **synapse_kwargs) -> list[float]:
    """Repeat identical pre/post pairs; return weight after each pair.

    Uses ``apply_pair`` directly so each pair contributes exactly one STDP
    update (no cross-pair contamination from neighbouring trials).
    """
    syn = StdPSynapse(weight=w0, **synapse_kwargs)
    history = [syn.weight]
    for _ in range(n_pairs):
        syn.apply_pair(dt_ms)
        history.append(syn.weight)
    return history


def pattern_learning(n_trials: int = 40, w0: float = 0.5,
                     dt_causal_ms: float = 10.0,
                     dt_anticausal_ms: float = -10.0,
                     **synapse_kwargs) -> dict[str, list[float]]:
    """Two-synapse competition over repeated trials.

    Each trial: input A spikes, the postsynaptic neuron spikes
    dt_causal_ms later, then input B spikes |dt_anticausal_ms| after that.
    Returns weight histories (including the starting weight) for both
    synapses, plus the final selectivity w_A - w_B.
    """
    syn_a = StdPSynapse(weight=w0, **synapse_kwargs)
    syn_b = StdPSynapse(weight=w0, **synapse_kwargs)
    hist_a = [syn_a.weight]
    hist_b = [syn_b.weight]
    for _ in range(n_trials):
        syn_a.apply_pair(dt_causal_ms)
        syn_b.apply_pair(dt_anticausal_ms)
        hist_a.append(syn_a.weight)
        hist_b.append(syn_b.weight)
    return {
        "w_causal": hist_a,
        "w_anticausal": hist_b,
        "selectivity": [a - b for a, b in zip(hist_a, hist_b)],
    }


def lif_response(weight: float, duration_ms: float = 300.0,
                 pre_period_ms: float = 100.0, pulse_ms: float = 2.0,
                 i_peak_na: float = 30.0) -> list[float]:
    """Postsynaptic spike times when presynaptic spikes drive a LIF neuron.

    Each presynaptic spike (every ``pre_period_ms``) injects a current
    pulse of amplitude ``weight * i_peak_na`` for ``pulse_ms``. Returns the
    postsynaptic spike times in ms.
    """
    neuron = LIFNeuron()

    def current(t_ms: float) -> float:
        phase = t_ms % pre_period_ms
        return weight * i_peak_na if phase < pulse_ms else 0.0

    return neuron.run(duration_ms, current)
