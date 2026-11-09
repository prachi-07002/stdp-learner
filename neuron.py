"""Leaky integrate-and-fire (LIF) postsynaptic neuron.

    tau_m * dV/dt = -(V - V_rest) + R * I(t)

Integrated with forward Euler at a fixed timestep. When V crosses V_th the
neuron emits a spike, V is reset to V_reset, and it is refractory for
t_ref ms. Synaptic input arrives as brief current pulses: each presynaptic
spike adds w * i_peak to the input current for one timestep (a simple
instantaneous-synapse model, enough to show causal spikes driving the
postsynaptic neuron once the weight is large enough).

All voltages in mV, currents in nA, times in ms, R in MOhm.
"""

from __future__ import annotations


class LIFNeuron:
    def __init__(self, tau_m: float = 20.0, v_rest: float = -65.0,
                 v_reset: float = -65.0, v_th: float = -50.0,
                 r_m: float = 10.0, t_ref_ms: float = 2.0, dt_ms: float = 0.1):
        self.tau_m = float(tau_m)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        self.v_th = float(v_th)
        self.r_m = float(r_m)
        self.t_ref_ms = float(t_ref_ms)
        self.dt_ms = float(dt_ms)
        self.v = float(v_rest)
        self.refractory_until_ms = -1.0
        self.spike_times_ms: list[float] = []

    def reset(self) -> None:
        self.v = self.v_rest
        self.refractory_until_ms = -1.0
        self.spike_times_ms = []

    def step(self, t_ms: float, i_na: float) -> bool:
        """Advance one timestep with input current i_na. Returns True on spike."""
        if t_ms < self.refractory_until_ms:
            self.v = self.v_reset
            return False
        dv = (-(self.v - self.v_rest) + self.r_m * i_na) * self.dt_ms / self.tau_m
        self.v += dv
        if self.v >= self.v_th:
            self.v = self.v_reset
            self.refractory_until_ms = t_ms + self.t_ref_ms
            self.spike_times_ms.append(float(t_ms))
            return True
        return False

    def run(self, duration_ms: float, current_fn) -> list[float]:
        """Run for duration_ms; current_fn(t_ms) -> input current in nA.

        Returns the list of spike times (ms).
        """
        self.reset()
        t = 0.0
        n_steps = int(round(duration_ms / self.dt_ms))
        for _ in range(n_steps):
            self.step(t, float(current_fn(t)))
            t += self.dt_ms
        return list(self.spike_times_ms)
