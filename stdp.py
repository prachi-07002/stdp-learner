"""Spike-timing-dependent plasticity (STDP).

Classic pair-based STDP (Bi & Poo, 1998): the change in a synaptic weight
depends only on the timing difference between a presynaptic spike and a
postsynaptic spike,

    dt = t_post - t_pre

* dt > 0  (pre before post, causal)  -> potentiation (LTP)
      dw =  A_plus  * exp(-dt / tau_plus)
* dt < 0  (post before pre)          -> depression (LTD)
      dw = -A_minus * exp(dt / tau_minus)      (dt is negative here)

The magnitude decays exponentially with |dt|, so only near-coincident
spikes (within a few tens of ms) change the weight appreciably. Weights are
hard-bounded to [0, w_max], which mirrors the finite conductance range of a
real (e.g. memristor / potentiometer-set) synapse on an analog board.

All times are in milliseconds, weights are dimensionless conductances in
[0, w_max]. Defaults are textbook values: A_plus = A_minus = 0.01,
tau_plus = tau_minus = 20 ms, w_max = 1.0.
"""

from __future__ import annotations

import math


def stdp_update(dt_ms: float, a_plus: float = 0.01, a_minus: float = 0.01,
                tau_plus: float = 20.0, tau_minus: float = 20.0) -> float:
    """Weight change for one pre/post spike pair.

    Args:
        dt_ms: t_post - t_pre in milliseconds.
        a_plus / a_minus: peak LTP / LTD amplitudes.
        tau_plus / tau_minus: decay time constants of the two windows.

    Returns:
        dw (positive for LTP, negative for LTD, ~0 for large |dt|).
    """
    if dt_ms > 0:
        return a_plus * math.exp(-dt_ms / tau_plus)
    if dt_ms < 0:
        return -a_minus * math.exp(dt_ms / tau_minus)
    return 0.0


class StdPSynapse:
    """A single plastic synapse with pair-based STDP and bounded weight."""

    def __init__(self, weight: float = 0.5, w_max: float = 1.0,
                 a_plus: float = 0.01, a_minus: float = 0.01,
                 tau_plus: float = 20.0, tau_minus: float = 20.0):
        if not 0.0 <= weight <= w_max:
            raise ValueError(f"weight {weight} outside [0, {w_max}]")
        if w_max <= 0:
            raise ValueError("w_max must be positive")
        self.weight = float(weight)
        self.w_max = float(w_max)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.tau_plus = float(tau_plus)
        self.tau_minus = float(tau_minus)
        self.last_pre_ms: float | None = None
        self.last_post_ms: float | None = None

    # -- low-level update -------------------------------------------------
    def apply_pair(self, dt_ms: float) -> float:
        """Apply one pre/post pair with timing difference dt_ms.

        Returns the weight change applied (after bounding).
        """
        dw = stdp_update(dt_ms, self.a_plus, self.a_minus,
                         self.tau_plus, self.tau_minus)
        new_w = min(self.w_max, max(0.0, self.weight + dw))
        applied = new_w - self.weight
        self.weight = new_w
        return applied

    # -- spike-driven update (nearest-neighbour pairing) ------------------
    def pre_spike(self, t_ms: float) -> float:
        """Record a presynaptic spike; pair it with the last post spike.

        Returns the weight change applied (0.0 if no post spike yet).
        """
        dw = 0.0
        if self.last_post_ms is not None:
            # dt = t_post - t_pre is negative here (post came first).
            dw = self.apply_pair(self.last_post_ms - t_ms)
        self.last_pre_ms = float(t_ms)
        return dw

    def post_spike(self, t_ms: float) -> float:
        """Record a postsynaptic spike; pair it with the last pre spike.

        Returns the weight change applied (0.0 if no pre spike yet).
        """
        dw = 0.0
        if self.last_pre_ms is not None:
            dw = self.apply_pair(t_ms - self.last_pre_ms)
        self.last_post_ms = float(t_ms)
        return dw
