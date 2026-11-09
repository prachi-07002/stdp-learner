# stdp-learner

Spike-timing-dependent plasticity (STDP) in pure Python + NumPy: synaptic
weights that learn from the *timing* of spikes, not from backprop — the
learning rule a low-power analog spiking-neuron board actually needs.

Built as the software prototype for a Spring 2027 Honors College research
project (Prof. Walter Daniel Leon-Salas): a coin-cell-powered analog board
emulating a spiking neuron, with potentiometers setting synaptic weights
manually. This repo answers the question that board has to answer first —
**given pre/post spike timing, how should a weight move, and by how much?**
The STDP rule here is the specification the analog circuit (and its
potentiometer-set weights) will implement and be tested against.

## What it does

- **STDP rule** (`stdp.py`) — classic pair-based STDP (Bi & Poo, 1998):
  `dt = t_post - t_pre`; pre-before-post (`dt > 0`) potentiates,
  `dw = A_plus * exp(-dt / tau_plus)`; post-before-pre depresses,
  `dw = -A_minus * exp(dt / tau_minus)`. Defaults `A_plus = A_minus = 0.01`,
  `tau_plus = tau_minus = 20 ms`, weights hard-bounded to `[0, w_max = 1.0]`
  (the finite conductance range of a real synapse).
- **LIF neuron** (`neuron.py`) — leaky integrate-and-fire postsynaptic
  neuron (Euler integration, refractory period).
- **Experiments** (`simulation.py`, run by `demo.py`):
  1. *Pairing protocol* — repeat identical pre/post pairs at fixed `dt`.
  2. *Pattern learning* — two inputs compete; the one that consistently
     fires just before the postsynaptic spike wins, the one that fires
     just after it loses, so the synapse population becomes selective for
     a repeated timing pattern.
  3. *LIF closed-loop check* — a weak synapse cannot make the postsynaptic
     neuron fire; a potentiated one can.

## How to run

```bash
python3 demo.py                 # runs all experiments, prints headline numbers
python3 -m unittest discover -v # runs the test suite
```

Requires Python 3.10+ and NumPy; matplotlib is used only for the plot in
`demo.py` (the CSV is always written). Runtime is ~1 s.

`demo.py` writes `results/weight_evolution.csv` (per-trial weights for
every experiment) and `results/weight_evolution.png` (weight-vs-trial
plot, regenerated on each run). All shipped files are plain text.

## Headline results (measured by running `demo.py`)

| Experiment | Result |
|---|---|
| LTP, 60 pairs at dt = +10 ms | weight **0.500 → 0.864** (+0.364; +0.0061 per pair) |
| LTD, 60 pairs at dt = −10 ms | weight **0.500 → 0.136** (−0.364) |
| Pattern learning, 40 trials | causal synapse **0.500 → 0.743**, anti-causal **0.500 → 0.257**; selectivity (w_A − w_B) = **0.485** |
| LIF, weak synapse w = 0.3 | **0** postsynaptic spikes in 300 ms |
| LIF, potentiated synapse w = 0.9 | **3** postsynaptic spikes, at 1.1 / 101.2 / 201.2 ms — ~1 ms after each presynaptic spike |

So learning changes behaviour, not just a number: the same presynaptic
drive is subthreshold before STDP and reliably fires the postsynaptic
neuron after it.

## Files

| File | Contents |
|---|---|
| `stdp.py` | `stdp_update()` rule and `StdPSynapse` (bounded weight, event-driven pairing) |
| `neuron.py` | `LIFNeuron` |
| `simulation.py` | `pairing_protocol()`, `pattern_learning()`, `lif_response()` |
| `demo.py` | Experiment runner; writes `results/weight_evolution.csv` (+ `.png` plot) |
| `test_stdp.py` | 12 unit tests (see below) |
| `README.md` | This file |

## Tests — 12 tests, all passing

`python3 -m unittest discover -v` → **12 passed, 0 failed** (runs in <0.1 s).
Covered: LTP direction (small dt > 0 → weight increases), LTD direction
(small dt < 0 → weight decreases), magnitude decays exponentially with
|dt| (exact values checked), no change for large |dt| (200 ms) or dt = 0,
weight stays bounded in [0, w_max] under 500 repeated LTP/LTD pairings,
repeated pairings move the weight monotonically in the expected direction,
event-driven pairing through the spike API, pattern-learning selectivity
(causal up, anti-causal down, selectivity > 0.3 and monotonic), and the
LIF check (no input → no spike; weak synapse → no spike; strong synapse →
spikes ~1 ms after each presynaptic spike).

## Research-application talking point

> Implemented spike-timing-dependent plasticity from scratch in Python/NumPy — no backprop, no framework — showing a synapse potentiating from 0.50 to 0.86 over 60 causal spike pairings and a two-input network becoming selective for a repeated timing pattern (selectivity 0.49 after 40 trials), as the learning-rule specification for a coin-cell-powered analog spiking-neuron board.

## Hardware mapping (for the analog board)

| Software concept | Board analogue |
|---|---|
| Synaptic weight `w ∈ [0, 1]` | Potentiometer-set conductance |
| STDP window (`A±`, `τ± = 20 ms`) | Timing window the coincidence-detection circuit must realise |
| LTP / LTD from spike order | Charge/discharge direction set by which spike arrives first |
| Pattern selectivity (0.49 after 40 trials) | The measurable prediction: the causal input's potentiometer-equivalent conductance should end up ~3× the anti-causal one's (0.74 vs 0.26) |
