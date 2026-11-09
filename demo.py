"""Demo / experiment runner for stdp-learner.

Runs three experiments and prints headline numbers:
  1. Pairing protocol: 60 pre-before-post pairs (dt=+10 ms) -> LTP,
     then 60 post-before-pre pairs (dt=-10 ms) -> LTD.
  2. Pattern learning: causal vs anti-causal synapse over 40 trials.
  3. LIF check: weak (w=0.3) vs potentiated (w=0.9) synapse driving a
     postsynaptic neuron.

Outputs (written next to this script):
  results/weight_evolution.csv  -- per-trial weights for all experiments
  results/weight_evolution.png  -- weight-vs-trial plot (matplotlib)

Run:  python3 demo.py
"""

from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import lif_response, pairing_protocol, pattern_learning  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
N_PAIRS = 60
N_TRIALS = 40


def main() -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Pairing protocol --------------------------------------------------
    ltp = pairing_protocol(10.0, n_pairs=N_PAIRS, w0=0.5)
    ltd = pairing_protocol(-10.0, n_pairs=N_PAIRS, w0=0.5)
    from stdp import stdp_update  # noqa: E402

    print(f"LTP:  w 0.500 -> {ltp[-1]:.3f} after {N_PAIRS} pairs at dt=+10 ms "
          f"({ltp[-1] - 0.5:+.3f}, {stdp_update(10.0):.4f} per pair)")
    print(f"LTD:  w 0.500 -> {ltd[-1]:.3f} after {N_PAIRS} pairs at dt=-10 ms "
          f"({ltd[-1] - 0.5:+.3f})")

    # 2. Pattern learning ---------------------------------------------------
    pat = pattern_learning(n_trials=N_TRIALS, w0=0.5)
    w_a, w_b = pat["w_causal"][-1], pat["w_anticausal"][-1]
    print(f"Pattern ({N_TRIALS} trials): causal synapse "
          f"0.500 -> {w_a:.3f}, anti-causal synapse 0.500 -> {w_b:.3f}, "
          f"selectivity (w_A - w_B) = {w_a - w_b:.3f}")

    # 3. LIF closed-loop check ---------------------------------------------
    weak_spikes = lif_response(0.3)
    strong_spikes = lif_response(0.9)
    print(f"LIF: weak synapse w=0.3 -> {len(weak_spikes)} post spikes; "
          f"potentiated synapse w=0.9 -> {len(strong_spikes)} post spikes "
          f"at {[round(s, 1) for s in strong_spikes]} ms")

    # CSV -------------------------------------------------------------------
    csv_path = os.path.join(RESULTS_DIR, "weight_evolution.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "ltp_w_dt_plus10", "ltd_w_dt_minus10",
                         "pattern_w_causal", "pattern_w_anticausal",
                         "pattern_selectivity"])
        for i in range(max(N_PAIRS, N_TRIALS) + 1):
            writer.writerow([
                i,
                ltp[i] if i <= N_PAIRS else "",
                ltd[i] if i <= N_PAIRS else "",
                pat["w_causal"][i] if i <= N_TRIALS else "",
                pat["w_anticausal"][i] if i <= N_TRIALS else "",
                pat["selectivity"][i] if i <= N_TRIALS else "",
            ])
    print(f"Saved {csv_path}")

    # Plot ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(ltp, label="dt = +10 ms (LTP)")
        axes[0].plot(ltd, label="dt = -10 ms (LTD)")
        axes[0].set_xlabel("pair #")
        axes[0].set_ylabel("synaptic weight")
        axes[0].set_title("Pairing protocol (w0 = 0.5)")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].plot(pat["w_causal"], label="causal input A")
        axes[1].plot(pat["w_anticausal"], label="anti-causal input B")
        axes[1].set_xlabel("trial #")
        axes[1].set_ylabel("synaptic weight")
        axes[1].set_title("Pattern learning (40 trials)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        png_path = os.path.join(RESULTS_DIR, "weight_evolution.png")
        fig.savefig(png_path, dpi=150)
        print(f"Saved {png_path}")
    except ImportError:
        print("matplotlib not available; skipping plot (CSV still saved)")

    return {"ltp_final": ltp[-1], "ltd_final": ltd[-1],
            "pattern_w_causal": w_a, "pattern_w_anticausal": w_b,
            "weak_spikes": len(weak_spikes),
            "strong_spikes": len(strong_spikes)}


if __name__ == "__main__":
    main()
