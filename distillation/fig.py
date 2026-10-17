#!/usr/bin/env python3
"""Cost figure of the paper (distillation_cost.pdf): cost of the cheapest sequence of at most five rounds
reaching delta_out <= 10^-x, from the seq_<delta>.pkl files written by seq.py. Prints only; run seq.py first."""
import os, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9})
XMAX = 25
curves = [("PS", "Prakash–Saha codes", "tab:blue", "--", 2.0),
          ("PS+new", r"Prakash–Saha codes with $[[15,1,3]]_3$, $[[28,1,4]]_3$, $[[37,1,5]]_3$", "tab:red", "-", 1.2),
          ("qubit", "qubit codes: 15-to-1, 10-to-2, Bravyi–Haah, $[[49,1,5]]$", "tab:orange", "-.", 1.0)]
fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
for ax, d0 in zip(axs, ("0.02", "0.1", "0.2")):
    out = pickle.load(open(os.path.join(HERE, f"seq_{d0}.pkl"), "rb"))
    for key, lab, col, ls, lw in curves:
        xs = [x for x in range(1, XMAX + 1) if out[key][x]]
        ax.step(xs, [out[key][x][1] for x in xs], where='mid', color=col, ls=ls, lw=lw, label=lab)
    ax.set_yscale('log'); ax.set_xlim(0, XMAX + 0.5)
    ax.set_title(fr"$\delta_{{\rm in}}={d0}$"); ax.set_xlabel(r"target $\delta_{\rm out}=10^{-x}$")
    ax.grid(alpha=.3)
axs[0].set_ylabel(r"cost $C$")
axs[0].text(0.97, 0.05, "dashed and solid\ncurves coincide", transform=axs[0].transAxes,
            ha='right', va='bottom', fontsize=7.5)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=8.5)
plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig(os.path.join(HERE, 'distillation_cost.pdf'))
print('wrote distillation_cost.pdf')
