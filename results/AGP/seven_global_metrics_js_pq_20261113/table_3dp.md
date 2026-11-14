# Four-generator AGP seven-metric summary with JS(P,Q)

| Generator | FD ↓ | PRD max F1 ↑ | Prevalence MAE ↓ | Richness KS ↓ | Shannon KS ↓ | JS(P,Q), nats ↓ | Spearman correlation RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mixture + ICFM | 0.082 | 0.631 ± 0.025 | 0.011 | 0.064 | 0.182 | 0.216 ± 0.026 | 0.075 |
| NADE + ICFM | 0.066 | 0.739 ± 0.018 | 0.010 | 0.028 | 0.116 | 0.078 ± 0.006 | 0.090 |
| MIDASim mask + ICFM | 0.070 | 0.670 ± 0.019 | 0.011 | 0.100 | 0.104 | 0.225 ± 0.011 | 0.081 |
| Full MIDASim | 0.048 | 0.770 ± 0.011 | 0.011 | 0.100 | 0.040 | 0.329 ± 0.065 | 0.089 |

All metrics use the original full test cohorts: 2,002 P and 2,089 Q per generator. FD and PRD use the original square-root abundance features. All six non-JS metrics are copied with their original CSV cell strings. PRD shows the original five-quantizer mean ± sample SD; JS shows mean ± sample SD over the five selected RDR fits (denominator 4).

`JS_PQ_hat = 0.5 mean_P(log(r_hat)) + 0.5 mean_Q(log(2-r_hat))`, in nats, with equal source weights despite unequal counts. Scaled checkpoint logits `t` use `log(r_hat)=log(2)-softplus(-t)` and `log(2-r_hat)=log(2)-softplus(t)`. Every frozen fit is evaluated, including chi-square-trained NADE. No training, reselection, transform fitting, clipping, subsampling, or truncation of negative estimates occurs.

The checkpoint replay recovers the one exact endpoint in archived Full MIDASim repeat 4 (P score rounded to 2). Float64 log evaluation uses the original float32 network outputs and saved ILR transforms. Per-fit CSV records replay agreement and the difference from direct evaluation of archived RDR scores. Original JS(P,M) files remain unchanged.

| Generator | Old JS(P,M) mean | New JS(P,Q) mean | Difference |
|---|---:|---:|---:|
| Mixture + ICFM | 0.066 | 0.216 | 0.150 |
| NADE + ICFM | 0.020 | 0.078 | 0.057 |
| MIDASim mask + ICFM | 0.081 | 0.225 | 0.143 |
| Full MIDASim | 0.109 | 0.329 | 0.220 |

These are fitted held-out variational estimates, not exact population divergences. Initialization SD on fixed observations is not a sampling confidence interval. The assessment is retrospective; finite-sample estimates need not be population lower bounds.

Reproduce from the private HPC archive into a fresh destination:

```bash
python -B experiments/AGP/seven_global_metrics_pq.py \
  --output results/AGP/seven_global_metrics_js_pq_reproduction
```

Check saved results without private artifacts:

```bash
python -B experiments/AGP/seven_global_metrics_pq.py \
  --verify results/AGP/seven_global_metrics_js_pq_20261113
```

`--verify-archives` additionally rehashes private inputs; full reproduction replays the checkpoints. Private arrays, row identifiers, logits and checkpoints are not copied into this output package.
