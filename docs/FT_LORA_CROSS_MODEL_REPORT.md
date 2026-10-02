# FT vs LoRA Cross-Model Report

This note summarizes the latest comparison across two model pairs:

- pythia70m_wikidata
- llama8b_wikiplus

Primary data sources:

- [results/mp_reservoir/ft_lora_dynamics/cross_model_ft_lora_mp_mem_table.csv](../results/mp_reservoir/ft_lora_dynamics/cross_model_ft_lora_mp_mem_table.csv)
- [results/mp_reservoir/ft_lora_dynamics/cross_model_matched_pass_deltas.csv](../results/mp_reservoir/ft_lora_dynamics/cross_model_matched_pass_deltas.csv)
- [results/mp_reservoir/ft_lora_dynamics/cross_model_interpolated_pass_deltas.csv](../results/mp_reservoir/ft_lora_dynamics/cross_model_interpolated_pass_deltas.csv)
- [results/mp_reservoir/ft_lora_dynamics/cross_model_auc_deltas.csv](../results/mp_reservoir/ft_lora_dynamics/cross_model_auc_deltas.csv)

## 1) MP Questions

Question: Does FT show a distinct MP signature relative to LoRA?

Findings:

1. Pythia-70M: yes, clearly.
2. Llama-8B: weaker and mixed.

Evidence (AUC deltas, FT - LoRA):

| Pair | Signal Fraction | Informative Features |
|---|---:|---:|
| pythia70m_wikidata | +0.0132 | +6.75 |
| llama8b_wikiplus | +0.0013 | +5.23 |

Interpretation:

1. On Pythia, FT shows a clear MP advantage, including the mid-run informative-feature spike.
2. On Llama-8B, FT is not uniformly above LoRA in MP at every pass endpoint, even though integrated deltas are slightly positive.

## 2) Memorization Questions

Question: Does FT produce stronger memorization dynamics than LoRA?

Findings:

1. Yes, for both model pairs.
2. The effect is much larger on Llama-8B than on Pythia-70M.

Evidence (AUC deltas, FT - LoRA):

| Pair | Train NLL | Exposure Delta | Avg Tokens Recovered |
|---|---:|---:|---:|
| pythia70m_wikidata | -0.2728 | +1.1830 | +0.0158 |
| llama8b_wikiplus | -5.2143 | +5.4520 | +0.8276 |

Endpoint-at-overlap evidence:

1. Llama-8B at pass 5.0: FT has much lower NLL, higher exposure delta, and much higher recovered tokens.
2. Pythia-70M at pass 1.0: FT also has lower NLL, higher exposure delta, and higher recovered tokens.

## 3) Salient Points

1. Memorization results are more consistent across scales than MP shape details.
2. MP metrics are informative, but not a single universal monotonic proxy for memorization across all regimes.
3. The strongest reproducible signal in this dataset is FT > LoRA on memorization pressure/readout metrics.
4. The Llama-8B FT rerun4 is now stable and usable (checkpoints 130, 260, 390, 520, 650, 685 with MEM+MP artifacts).

## Figures

### Cross-Model Pair Trajectories

![Cross-model trajectories](../results/mp_reservoir/ft_lora_dynamics/plots/cross_model_pair_trajectories.png)

### Cross-Model Interpolated FT-LoRA Deltas

![Cross-model interpolated deltas](../results/mp_reservoir/ft_lora_dynamics/plots/cross_model_interpolated_deltas.png)
