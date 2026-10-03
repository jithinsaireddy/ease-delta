# EASE-Delta: results

Generated 2026-10-03 08:23 UTC by `scripts/make_report.py` from `runs/results/*.json`. No figure in this document was typed by hand.

Pre-registration: `docs/PREREGISTRATION.md`, SHA-256 `babc7259667e922ff9edce371c64d0905f15f4e38568cb99642b383355bb0af3`, recorded 2026-09-29T21:24:22Z before any test split was evaluated.

Machine: macOS-27.2-arm64-arm-64bit, Apple M4 Max, 48 GB, 40-core GPU.

## 1. Verdicts

Decision rules are those written down in advance. "Not supported" means the rule was not met; it is a result, and it is reported as prominently as the others.

| | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | Incremental equals full rebuild | **supported** | 0 of 195,434 cached nodes differed from a full rebuild over 3,559 events (canonical mode) |
| H2 | Update cost at most half | **supported** | STANDARD tokens vs E-full: ratio 0.058 [0.056, 0.060]; STANDARD tokens vs B2: ratio 0.065 [0.062, 0.068]; WIDE tokens vs E-full: ratio 0.020 [0.019, 0.020]; WIDE tokens vs B2: ratio 0.023 [0.022, 0.025]; STANDARD time E / E-full: 0.088 [0.080, 0.096]; STANDARD time E / B2: 0.140 [0.128, 0.153]; WIDE time E / E-full: 0.038 [0.034, 0.042]; WIDE time E / B2: 0.039 [0.034, 0.045] |
| H3 | Revision fidelity vs dense reader (seen structure) | **supported** | stale-decision rate E 13.06% vs B2 16.99%, difference -3.93 pts [-6.56 pts, -1.36 pts]; disposition accuracy E 90.49% vs B2 86.94%, difference +3.55 pts [+2.11 pts, +5.13 pts] |
| H4 | Revision fidelity vs dense reader (unseen structure) | **not supported** | stale-decision rate E 15.29% vs B2 18.82%, difference -3.53 pts [-7.96 pts, +0.85 pts]; disposition accuracy E 85.98% vs B2 80.63%, difference +5.35 pts [+2.00 pts, +8.71 pts] |
| H5 | Learned refinement beats calibrated rules | **supported** | STANDARD: status NLL E 0.2604 vs B1-soft 0.2791, difference -0.0187 [-0.0264, -0.0118]; WIDE: status NLL E 0.3512 vs B1-soft 0.4032, difference -0.0521 [-0.0696, -0.0356] |
| H6 | Executing the policy generalises better than learning it | **not supported** | WIDE disposition accuracy N 86.19% vs E 85.70%, difference (N - E) +0.50 pts [-0.60 pts, +1.58 pts] |
| H7 | Improves from feedback, safely | **not supported** | stream snli.test, second half: evolving 71.76% vs frozen 71.76%, difference +0.00 pts [+0.00 pts, +0.00 pts]; largest regression drop after an adoption 0.00% (truthful), 0.00% (half the labels wrong) |
| H8 | Dependency proposals within the miss bound | **not supported** | STANDARD: missed links 4.56% [4.02%, 5.15%], work reduced x5.48, accuracy change -2.31 pts; WIDE: missed links 5.11% [4.36%, 5.87%], work reduced x11.22, accuracy change +0.99 pts |
| H9 | Propagated conflicts are harder (exploratory) | **supported** | detection at 5.0% false positives: direct conflicts (T1) 77.0% [71.0%, 82.5%], propagated (T2) 34.5% [28.0%, 41.0%] |

**Architecture claim (pre-registration, section 6): supported on one test set and not the other.** See section 3.

## 2. The edge model (Stage A)

- Parameters: 149,113,347. Backbone `answerdotai/ModernBERT-base`, fp32, MPS.
- Trained on 900,000 examples streamed from the Hugging Face Hub in 152.4 minutes (98.5 examples/s). Fixed budget, final weights, no early stopping.
- Examples by source: mnli 243,178, unrelated 179,460, vitaminc 359,978, wanli 117,384.
- Licences: mnli: cc-by-3.0 / cc-by-sa-3.0 / mit / other (per genre); unrelated: derived: texts drawn from the real sources above; vitaminc: cc-by-sa-3.0; wanli: cc-by-4.0.
- Stream reconnections during training: {'mnli': 0, 'vitaminc': 0, 'wanli': 0}. Passes over each corpus: {'mnli': 0, 'vitaminc': 0, 'wanli': 1}.

Temperature 1.2268, fitted on vitaminc.dev, mnli.dev, unrelated.dev only.

| Set | Role | n | Accuracy | Macro-F1 | NLL raw -> calibrated | ECE raw -> calibrated | Public NLI model (B3) accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| anli_r1.test | test | 1,000 | 46.40% | 0.463 | 1.403 -> 1.264 | 0.285 -> 0.241 | 63.40% |
| anli_r2.test | test | 1,000 | 33.30% | 0.331 | 1.761 -> 1.556 | 0.416 -> 0.372 | 48.20% |
| anli_r3.test | test | 1,200 | 34.75% | 0.339 | 1.770 -> 1.563 | 0.400 -> 0.359 | 42.25% |
| mnli.dev | dev | 5,000 | 88.58% | 0.885 | 0.304 -> 0.312 | 0.012 -> 0.030 | 87.42% |
| mnli_mm.test | test | 9,832 | 88.56% | 0.885 | 0.305 -> 0.312 | 0.011 -> 0.027 | 89.84% |
| snli.dev | dev | 3,000 | 80.87% | 0.810 | 0.515 -> 0.502 | 0.043 -> 0.026 | 89.60% |
| snli.test | test | 9,824 | 80.33% | 0.804 | 0.518 -> 0.505 | 0.044 -> 0.012 | 89.15% |
| unrelated.dev | dev | 3,000 | 99.57% | 0.333 | 0.011 -> 0.015 | 0.003 -> 0.007 | 60.23% |
| unrelated.test | test | 6,000 | 99.18% | 0.332 | 0.023 -> 0.025 | 0.003 -> 0.008 | 54.93% |
| vitaminc.dev | dev | 6,001 | 85.50% | 0.816 | 0.457 -> 0.428 | 0.059 -> 0.031 | 64.29% |
| vitaminc.test | test | 55,197 | 90.20% | 0.869 | 0.271 -> 0.277 | 0.007 -> 0.024 | 68.30% |
| wanli.test | test | 5,000 | 74.74% | 0.729 | 0.617 -> 0.597 | 0.065 -> 0.030 | 66.34% |

**Revision sensitivity.** On VitaminC test, 75.99% of 16,361 contrast sets were answered entirely correctly. A contrast set is the same claims judged against the evidence before and after a real edit; one wrong member fails the set.

**Unrelated records.** 0.82% of 6,000 unrelated pairs were read as decisive. At that rate the chance that none of 10 unrelated records is misread is 92.1%, and of 30, 78.2%. The public NLI model read 45.07% as decisive.

**Numbers.** Accuracy by the form of the claim. A claim counts as a comparison if it contains a digit and a comparative phrase.

| Set | Claim | n | Accuracy |
|---|---|---:|---:|
| vitaminc.dev | no number | 2,028 | 83.38% [81.76%, 85.06%] |
| vitaminc.dev | number with comparison | 2,992 | 86.56% [85.36%, 87.80%] |
| vitaminc.dev | number, no comparison | 981 | 86.65% [84.51%, 88.79%] |
| vitaminc.test | no number | 30,215 | 89.27% [88.91%, 89.62%] |
| vitaminc.test | number with comparison | 14,750 | 92.33% [91.93%, 92.75%] |
| vitaminc.test | number, no comparison | 10,232 | 89.87% [89.27%, 90.46%] |

ANLI is adversarial and was never trained on (its licence is non-commercial). SNLI was excluded from training because its labels assume both sentences describe one scene. Lower accuracy on both is expected and is shown rather than omitted.

## 3. Evolving episodes

All systems share the ledger, schema, exact logic and planner. Intervals are 95% bootstrap intervals over whole episodes.

| Id | System |
|---|---|
| E-seedN | EASE-Delta: edge model, refiner trained through the exact policy (three training seeds) |
| E-full | the same model re-reading every active record against every predicate at each change |
| E-declared | E, but each record is compared only with the predicate it was written for. This uses information a deployment would have only if someone declared it, so it is an upper bound on what linking can save, not a system on equal terms |
| B1-soft | edge model, probabilities calibrated through the policy, no refiner |
| B1-hard | edge model, top label per edge, policy applied to labels |
| N-seedN | edge model, GRU that outputs the status itself; the policy is learned, not executed |
| B2 | dense reader: same backbone and initial weights, reads the claim against all active records at once |
| B3 | public NLI model `tasksource/ModernBERT-base-nli` in place of the edge model, calibrated through the policy on the same validation episodes as B1-soft |
| B3-uncalibrated | the same without calibration |

### STANDARD

1,000 episodes, 9,986 scored events, 5.5 predicates and 11.3 records per task on average. 2,090 events changed some action's correct disposition. Ledger outcomes: {'applied': 7747, 'conflict': 316, 'duplicate': 1208, 'stale': 715}.

| System | Disposition accuracy | Stale-decision rate | Spurious-change rate | False-READY rate | Status accuracy | Status NLL | P(success) Brier | Tokens per changed event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1-hard | 89.80% [88.70%, 90.93%] | 12.40% [10.63%, 14.12%] | 1.13% [0.94%, 1.35%] | 3.31% [2.56%, 4.10%] | 89.91% [89.18%, 90.66%] | 2.788 [2.580, 2.990] | 0.1259 [0.1180, 0.1342] | 146 [141, 151] |
| B1-soft | 89.52% [88.46%, 90.60%] | 13.05% [11.30%, 14.84%] | 1.19% [0.99%, 1.40%] | 2.97% [2.27%, 3.72%] | 89.97% [89.24%, 90.68%] | 0.279 [0.260, 0.297] | 0.1162 [0.1095, 0.1229] | 146 [141, 151] |
| B2 | 86.94% [85.15%, 88.75%] | 16.99% [13.82%, 20.16%] | 3.91% [3.22%, 4.62%] | 3.93% [2.77%, 5.27%] | 88.12% [86.94%, 89.31%] | 0.429 [0.383, 0.478] | 0.1184 [0.1075, 0.1298] | 2234 [2117, 2360] |
| B3 | 70.55% [68.91%, 72.14%] | 41.96% [39.75%, 44.12%] | 2.61% [2.33%, 2.94%] | 5.74% [4.74%, 6.81%] | 69.30% [68.21%, 70.42%] | 0.788 [0.766, 0.812] | 0.1580 [0.1504, 0.1657] | 146 [141, 151] |
| B3-uncalibrated | 43.71% [41.91%, 45.59%] | 56.13% [53.96%, 58.27%] | 2.29% [1.96%, 2.65%] | 9.27% [7.96%, 10.67%] | 32.81% [31.74%, 33.93%] | 3.804 [3.667, 3.940] | 0.2774 [0.2637, 0.2917] | 146 [141, 151] |
| E-declared | 87.03% [85.71%, 88.33%] | 14.92% [13.06%, 16.85%] | 1.26% [1.03%, 1.52%] | 1.54% [1.09%, 2.08%] | 86.13% [85.19%, 87.02%] | 0.353 [0.330, 0.377] | 0.1223 [0.1155, 0.1291] | 18 [17, 18] |
| E-full | same as E (H1) | | | | | | | 2523 [2431, 2611] |
| E-seed1 | 90.48% [89.40%, 91.51%] | 12.63% [10.94%, 14.40%] | 1.15% [0.95%, 1.35%] | 3.49% [2.69%, 4.33%] | 90.54% [89.81%, 91.24%] | 0.260 [0.242, 0.279] | 0.1150 [0.1084, 0.1217] | 146 [141, 151] |
| E-seed2 | 90.01% [88.96%, 91.08%] | 12.35% [10.70%, 14.12%] | 1.28% [1.06%, 1.49%] | 3.05% [2.34%, 3.82%] | 90.53% [89.79%, 91.25%] | 0.265 [0.247, 0.283] | 0.1153 [0.1089, 0.1220] | 146 [141, 151] |
| E-seed3 | 90.45% [89.40%, 91.44%] | 12.68% [11.02%, 14.41%] | 1.23% [1.03%, 1.44%] | 3.34% [2.58%, 4.08%] | 90.62% [89.91%, 91.32%] | 0.271 [0.251, 0.291] | 0.1150 [0.1084, 0.1217] | 146 [141, 151] |
| N-seed1 | 89.95% [88.83%, 90.94%] | 12.59% [10.87%, 14.29%] | 1.67% [1.40%, 1.95%] | 3.31% [2.57%, 4.05%] | 90.17% [89.42%, 90.89%] | 0.277 [0.256, 0.298] | 0.1158 [0.1092, 0.1225] | 146 [141, 151] |
| N-seed2 | 89.97% [88.91%, 91.02%] | 13.01% [11.26%, 14.77%] | 1.62% [1.38%, 1.87%] | 3.23% [2.49%, 3.97%] | 90.01% [89.31%, 90.73%] | 0.274 [0.253, 0.294] | 0.1155 [0.1090, 0.1221] | 146 [141, 151] |
| N-seed3 | 90.22% [89.16%, 91.24%] | 12.68% [10.93%, 14.42%] | 1.50% [1.27%, 1.74%] | 3.15% [2.41%, 3.87%] | 90.23% [89.49%, 90.96%] | 0.274 [0.254, 0.293] | 0.1156 [0.1090, 0.1224] | 146 [141, 151] |
| B1-soft@B2 | 89.17% [87.29%, 91.13%] | 13.41% [10.23%, 16.74%] | 1.14% [0.81%, 1.47%] | 2.60% [1.58%, 3.72%] | 90.47% [89.41%, 91.53%] | 0.268 [0.241, 0.295] | 0.1124 [0.1021, 0.1226] | 145 [137, 153] |
| E-full@B2 | same as E (H1) | | | | | | | 2538 [2401, 2688] |
| E-seed1@B2 | 90.49% [88.68%, 92.36%] | 13.06% [9.98%, 16.14%] | 1.05% [0.74%, 1.40%] | 3.20% [1.92%, 4.66%] | 90.93% [89.89%, 91.95%] | 0.246 [0.220, 0.272] | 0.1103 [0.1004, 0.1207] | 145 [137, 153] |
| E-seed2@B2 | 89.85% [88.00%, 91.75%] | 13.06% [10.09%, 16.12%] | 1.09% [0.78%, 1.42%] | 2.60% [1.53%, 3.77%] | 91.24% [90.19%, 92.27%] | 0.248 [0.223, 0.272] | 0.1102 [0.1004, 0.1203] | 145 [137, 153] |
| E-seed3@B2 | 90.53% [88.81%, 92.21%] | 12.95% [10.19%, 15.88%] | 1.08% [0.78%, 1.40%] | 3.08% [1.90%, 4.38%] | 91.11% [90.04%, 92.17%] | 0.253 [0.225, 0.280] | 0.1097 [0.0997, 0.1200] | 145 [137, 153] |

Rows ending in `@B2` and the row `B2` use the first 400 episodes of this set, because the dense reader is expensive to evaluate (see Deviations). Comparisons with B2 are paired on those episodes.

Dense reader context length: mean 310 tokens, median 297, 95th percentile 598, maximum 834; 0.00% truncated.

Paired comparisons (first minus second):

| Comparison | Disposition accuracy | Stale-decision rate | False-READY rate | Status NLL | Tokens ratio |
|---|---:|---:|---:|---:|---:|
| B1-soft@B2 vs B2 | +2.23 pts [+0.57 pts, +3.92 pts] | -3.58 pts [-6.18 pts, -0.99 pts] | -1.33 pts [-2.21 pts, -0.46 pts] | -0.1616 [-0.1981, -0.1286] | 0.065 [0.062, 0.068] |
| E-full@B2 vs B2 | n/a | n/a | n/a | n/a | 1.136 [1.130, 1.142] |
| E-seed1 vs B1-hard | +0.68 pts [+0.06 pts, +1.30 pts] | +0.23 pts [-0.52 pts, +0.96 pts] | +0.18 pts [-0.14 pts, +0.56 pts] | -2.5279 [-2.7152, -2.3358] | 1.000 [1.000, 1.000] |
| E-seed1 vs B1-soft | +0.96 pts [+0.30 pts, +1.61 pts] | -0.42 pts [-1.30 pts, +0.44 pts] | +0.52 pts [+0.17 pts, +0.93 pts] | -0.0187 [-0.0264, -0.0118] | 1.000 [1.000, 1.000] |
| E-seed1 vs B3 | +19.93 pts [+18.23 pts, +21.69 pts] | -29.32 pts [-31.92 pts, -26.72 pts] | -2.25 pts [-3.41 pts, -1.17 pts] | -0.5276 [-0.5515, -0.5025] | 1.000 [1.000, 1.000] |
| E-seed1 vs E-declared | +3.45 pts [+2.41 pts, +4.61 pts] | -2.28 pts [-3.65 pts, -0.93 pts] | +1.96 pts [+1.35 pts, +2.61 pts] | -0.0929 [-0.1086, -0.0770] | 8.174 [7.970, 8.389] |
| E-seed1 vs E-full | n/a | n/a | n/a | n/a | 0.058 [0.056, 0.060] |
| E-seed1@B2 vs B2 | +3.55 pts [+2.11 pts, +5.13 pts] | -3.93 pts [-6.56 pts, -1.36 pts] | -0.73 pts [-1.62 pts, +0.21 pts] | -0.1830 [-0.2171, -0.1515] | 0.065 [0.062, 0.068] |
| E-seed2 vs B1-soft | +0.49 pts [-0.21 pts, +1.19 pts] | -0.70 pts [-1.76 pts, +0.37 pts] | +0.08 pts [-0.21 pts, +0.42 pts] | -0.0144 [-0.0226, -0.0065] | 1.000 [1.000, 1.000] |
| E-seed2@B2 vs B2 | +2.91 pts [+1.51 pts, +4.43 pts] | -3.93 pts [-6.52 pts, -1.37 pts] | -1.33 pts [-2.33 pts, -0.41 pts] | -0.1817 [-0.2159, -0.1487] | 0.065 [0.062, 0.068] |
| E-seed3 vs B1-soft | +0.93 pts [+0.23 pts, +1.63 pts] | -0.37 pts [-1.35 pts, +0.64 pts] | +0.37 pts [-0.06 pts, +0.83 pts] | -0.0083 [-0.0168, +0.0001] | 1.000 [1.000, 1.000] |
| E-seed3@B2 vs B2 | +3.58 pts [+2.10 pts, +5.14 pts] | -4.05 pts [-6.62 pts, -1.39 pts] | -0.85 pts [-1.76 pts, +0.10 pts] | -0.1763 [-0.2106, -0.1442] | 0.065 [0.062, 0.068] |
| N-seed1 vs E-seed1 | -0.53 pts [-1.06 pts, -0.01 pts] | -0.05 pts [-0.84 pts, +0.71 pts] | -0.18 pts [-0.53 pts, +0.15 pts] | +0.0164 [+0.0101, +0.0230] | 1.000 [1.000, 1.000] |
| N-seed2 vs E-seed2 | -0.05 pts [-0.65 pts, +0.54 pts] | +0.65 pts [-0.23 pts, +1.52 pts] | +0.18 pts [-0.12 pts, +0.48 pts] | +0.0089 [+0.0023, +0.0154] | 1.000 [1.000, 1.000] |
| N-seed3 vs E-seed3 | -0.23 pts [-0.81 pts, +0.39 pts] | +0.00 pts [-0.89 pts, +0.89 pts] | -0.19 pts [-0.57 pts, +0.19 pts] | +0.0029 [-0.0041, +0.0092] | 1.000 [1.000, 1.000] |

Disposition accuracy by the kind of event that preceded it:

| Event | n | E-seed1 | B1-soft | N-seed1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|
| add | 1,806 | 90.8% | 89.4% | 90.1% | 85.1% (n=693) | 70.3% |
| conflict_same_rev | 618 | 92.9% | 92.1% | 92.6% | 92.0% (n=238) | 73.5% |
| distractor | 2,682 | 89.8% | 89.3% | 89.5% | 86.1% (n=980) | 69.9% |
| duplicate | 2,425 | 89.9% | 88.5% | 88.7% | 84.3% (n=961) | 70.0% |
| reinstate | 196 | 90.8% | 91.8% | 91.3% | 91.5% (n=82) | 74.0% |
| retract | 2,744 | 91.4% | 90.7% | 91.1% | 89.5% (n=1088) | 74.6% |
| revise | 6,065 | 90.2% | 89.3% | 89.8% | 87.5% (n=2344) | 69.1% |
| second_source | 1,870 | 89.8% | 88.7% | 89.3% | 84.4% (n=756) | 68.5% |
| stale | 1,450 | 91.7% | 90.3% | 90.6% | 88.2% (n=585) | 71.9% |

Confidence required before an action is called READY (exploratory; the pre-registered metrics above use 0.5):

| Threshold | False-READY rate | READY missed | Disposition accuracy |
|---:|---:|---:|---:|
| 0.5 | 3.49% [2.78%, 4.30%] | 7.41% [5.87%, 9.01%] | 90.48% [89.44%, 91.49%] |
| 0.7 | 1.79% [1.26%, 2.35%] | 12.08% [9.98%, 14.20%] | 90.04% [88.97%, 91.11%] |
| 0.8 | 1.33% [0.89%, 1.81%] | 15.25% [12.81%, 18.13%] | 89.46% [88.32%, 90.58%] |
| 0.9 | 0.83% [0.49%, 1.19%] | 23.44% [20.87%, 26.60%] | 87.66% [86.41%, 88.87%] |
| 0.95 | 0.35% [0.13%, 0.57%] | 33.14% [30.05%, 36.38%] | 85.47% [84.06%, 86.59%] |
| 0.99 | 0.12% [0.00%, 0.30%] | 58.88% [55.40%, 62.46%] | 79.01% [77.54%, 80.37%] |

Worst episodes for E: #775 (0% of 9 rows), #56 (0% of 14 rows), #986 (0% of 6 rows), #687 (11% of 9 rows), #917 (11% of 9 rows).

### WIDE

300 episodes, 5,281 scored events, 12.6 predicates and 34.4 records per task on average. 846 events changed some action's correct disposition. Ledger outcomes: {'applied': 4169, 'conflict': 169, 'duplicate': 555, 'stale': 388}.

| System | Disposition accuracy | Stale-decision rate | Spurious-change rate | False-READY rate | Status accuracy | Status NLL | P(success) Brier | Tokens per changed event |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1-hard | 84.69% [82.91%, 86.59%] | 15.56% [12.79%, 18.49%] | 0.73% [0.57%, 0.91%] | 4.84% [3.58%, 6.15%] | 86.58% [85.62%, 87.58%] | 3.709 [3.431, 3.974] | 0.1259 [0.1141, 0.1377] | 359 [344, 373] |
| B1-soft | 84.63% [82.69%, 86.59%] | 16.26% [13.49%, 19.26%] | 0.87% [0.69%, 1.09%] | 4.20% [3.07%, 5.35%] | 86.69% [85.72%, 87.67%] | 0.403 [0.373, 0.433] | 0.1130 [0.1033, 0.1226] | 359 [344, 373] |
| B2 | 80.63% [77.68%, 83.60%] | 18.82% [14.23%, 24.10%] | 4.91% [4.05%, 5.81%] | 6.26% [4.08%, 8.59%] | 80.99% [79.19%, 82.76%] | 0.636 [0.572, 0.702] | 0.1139 [0.0958, 0.1335] | 15847 [14867, 16842] |
| B3 | 64.32% [61.82%, 66.89%] | 46.81% [43.85%, 49.84%] | 0.96% [0.75%, 1.17%] | 13.23% [11.16%, 15.52%] | 52.88% [51.43%, 54.33%] | 1.236 [1.186, 1.287] | 0.1496 [0.1397, 0.1590] | 359 [344, 373] |
| B3-uncalibrated | 43.25% [40.38%, 46.16%] | 61.09% [58.52%, 63.87%] | 0.43% [0.31%, 0.55%] | 9.17% [7.36%, 11.04%] | 27.00% [25.76%, 28.32%] | 8.878 [8.557, 9.210] | 0.3030 [0.2812, 0.3240] | 359 [344, 373] |
| E-declared | 85.55% [83.69%, 87.61%] | 16.72% [13.82%, 19.67%] | 0.68% [0.53%, 0.83%] | 2.10% [1.35%, 2.94%] | 86.59% [85.58%, 87.64%] | 0.352 [0.324, 0.382] | 0.1093 [0.1000, 0.1184] | 22 [21, 23] |
| E-full | same as E (H1) | | | | | | | 18310 [17636, 18991] |
| E-seed1 | 85.70% [83.93%, 87.53%] | 14.52% [12.00%, 17.25%] | 0.80% [0.62%, 1.01%] | 5.01% [3.76%, 6.26%] | 86.89% [85.99%, 87.83%] | 0.351 [0.327, 0.376] | 0.1100 [0.1012, 0.1195] | 359 [344, 373] |
| E-seed2 | 85.31% [83.52%, 87.04%] | 15.45% [12.90%, 18.30%] | 0.96% [0.75%, 1.20%] | 4.76% [3.61%, 5.98%] | 86.89% [85.89%, 87.83%] | 0.352 [0.328, 0.378] | 0.1096 [0.1006, 0.1187] | 359 [344, 373] |
| E-seed3 | 77.80% [75.43%, 80.10%] | 24.39% [21.15%, 27.93%] | 1.87% [1.57%, 2.21%] | 5.83% [4.53%, 7.26%] | 78.98% [77.39%, 80.55%] | 0.666 [0.598, 0.740] | 0.1295 [0.1176, 0.1413] | 359 [344, 373] |
| N-seed1 | 86.19% [84.46%, 87.99%] | 16.26% [13.51%, 19.12%] | 1.00% [0.80%, 1.21%] | 4.63% [3.48%, 5.79%] | 87.00% [86.15%, 87.95%] | 0.372 [0.341, 0.405] | 0.1111 [0.1022, 0.1204] | 359 [344, 373] |
| N-seed2 | 86.19% [84.46%, 87.98%] | 16.38% [13.76%, 19.18%] | 1.16% [0.93%, 1.40%] | 4.94% [3.71%, 6.17%] | 86.71% [85.75%, 87.70%] | 0.379 [0.346, 0.411] | 0.1118 [0.1027, 0.1212] | 359 [344, 373] |
| N-seed3 | 85.58% [83.84%, 87.36%] | 16.03% [13.31%, 19.05%] | 1.32% [1.08%, 1.56%] | 5.01% [3.76%, 6.22%] | 86.64% [85.68%, 87.61%] | 0.381 [0.351, 0.412] | 0.1127 [0.1036, 0.1225] | 359 [344, 373] |
| B1-soft@B2 | 86.08% [82.61%, 89.44%] | 16.08% [11.55%, 20.63%] | 0.76% [0.46%, 1.08%] | 3.65% [1.99%, 5.58%] | 85.70% [83.86%, 87.52%] | 0.446 [0.388, 0.508] | 0.0955 [0.0807, 0.1108] | 370 [347, 394] |
| E-full@B2 | same as E (H1) | | | | | | | 18511 [17392, 19659] |
| E-seed1@B2 | 85.98% [82.53%, 89.13%] | 15.29% [10.90%, 20.19%] | 0.83% [0.53%, 1.15%] | 4.66% [2.73%, 6.89%] | 85.62% [83.96%, 87.21%] | 0.385 [0.338, 0.436] | 0.0948 [0.0807, 0.1101] | 370 [347, 394] |
| E-seed2@B2 | 85.39% [81.98%, 88.50%] | 16.86% [12.55%, 21.72%] | 0.97% [0.61%, 1.37%] | 4.15% [2.35%, 6.23%] | 85.47% [83.66%, 87.15%] | 0.389 [0.339, 0.450] | 0.0942 [0.0799, 0.1094] | 370 [347, 394] |
| E-seed3@B2 | 78.03% [73.92%, 81.99%] | 24.31% [18.87%, 30.30%] | 2.02% [1.44%, 2.69%] | 5.89% [3.65%, 8.21%] | 77.29% [74.77%, 79.78%] | 0.755 [0.608, 0.946] | 0.1058 [0.0877, 0.1253] | 370 [347, 394] |

Rows ending in `@B2` and the row `B2` use the first 100 episodes of this set, because the dense reader is expensive to evaluate (see Deviations). Comparisons with B2 are paired on those episodes.

Dense reader context length: mean 850 tokens, median 876, 95th percentile 1617, maximum 2141; 0.00% truncated.

Paired comparisons (first minus second):

| Comparison | Disposition accuracy | Stale-decision rate | False-READY rate | Status NLL | Tokens ratio |
|---|---:|---:|---:|---:|---:|
| B1-soft@B2 vs B2 | +5.45 pts [+2.25 pts, +8.67 pts] | -2.75 pts [-7.47 pts, +1.97 pts] | -2.61 pts [-4.66 pts, -0.53 pts] | -0.1907 [-0.2606, -0.1179] | 0.023 [0.022, 0.025] |
| E-full@B2 vs B2 | n/a | n/a | n/a | n/a | 1.168 [1.160, 1.176] |
| E-seed1 vs B1-hard | +1.00 pts [-0.05 pts, +2.10 pts] | -1.05 pts [-2.93 pts, +0.69 pts] | +0.17 pts [-0.34 pts, +0.72 pts] | -3.3579 [-3.6042, -3.1005] | 1.000 [1.000, 1.000] |
| E-seed1 vs B1-soft | +1.06 pts [-0.08 pts, +2.26 pts] | -1.74 pts [-3.72 pts, +0.00 pts] | +0.81 pts [+0.19 pts, +1.48 pts] | -0.0521 [-0.0696, -0.0356] | 1.000 [1.000, 1.000] |
| E-seed1 vs B3 | +21.38 pts [+18.76 pts, +23.95 pts] | -32.29 pts [-35.96 pts, -28.47 pts] | -8.22 pts [-10.57 pts, -5.99 pts] | -0.8847 [-0.9365, -0.8335] | 1.000 [1.000, 1.000] |
| E-seed1 vs E-declared | +0.15 pts [-1.51 pts, +1.83 pts] | -2.21 pts [-4.84 pts, +0.23 pts] | +2.91 pts [+1.98 pts, +3.90 pts] | -0.0012 [-0.0250, +0.0225] | 16.564 [16.109, 17.060] |
| E-seed1 vs E-full | n/a | n/a | n/a | n/a | 0.020 [0.019, 0.020] |
| E-seed1@B2 vs B2 | +5.35 pts [+2.00 pts, +8.71 pts] | -3.53 pts [-7.96 pts, +0.85 pts] | -1.60 pts [-3.71 pts, +0.61 pts] | -0.2510 [-0.3093, -0.1913] | 0.023 [0.022, 0.025] |
| E-seed2 vs B1-soft | +0.68 pts [-0.68 pts, +2.02 pts] | -0.81 pts [-2.78 pts, +1.18 pts] | +0.56 pts [-0.09 pts, +1.21 pts] | -0.0512 [-0.0721, -0.0307] | 1.000 [1.000, 1.000] |
| E-seed2@B2 vs B2 | +4.77 pts [+1.49 pts, +8.06 pts] | -1.96 pts [-5.93 pts, +2.01 pts] | -2.11 pts [-4.06 pts, -0.14 pts] | -0.2477 [-0.3065, -0.1888] | 0.023 [0.022, 0.025] |
| E-seed3 vs B1-soft | -6.83 pts [-8.77 pts, -4.98 pts] | +8.13 pts [+5.20 pts, +11.14 pts] | +1.63 pts [+0.63 pts, +2.68 pts] | +0.2630 [+0.2035, +0.3270] | 1.000 [1.000, 1.000] |
| E-seed3@B2 vs B2 | -2.60 pts [-6.46 pts, +1.37 pts] | +5.49 pts [-0.38 pts, +11.16 pts] | -0.37 pts [-2.40 pts, +1.71 pts] | +0.1184 [-0.0201, +0.2889] | 0.023 [0.022, 0.025] |
| N-seed1 vs E-seed1 | +0.50 pts [-0.60 pts, +1.58 pts] | +1.74 pts [-0.11 pts, +3.68 pts] | -0.38 pts [-1.02 pts, +0.25 pts] | +0.0213 [+0.0028, +0.0403] | 1.000 [1.000, 1.000] |
| N-seed2 vs E-seed2 | +0.88 pts [-0.36 pts, +2.05 pts] | +0.93 pts [-0.71 pts, +2.56 pts] | +0.18 pts [-0.50 pts, +0.92 pts] | +0.0268 [+0.0099, +0.0438] | 1.000 [1.000, 1.000] |
| N-seed3 vs E-seed3 | +7.78 pts [+5.78 pts, +9.71 pts] | -8.36 pts [-11.42 pts, -5.58 pts] | -0.82 pts [-2.01 pts, +0.41 pts] | -0.2851 [-0.3429, -0.2324] | 1.000 [1.000, 1.000] |

Disposition accuracy by the kind of event that preceded it:

| Event | n | E-seed1 | B1-soft | N-seed1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|
| add | 2,290 | 86.5% | 85.5% | 86.8% | 80.9% (n=810) | 66.2% |
| conflict_same_rev | 564 | 85.8% | 85.1% | 85.8% | 84.4% (n=179) | 64.7% |
| distractor | 2,184 | 84.3% | 83.7% | 84.6% | 79.5% (n=738) | 63.9% |
| duplicate | 1,878 | 85.6% | 85.4% | 86.3% | 79.0% (n=606) | 66.1% |
| reinstate | 156 | 82.1% | 80.1% | 85.9% | 82.1% (n=28) | 63.5% |
| retract | 2,274 | 85.5% | 84.4% | 85.9% | 78.5% (n=712) | 63.0% |
| revise | 5,862 | 85.8% | 84.5% | 86.6% | 82.0% (n=1816) | 63.7% |
| second_source | 1,603 | 85.5% | 83.5% | 85.5% | 79.8% (n=519) | 64.4% |
| stale | 1,319 | 87.3% | 86.3% | 87.6% | 81.5% (n=405) | 64.2% |

Confidence required before an action is called READY (exploratory; the pre-registered metrics above use 0.5):

| Threshold | False-READY rate | READY missed | Disposition accuracy |
|---:|---:|---:|---:|
| 0.5 | 5.01% [3.74%, 6.15%] | 11.68% [8.05%, 15.03%] | 85.70% [84.08%, 87.43%] |
| 0.7 | 2.91% [2.00%, 3.85%] | 15.06% [11.35%, 18.76%] | 86.24% [84.55%, 87.93%] |
| 0.8 | 1.78% [1.16%, 2.53%] | 17.90% [14.09%, 21.87%] | 86.27% [84.63%, 87.95%] |
| 0.9 | 1.33% [0.75%, 1.99%] | 25.73% [21.08%, 30.42%] | 84.49% [82.76%, 86.32%] |
| 0.95 | 0.41% [0.11%, 0.79%] | 36.15% [30.85%, 40.87%] | 82.54% [80.60%, 84.35%] |
| 0.99 | 0.18% [0.00%, 0.46%] | 58.42% [53.42%, 63.36%] | 77.23% [75.25%, 79.27%] |

Worst episodes for E: #7 (28% of 46 rows), #25 (31% of 36 rows), #297 (33% of 36 rows), #65 (39% of 100 rows), #184 (43% of 75 rows).

**The dense reader's training.** Initialised from the Stage A weights ({'copied_tensors': 138, 'total_tensors': 140}). 60,000 contexts, 18,477,017 tokens, 60.4 minutes. Validation accuracy 93.10% on 4,000 held-out contexts.

## 4. Exactness (H1)

| Mode | Episodes | Events | Cached nodes checked | Not bit-identical | Largest difference | Proposals that differ |
|---|---:|---:|---:|---:|---:|---:|
| canonical | 200 | 3,559 | 195,434 | 0 | 0.000e+00 | 0 |
| dynamic | 50 | 926 | 53,302 | 46,659 | 8.831e-06 | 0 |

The rebuild bypasses the cache, so every edge is recomputed by the model. Canonical mode fixes the shape of every forward pass; dynamic mode batches freely and is faster.

Separate probe of 200 pairs regrouped into different batches (`runs/determinism_probe.json`): mps/dynamic: 14 differ, max 4.77e-06; mps/canonical: 0 differ, max 0.00e+00; cpu/dynamic: 0 differ, max 0.00e+00; cpu/canonical: 0 differ, max 0.00e+00.

## 5. Wall-clock cost (H2)

Device mps. Times include ledger write, graph propagation, tokenisation and model forward passes.

**dynamic batching**

| Set | Changed events | E p50 / p95 ms | E-full p50 / p95 ms | B2 p50 / p95 ms | E / E-full | E / B2 | Duplicate or stale p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| STANDARD | 635 | 15.9 / 38.0 | 166.7 / 340.1 | 101.5 / 225.0 | 0.088 [0.080, 0.096] | 0.140 [0.128, 0.153] | 0.11 |
| WIDE | 363 | 36.4 / 78.6 | 959.6 / 1385.0 | 931.9 / 1455.0 | 0.038 [0.034, 0.042] | 0.039 [0.034, 0.045] | 0.15 |

**canonical shapes**

| Set | Changed events | E p50 / p95 ms | E-full p50 / p95 ms | B2 p50 / p95 ms | E / E-full | E / B2 | Duplicate or stale p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| STANDARD | 325 | 26.5 / 68.3 | 234.7 / 394.9 | 103.5 / 184.9 | 0.090 [0.080, 0.101] | 0.200 [0.176, 0.230] | 0.11 |
| WIDE | 184 | 61.9 / 133.1 | 1444.7 / 2150.8 | 905.0 / 1656.1 | 0.038 [0.033, 0.046] | 0.060 [0.049, 0.078] | 0.15 |

**Where the advantage disappears.** Replacing the model weights in a task with 630 edges recomputed 630 of them and took 1342 ms, against 1155 ms for a full rebuild (ratio 1.16). A change that touches everything costs as much as starting again.

## 6. Dependency proposals (H8)

Threshold 0.310 on cosine similarity (`sentence-transformers/all-MiniLM-L6-v2`), chosen by conformal risk control at alpha = 0.05 from 1000 calibration tasks (corrected risk 0.0491).

| Set | Missed links (mean over tasks) | Tasks with any miss | Pairs compared | Work reduction | Disposition accuracy with / without | Change |
|---|---:|---:|---:|---:|---:|---:|
| STANDARD | 4.56% [4.02%, 5.15%] | 26.1% | 12.0% | x5.48 | 88.17% / 90.48% | -2.31 pts [-3.22 pts, -1.43 pts] |
| WIDE | 5.11% [4.36%, 5.87%] | 51.3% | 4.4% | x11.22 | 86.69% / 85.70% | +0.99 pts [-0.30 pts, +2.21 pts] |

Pre-registered consequence: keep the proposer off by default = **yes**.

## 7. Implicit conflicts from STALE (H9, exploratory)

400 conflict pairs and 2000 control pairs. Threshold on P(REFUTES) = 0.2071, set for 5.0% false positives on controls. **Not comparable with accuracies reported in the STALE paper (different task, different scoring).**

| Type | n | Detected | AUROC vs controls | Read as REFUTES | Read as NEI | Read as SUPPORTS |
|---|---:|---:|---:|---:|---:|---:|
| T1 | 200 | 77.0% [71.0%, 82.5%] | 0.955 | 56.0% | 43.5% | 0.5% |
| T2 | 200 | 34.5% [28.0%, 41.0%] | 0.880 | 18.0% | 81.5% | 0.5% |
| all | 400 | 55.8% [50.7%, 60.5%] | 0.918 | 37.0% | 62.5% | 0.5% |

At the proposer's threshold (0.310) the old and new statements would not even have been compared in 27.5% of direct conflicts and 68.0% of propagated ones. Similarity separates conflict pairs from controls with AUROC 0.968 (T1) and 0.904 (T2).

Least and most confidently detected:

- T2, P(REFUTES) 0.003, missed: "I usually keep the traditional woven baskets from Ghana on the top shelf so they don't get crushed by heavier items." / "I’m filling out the paperwork for my long-term move to Tokyo next spring, so I’ve been ruthless about downsizing—only two suitcases’ worth is coming with me, and everything else is getting sold or don"
- T2, P(REFUTES) 0.005, missed: "I keep a few emergency rations and water jugs stored in the garage in case I am unable to travel to the store." / "I’m still getting used to this condo—it is much smaller than the house I used to live before."
- T2, P(REFUTES) 0.005, missed: "I generally avoid streaming subscriptions since I prefer owning physical copies of my movies and music." / "I finally caved and bought one of those tiny apartments with almost no storage—between the downsizing and the building’s strict clutter rules, I spent the weekend photographing everything I’m keeping "
- T2, P(REFUTES) 0.007, missed: "I've been hovering around 150 pounds lately, which is where I usually sit." / "Since the thyroid surgery last fall, my endocrinologist has been tweaking my levothyroxine dose every few weeks, and I’m still trying to get my energy and temperature swings to settle down."
- T1, P(REFUTES) 0.974, detected: "I'm settled in the coastal lowlands, so I'm consistently at sea level." / "Ever since I moved up to this mountain town, I’ve had to take it easy on hikes because the thin air hits me pretty fast."
- T1, P(REFUTES) 0.976, detected: "Woodworking is my main passion, and I prioritize spending my free time creating furniture." / "Since the hand injury last year, I’ve switched to collecting and listening to vinyl records as my main pastime, and that’s what I plan my free time around now."
- T1, P(REFUTES) 0.978, detected: "The humidity is so constant that surfaces stay damp all the time." / "Ever since the dry season settled in, the air indoors feels crisp and my wooden table has started to get those tiny static shocks when I wipe it down."
- T1, P(REFUTES) 0.979, detected: "I'm keeping my spending strict because I have about two thousand dollars in liquid funds to work with." / "I’m feeling a lot more flexible with my day-to-day purchases lately since I’ve got around fifteen grand sitting in my checking account right now."

## 8. Learning after deployment (H7)

Verified labels were returned for 30% of items. The encoder was never updated.

### Stream `snli.test` (9,824 items, 2,944 verified)

| Run | Accuracy, whole stream | First half | Second half | Consolidations attempted | Adopted |
|---|---:|---:|---:|---:|---:|
| frozen | 71.83% | 71.91% | 71.76% | 0 | 0 |
| evolving | 71.83% | 71.91% | 71.76% | 8 | 0 |
| corrupted | 71.83% | 71.91% | 71.76% | 8 | 0 |

Second half, evolving minus frozen: +0.00 pts [+0.00 pts, +0.00 pts]. With half of the returned labels wrong, minus frozen: +0.00 pts [+0.00 pts, +0.00 pts].

Largest fall in regression accuracy after any adoption: 0.00% (truthful feedback), 0.00% (corrupted feedback). Tolerance 0.50%.

Consolidations, evolving:

| After items | Verified | Adopted | What | Memory (tau, lam) | Held-out gain [interval] | Regression change |
|---:|---:|---|---|---|---:|---|
| 1000 | 341 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 2250 | 708 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 3250 | 1014 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 4500 | 1387 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 5750 | 1758 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 7000 | 2102 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 8000 | 2412 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 9000 | 2719 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |

Consolidations, corrupted:

| After items | Verified | Adopted | What | Memory (tau, lam) | Held-out gain [interval] | Regression change |
|---:|---:|---|---|---|---:|---|
| 1000 | 341 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 2250 | 708 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 3250 | 1014 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 4500 | 1387 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 5750 | 1758 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 7000 | 2102 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 8000 | 2412 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 9000 | 2719 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |

### Stream `wanli.test` (5,000 items, 1,514 verified)

| Run | Accuracy, whole stream | First half | Second half | Consolidations attempted | Adopted |
|---|---:|---:|---:|---:|---:|
| frozen | 60.46% | 60.80% | 60.12% | 0 | 0 |
| evolving | 69.32% | 66.84% | 71.80% | 4 | 2 |
| corrupted | 60.46% | 60.80% | 60.12% | 4 | 0 |

Second half, evolving minus frozen: +11.68 pts [+9.76 pts, +13.72 pts]. With half of the returned labels wrong, minus frozen: +0.00 pts [+0.00 pts, +0.00 pts].

Largest fall in regression accuracy after any adoption: 0.30% (truthful feedback), 0.00% (corrupted feedback). Tolerance 0.50%.

Consolidations, evolving:

| After items | Verified | Adopted | What | Memory (tau, lam) | Held-out gain [interval] | Regression change |
|---:|---:|---|---|---|---:|---|
| 1250 | 365 | yes | memory | 0.6, 0.6 | +0.2019 [+0.0753, +0.3339] | mnli.dev +13.84, unrelated.dev -0.10, vitaminc.dev +0.28 |
| 2500 | 718 | no | champion | 0.6, 0.6 | +0.0000 [+0.0000, +0.0000] | mnli.dev +13.88, unrelated.dev -0.03, vitaminc.dev +0.30 |
| 3500 | 1020 | yes | refiner+memory | 0.6, 0.0 | +0.0744 [+0.0135, +0.1374] | mnli.dev +17.34, unrelated.dev -0.30, vitaminc.dev +0.44 |
| 4500 | 1357 | no | champion | 0.6, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +17.34, unrelated.dev -0.30, vitaminc.dev +0.44 |

Consolidations, corrupted:

| After items | Verified | Adopted | What | Memory (tau, lam) | Held-out gain [interval] | Regression change |
|---:|---:|---|---|---|---:|---|
| 1250 | 365 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 2500 | 718 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 3500 | 1020 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |
| 4500 | 1357 | no | champion | 0.8, 0.0 | +0.0000 [+0.0000, +0.0000] | mnli.dev +0.00, unrelated.dev +0.00, vitaminc.dev +0.00 |

## 8b. Requirements that need two records read together (X1, exploratory)

Not pre-registered. 800 cases; wording is templated; 10 subjects x 3 claim, 3 approval and 3 latest-version phrasings. Example: claim "The client has approved the latest version of the logo", records "The client approved version 4 of the logo" and "The latest version of the logo is version 5". Calling the claim satisfied when the versions differ is the unsafe outcome.

| Reading | Correct | Unsafe (says satisfied when it is not) | Versions differ: satisfied / violated / open | Versions match: satisfied / violated / open | Only one record: open |
|---|---:|---:|---:|---:|---:|
| dense reader (B2) | 63.6% | 48.5% | 54% / 46% / 0% | 100% / 0% / 0% | 54% |
| joint, same edge model | 67.6% | 27.5% | 26% / 66% / 8% | 65% / 0% / 35% | 70% |
| separate (E) | 55.8% | 46.3% | 82% / 1% / 16% | 82% / 1% / 16% | 70% |

## 8c. Small tasks (X2, exploratory)

Not pre-registered; designed after the demo below, before any episode-level result was known. 1,000 episodes from the test pool with 3.0 predicates and 4.6 records per task on average.

| System | Disposition accuracy | False-READY rate | Status accuracy | Status NLL |
|---|---:|---:|---:|---:|
| B1-hard | 91.25% [89.99%, 92.53%] | 3.03% [2.13%, 3.91%] | 92.02% [91.10%, 92.97%] | 2.204 [1.943, 2.459] |
| B1-soft | 91.30% [90.10%, 92.48%] | 2.66% [1.88%, 3.50%] | 91.79% [90.84%, 92.72%] | 0.222 [0.201, 0.243] |
| E-seed1 | 91.19% [89.96%, 92.41%] | 2.45% [1.68%, 3.29%] | 91.66% [90.74%, 92.58%] | 0.235 [0.213, 0.260] |
| E-seed2 | 91.08% [89.85%, 92.28%] | 2.48% [1.72%, 3.31%] | 91.29% [90.32%, 92.22%] | 0.242 [0.219, 0.268] |
| E-seed3 | 90.40% [89.07%, 91.64%] | 2.75% [1.98%, 3.61%] | 91.20% [90.19%, 92.13%] | 0.252 [0.225, 0.281] |
| N-seed1 | 91.27% [90.09%, 92.45%] | 2.81% [2.03%, 3.64%] | 91.61% [90.73%, 92.46%] | 0.234 [0.211, 0.259] |
| N-seed2 | 91.20% [89.99%, 92.35%] | 2.51% [1.77%, 3.28%] | 91.50% [90.56%, 92.38%] | 0.235 [0.212, 0.260] |
| N-seed3 | 91.21% [90.02%, 92.41%] | 2.67% [1.90%, 3.49%] | 91.55% [90.63%, 92.44%] | 0.232 [0.209, 0.257] |

| Comparison | Status NLL | Disposition accuracy |
|---|---:|---:|
| B1-hard vs B1-soft | +1.9819 [+1.7399, +2.2226] | -0.06 pts [-0.54 pts, +0.43 pts] |
| E-seed1 vs B1-soft | +0.0137 [+0.0064, +0.0217] | -0.11 pts [-0.67 pts, +0.44 pts] |
| E-seed2 vs B1-soft | +0.0207 [+0.0107, +0.0317] | -0.22 pts [-0.80 pts, +0.33 pts] |
| E-seed3 vs B1-soft | +0.0301 [+0.0175, +0.0435] | -0.91 pts [-1.68 pts, -0.20 pts] |
| N-seed1 vs B1-soft | +0.0121 [+0.0033, +0.0209] | -0.03 pts [-0.62 pts, +0.52 pts] |
| N-seed2 vs B1-soft | +0.0133 [+0.0047, +0.0220] | -0.10 pts [-0.76 pts, +0.49 pts] |
| N-seed3 vs B1-soft | +0.0103 [+0.0018, +0.0186] | -0.09 pts [-0.79 pts, +0.56 pts] |

## 8d. Readings per predicate (X3, exploratory)

Not pre-registered; designed after the STANDARD results showed E-declared below E. "Declared links" compares each record only with the predicate it was written for, so each predicate has one to three readings instead of about eleven.

| Set | System | Disposition accuracy | Status accuracy | Status NLL |
|---|---|---:|---:|---:|
| STANDARD | E, all links | 90.48% [89.40%, 91.51%] | 90.54% [89.81%, 91.24%] | 0.260 [0.242, 0.279] |
| STANDARD | E, declared links | 87.03% [85.71%, 88.33%] | 86.13% [85.19%, 87.02%] | 0.353 [0.330, 0.377] |
| STANDARD | rules, all links | 89.52% [88.46%, 90.60%] | 89.97% [89.24%, 90.68%] | 0.279 [0.260, 0.297] |
| STANDARD | rules, declared links | 90.74% [89.71%, 91.77%] | 91.48% [90.76%, 92.19%] | 0.231 [0.215, 0.248] |
| WIDE | E, all links | 85.70% [83.93%, 87.53%] | 86.89% [85.99%, 87.83%] | 0.351 [0.327, 0.376] |
| WIDE | E, declared links | 85.55% [83.69%, 87.61%] | 86.59% [85.58%, 87.64%] | 0.352 [0.324, 0.382] |
| WIDE | rules, all links | 84.63% [82.69%, 86.59%] | 86.69% [85.72%, 87.67%] | 0.403 [0.373, 0.433] |
| WIDE | rules, declared links | 88.70% [86.92%, 90.39%] | 90.90% [90.03%, 91.74%] | 0.247 [0.227, 0.268] |

| Set | Comparison | Disposition accuracy | Status NLL |
|---|---|---:|---:|
| STANDARD | E vs rules, all links | +0.96 pts [+0.30 pts, +1.61 pts] | -0.0187 [-0.0264, -0.0118] |
| STANDARD | E vs rules, declared links | -3.71 pts [-4.78 pts, -2.67 pts] | +0.1221 [+0.1090, +0.1360] |
| STANDARD | E: declared vs all links | -3.45 pts [-4.61 pts, -2.41 pts] | +0.0929 [+0.0770, +0.1086] |
| STANDARD | rules: declared vs all links | +1.22 pts [+0.72 pts, +1.72 pts] | -0.0479 [-0.0592, -0.0375] |
| WIDE | E vs rules, all links | +1.06 pts [-0.08 pts, +2.26 pts] | -0.0521 [-0.0696, -0.0356] |
| WIDE | E vs rules, declared links | -3.15 pts [-4.63 pts, -1.74 pts] | +0.1054 [+0.0907, +0.1203] |
| WIDE | E: declared vs all links | -0.15 pts [-1.83 pts, +1.51 pts] | +0.0012 [-0.0225, +0.0250] |
| WIDE | rules: declared vs all links | +4.07 pts [+2.81 pts, +5.41 pts] | -0.1563 [-0.1818, -0.1327] |

Status accuracy by the number of readings a predicate has (first 300 episodes of each set):

| Set | Readings | n | E | Calibrated rules |
|---|---|---:|---:|---:|
| STANDARD | 0-0 | 11 | 100.0% | 100.0% |
| STANDARD | 1-1 | 106 | 95.3% | 97.2% |
| STANDARD | 2-2 | 245 | 95.1% | 93.1% |
| STANDARD | 3-4 | 893 | 92.7% | 92.3% |
| STANDARD | 5-8 | 5,649 | 91.0% | 90.5% |
| STANDARD | 9-16 | 9,699 | 90.7% | 90.2% |
| STANDARD | 17+ | 245 | 80.4% | 77.6% |
| WIDE | 9-16 | 317 | 93.1% | 88.6% |
| WIDE | 17+ | 66,267 | 86.9% | 86.7% |

## 8e. The dependency proposer with calibrated rules (X3b, exploratory)

Not pre-registered. Same proposer and threshold as H8 (0.310); nothing re-tuned.

| Set | System | Disposition accuracy | False-READY rate | Tokens per changed event |
|---|---|---:|---:|---:|
| STANDARD | E + proposer | 88.17% [86.91%, 89.40%] | 2.19% [1.59%, 2.79%] | 27 [26, 27] |
| STANDARD | rules + proposer | 89.71% [88.57%, 90.80%] | 2.91% [2.21%, 3.62%] | 27 [26, 27] |
| STANDARD | rules, all links | 89.52% [88.46%, 90.60%] | 2.97% [2.27%, 3.72%] | 146 [141, 151] |
| WIDE | E + proposer | 86.69% [84.81%, 88.50%] | 3.13% [2.19%, 4.13%] | 32 [31, 33] |
| WIDE | rules + proposer | 86.62% [84.78%, 88.46%] | 3.76% [2.71%, 4.86%] | 32 [31, 33] |
| WIDE | rules, all links | 84.63% [82.69%, 86.59%] | 4.20% [3.07%, 5.35%] | 359 [344, 373] |

| Set | Comparison | Disposition accuracy | Tokens ratio |
|---|---|---:|---:|
| STANDARD | rules + proposer vs E + proposer | +1.54 pts [+0.72 pts, +2.39 pts] | 1.000 [1.000, 1.000] |
| STANDARD | rules + proposer vs rules, all links | +0.19 pts [-0.26 pts, +0.63 pts] | 0.183 [0.179, 0.187] |
| WIDE | rules + proposer vs E + proposer | -0.07 pts [-1.14 pts, +1.00 pts] | 1.000 [1.000, 1.000] |
| WIDE | rules + proposer vs rules, all links | +1.99 pts [+0.88 pts, +3.08 pts] | 0.089 [0.087, 0.092] |

## 8f. Would set suites have caught the failing refiner? (non-test data)

Set suites are built from Stage B-pool episodes only (`scripts/build_set_suites.py`). Accuracy and NLL of each predicate's status:

| Aggregator | small | standard | wide |
|---|---:|---:|---:|
| rules | 92.2% (NLL 0.205) | 90.7% (NLL 0.257) | 86.9% (NLL 0.401) |
| refined-seed1 | 92.0% (NLL 0.210) | 91.9% (NLL 0.223) | 88.9% (NLL 0.317) |
| refined-seed2 | 91.6% (NLL 0.223) | 91.9% (NLL 0.228) | 88.5% (NLL 0.312) |
| refined-seed3 | 91.7% (NLL 0.223) | 91.9% (NLL 0.227) | 77.3% (NLL 0.762) |

## 8g. Self-evolution from calibrated rules, with set suites (X4, exploratory)

Not pre-registered. The H7 procedure repeated with the changes described in Deviations, item 9: calibrated rules as the starting point, set suites in the gate, context frozen during fine-tuning.

| Stream | Items | Verified | Frozen accuracy | Evolving, second half | Corrupted, second half | Consolidations adopted (evolving / corrupted) |
|---|---:|---:|---:|---:|---:|---:|
| mnli_mm.test | 9,832 | 2,950 | 88.27% | 88.00% (+0.00 pts) | 88.00% (+0.00 pts) | 0/8 / 0/8 |
| snli.test | 9,824 | 2,944 | 79.82% | 79.93% (+0.00 pts) | 79.93% (+0.00 pts) | 0/8 / 0/8 |
| wanli.test | 5,000 | 1,514 | 74.66% | 74.04% (+0.00 pts) | 74.04% (+0.00 pts) | 0/4 / 0/4 |

Why candidates were rejected (counts over all attempts and candidates): sets:wide 55, sets:standard 36, sets:small 28, vitaminc.dev 27, unrelated.dev 24, mnli.dev 16, no reliable held-out gain 5.

## 8h. The hand-off demo, scored (illustration)

One hand-written scenario; an illustration, not an estimate of error rates. The scenario's twelve messages are email-like text of a kind absent from training. Correct statuses were written from the messages' wording before any model was trained.

| System | READY threshold | Predicate statuses correct | Action dispositions correct | False READY |
|---|---:|---:|---:|---:|
| E (refined, seed 1) | 0.5 | 45/48 | 21/24 | 0 |
| B1-soft (calibrated rules) | 0.5 | 48/48 | 24/24 | 0 |
| B1-hard (top label per record) | 0.5 | 48/48 | 24/24 | 0 |
| E (refined, seed 1) | 0.9 | 45/48 | 15/24 | 0 |
| B1-soft (calibrated rules) | 0.9 | 48/48 | 18/24 | 0 |
| B1-hard (top label per record) | 0.9 | 48/48 | 24/24 | 0 |

Misreadings at threshold 0.5:

- E (refined, seed 1), event 7 (The client withdraws the approval.): `design_approved` read as UNRESOLVED, should be VIOLATED (satisfied 0.12, violated 0.31, open 0.57)
- E (refined, seed 1), event 8 (A late copy of the original approval arrives from an archive.): `design_approved` read as UNRESOLVED, should be VIOLATED (satisfied 0.12, violated 0.31, open 0.57)
- E (refined, seed 1), event 10 (The revised design is approved.): `date_confirmed` read as SATISFIED, should be VIOLATED (satisfied 0.53, violated 0.35, open 0.12)

## 8i. What the release ships, and why

Aggregator: **calibrated rules (refiner with zero correction)**, on the large edge model: fixed by the plan (docs/PREREGISTRATION_LARGE.md): calibrated rules on top of the edge model.

## 8j. Where more accuracy would come from (X5, exploratory)

Not pre-registered; run after every result above was known, to decide what to do next. The released model is unchanged by any of it.

**Replay with reading errors corrected.** The test episodes above, run again with the shipped rules after replacing some of the edge model's readings by the label the generator recorded. A reading is *wrong* when its most probable label differs from that label. The *linked* record is the one written for the requirement; every other record in the task is *unrelated* to it.

*STANDARD*: 71,188 readings, 98.48% correct (88.85% on the 6,547 linked, 99.46% on the unrelated); 730 errors on linked records, 350 on unrelated ones.

| Readings replaced | Number | Disposition accuracy | Stale decisions | False READY | Missed READY |
|---|---:|---:|---:|---:|---:|
| as measured | 0 | 89.52% [88.47%, 90.55%] | 13.05% | 2.97% | 9.04% |
| 25% of reading errors corrected | 270 | 91.91% [90.91%, 92.81%] | 10.16% | 2.26% | 7.57% |
| 50% of reading errors corrected | 540 | 94.18% [93.30%, 94.97%] | 7.18% | 1.66% | 5.52% |
| 75% of reading errors corrected | 810 | 96.42% [95.70%, 97.02%] | 4.66% | 0.89% | 4.10% |
| errors on unrelated records corrected | 350 | 90.78% [89.73%, 91.73%] | 11.75% | 2.71% | 8.18% |
| errors on the linked record corrected | 730 | 97.13% [96.49%, 97.72%] | 3.22% | 0.45% | 2.80% |
| all reading errors corrected | 1,080 | 98.78% [98.34%, 99.15%] | 1.49% | 0.12% | 1.83% |
| every reading correct and confident | 71,188 | 100.00% [100.00%, 100.00%] | 0.00% | 0.00% | 0.00% |

*WIDE*: 141,373 readings, 99.13% correct (88.66% on the 4,393 linked, 99.47% on the unrelated); 498 errors on linked records, 730 on unrelated ones.

| Readings replaced | Number | Disposition accuracy | Stale decisions | False READY | Missed READY |
|---|---:|---:|---:|---:|---:|
| as measured | 0 | 84.63% [82.63%, 86.53%] | 16.26% | 4.20% | 13.89% |
| 25% of reading errors corrected | 307 | 87.14% [85.33%, 88.90%] | 13.24% | 3.69% | 11.53% |
| 50% of reading errors corrected | 614 | 89.25% [87.62%, 90.78%] | 11.03% | 3.08% | 9.64% |
| 75% of reading errors corrected | 921 | 93.91% [92.62%, 95.16%] | 5.46% | 1.37% | 6.21% |
| errors on unrelated records corrected | 730 | 88.94% [87.21%, 90.56%] | 12.43% | 3.02% | 9.93% |
| errors on the linked record corrected | 498 | 93.06% [91.71%, 94.55%] | 6.97% | 1.48% | 5.97% |
| all reading errors corrected | 1,228 | 98.59% [97.90%, 99.19%] | 1.97% | 0.00% | 1.42% |
| every reading correct and confident | 141,373 | 100.00% [100.00%, 100.00%] | 0.00% | 0.00% | 0.00% |

With every reading correct and confident the exact parts reproduce the generator's answer for every action at every step, so on these tasks all error comes from reading. The gap between the last two rows is readings that were right but hesitant. The partial rows correct a random share, chosen once (seed 0).

**Fit.** Accuracy on a systematic sample of each training split against held-out data. The sample includes rows the model never drew during training.

| Corpus | Training split (sample) | n | Held-out | n | Gap |
|---|---:|---:|---:|---:|---:|
| mnli | 91.22% | 4,909 | 88.56% (`mnli_mm.test`) | 9,832 | +2.66 pts |
| vitaminc | 93.56% | 6,178 | 90.20% (`vitaminc.test`) | 55,197 | +3.36 pts |
| wanli | 82.00% | 4,116 | 74.74% (`wanli.test`) | 5,000 | +7.26 pts |

**Evidence on the same topic that settles nothing.** On `vitaminc.test`, of the passages labelled as not settling their claim, 22.29% were read as supporting or refuting it. These passages are about the claim's subject. For passages on another subject the figure is 0.82% (`unrelated.test`).

**More training of the same model on the same corpora.** The released edge model (A, 900,000 examples) was trained for 200,000 more examples on the same mixture (B; `configs/diag_more_training.yaml`) and both were run on the development sets. No test split was read, and B is not released.

| Development set | n | A | B | B − A | Only A right / only B right | p (McNemar, exact) | Log loss A → B |
|---|---:|---:|---:|---:|---:|---:|---:|
| mnli.dev | 9,815 | 88.53% | 88.14% | -0.39 pts [-0.74, -0.04] | 176 / 138 | 0.037 | 0.310 → 0.327 |
| snli.dev | 3,000 | 80.87% | 81.30% | +0.43 pts [-0.33, +1.17] | 62 / 75 | 0.305 | 0.515 → 0.530 |
| unrelated.dev | 3,000 | 99.57% | 99.73% | +0.17 pts [+0.00, +0.33] | 1 / 6 | 0.125 | 0.011 → 0.010 |
| vitaminc.dev | 6,001 | 85.50% | 85.02% | -0.48 pts [-0.85, -0.15] | 69 / 40 | 0.007 | 0.457 → 0.498 |

## 8k. Running without a GPU (exploratory)

Timing of the shipped configuration by device; exploratory. Canonical shapes. Apple M4 Max, 12 performance and 4 efficiency cores. Times include ledger write, propagation, tokenisation and the model's forward passes. The shipped configuration on the first test episodes of each kind.

| Device | Threads | STANDARD update, median (95th) | WIDE update, median (95th) | First full reading of a task, median (STANDARD / WIDE) | Peak memory |
|---|---:|---:|---:|---:|---:|
| Apple GPU | 12 | 23 ms (57 ms) | 48 ms (87 ms) | 0.3 s / 1.6 s | 2.6 GB |
| CPU, 4 threads | 4 | 93 ms (285 ms) | 190 ms (406 ms) | 1.1 s / 6.8 s | 2.5 GB |
| CPU, all cores | 12 | 90 ms (242 ms) | 187 ms (376 ms) | 1.0 s / 6.3 s | 2.5 GB |

## 8l. A larger encoder (pre-registered addendum)

Plan: `docs/PREREGISTRATION_LARGE.md`, SHA-256 `9fec551bc3764860dcf355de347113009b79299d15d85679d1a6eb44c8606417`, recorded 2026-10-03T00:42:26Z before the large model was trained. `answerdotai/ModernBERT-large` by the released recipe, calibrated rules on top, against the released base model; everything paired.

- **H10** (reading: the large model is more accurate on `vitaminc.test` and `mnli_mm.test`): **supported**
- **H11** (decisions: higher disposition accuracy on STANDARD): **supported**
- **H12** (cost: median update on STANDARD at most 3.5× the base model's): **supported**
- **Which model ships** by the rule in the plan: the large model.

| Edge test set | n | Base | Large | Large − Base | Only base right / only large right | p (McNemar) |
|---|---:|---:|---:|---:|---:|---:|
| anli_r1.test | 1,000 | 46.40% | 53.70% | +7.30 pts [+4.50, +10.10] | 72 / 145 | 8.12e-07 |
| anli_r2.test | 1,000 | 33.30% | 38.10% | +4.80 pts [+1.90, +7.50] | 77 / 125 | 0.000898 |
| anli_r3.test | 1,200 | 34.75% | 36.00% | +1.25 pts [-1.33, +3.92] | 114 / 129 | 0.369 |
| mnli_mm.test | 9,832 | 88.56% | 90.21% | +1.65 pts [+1.12, +2.15] | 267 / 429 | 8.8e-10 |
| snli.test | 9,824 | 80.33% | 84.39% | +4.05 pts [+3.39, +4.72] | 364 / 762 | 5.86e-33 |
| unrelated.test | 6,000 | 99.18% | 99.42% | +0.23 pts [+0.07, +0.38] | 5 / 19 | 0.00661 |
| vitaminc.test | 55,197 | 90.20% | 91.53% | +1.33 pts [+1.14, +1.52] | 1044 / 1779 | 6.88e-44 |
| wanli.test | 5,000 | 74.74% | 77.42% | +2.68 pts [+1.82, +3.64] | 194 / 328 | 4.84e-09 |

Unrelated passages read as decisive: base 0.82%, large 0.58%.

VitaminC real revisions: base 88.65%, large 89.90%; synthetic: base 92.78%, large 94.25%.

| Test set | Episodes | Disposition accuracy, base | large | Large − Base | Stale decisions, base | large | False READY, base | large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SMALL | 1,000 | 91.30% | 92.00% | +0.70 pts [-0.25, +1.61] | 8.99% | 7.47% | 2.66% | 2.38% |
| STANDARD | 1,000 | 89.52% | 91.55% | +2.03 pts [+1.06, +2.94] | 13.05% | 11.10% | 2.97% | 2.75% |
| WIDE | 300 | 84.63% | 86.83% | +2.20 pts [+0.68, +3.61] | 16.26% | 16.14% | 4.20% | 3.38% |

| Model | Device | STANDARD update, median (95th) | WIDE update, median (95th) | Peak memory |
|---|---|---:|---:|---:|
| base | mps | 22 ms (57 ms) | 48 ms (85 ms) | 2.5 GB |
| large | mps | 59 ms (171 ms) | 118 ms (230 ms) | 4.8 GB |
| large, CPU | cpu | 210 ms (581 ms) | 433 ms (816 ms) | 4.8 GB |

Median update time, large / base, on STANDARD: 2.68.

## 9. Supporting checks

- Mechanism probe, sparse: 21.52 of 208 nodes recomputed per change (10.35%); 0 nodes differed from a full rebuild over 1500 events.
- Mechanism probe, dense: 208.00 of 208 nodes recomputed per change (100.00%); 0 nodes differed from a full rebuild over 1500 events.
- Leakage audit, stage_b pool: 0 of 68,752 pairs occur in a training split; 328 of 28,590 VitaminC atoms share a page with training data.
- Leakage audit, test pool: 14 of 79,851 pairs occur in a training split; 279 of 27,661 VitaminC atoms share a page with training data.

## 10. Deviations from the pre-registered plan

Each entry is something done differently from `docs/PREREGISTRATION.md`, with the reason and with
what it could affect.

1. **The dense reader (B2) was evaluated on a subset of the test episodes**: the first 400 of 1,000
   STANDARD episodes and the first 100 of 300 WIDE episodes. Reason: a dry run on non-test data
   showed B2 needs about 5.0 million tokens of inference for 12 WIDE episodes, which puts the full
   sets at several hours. Episodes are independent draws from the generator, so the first K are a
   random sample. Effect: intervals for comparisons involving B2 are wider than planned. The
   comparisons are paired on exactly those episodes (rows marked `@B2`). All other systems were
   evaluated on the full sets. Decided before any test episode was evaluated.

2. **A dry run printed test-set accuracy of a discarded model.** While checking that
   `scripts/eval_edge.py` ran, it was pointed at the first 150 rows of every evaluation set,
   including test sets, using the pilot model that had already been discarded for its training
   mixture. Those figures were printed. Nothing was chosen or changed on the basis of them, and the
   final model had not finished training. Later dry runs used non-test data or suppressed output.

3. **H2 was given two tables instead of one.** Wall-clock time was measured both with dynamic
   batching (the faster mode) and with canonical shapes (the mode in which H1 holds). The
   pre-registered decision rule is applied to dynamic batching, which was the mode intended when
   the plan was written; canonical timings are reported beside it.

4. **H6 was given a decision rule.** The plan said only "reported either way". The report counts
   the prediction as held if the 95% interval of (N minus E) disposition accuracy on WIDE lies
   below zero.

5. **The edge model was trained twice; the first run was discarded on development data.** The first
   full run finished with 78.8% on the VitaminC development set, below a pilot trained on
   one fourteenth of the data (82.2%). The cause was an ordered training file combined with a
   shuffle buffer smaller than the file (`docs/RESEARCH.md`, section 6). The experiment pipeline
   had started automatically when that run finished. It was stopped within a minute, during its
   first step. It had evaluated the four development sets and no test set; its output was set
   aside unread (`runs/aborted_pipeline_1`). The decision to retrain, and the fix, were made from
   development data and from the training file's own metadata. Nothing about the evaluation plan
   was changed.

6. **Additions made after the pipeline was frozen.** The experiments ran from a frozen copy of the
   code (`runs/code_snapshot`, hash in `runs/results_waiter.log`). Afterwards, and before any
   episode-level result was known, three exploratory additions were made, each labelled as such in
   the results:
   * *the demo, scored* (`scripts/demo_trace.py`): the hand-off scenario was scored against
     statuses written from its messages. The learned refiner misread three predicate states that
     calibrated rules read correctly.
   * *X2, small tasks* (`scripts/eval_small.py`): designed because of that observation, to test
     whether it holds beyond one scenario.
   * *report sections* for those two. The code that computes verdicts in `scripts/make_report.py`
     is unchanged from the frozen copy (checked by `diff`).
   By then the edge model's test-set results (`runs/results/edge.json`) and the public model's
   were known. They did not bear on these additions.

7. **How the released default is chosen, fixed before episode-level results.** The release ships
   the learned refiner (seed 1) only if (a) H5 is supported, and (b) on each of STANDARD, WIDE and
   the small tasks of X2, the refiner is not significantly worse than calibrated rules: the 95%
   interval of its status-NLL difference does not lie entirely above zero, and the interval of its
   disposition-accuracy difference does not lie entirely below zero. Otherwise the release ships
   calibrated rules, written as a refiner whose learned correction is zero
   (`ease.aggregate.rule_refiner`). That form computes exactly what calibrated rules compute and
   can still be improved later by gated consolidation.

8. **X3, readings per predicate, was designed after the STANDARD episode results were known.** Those
   results showed E-declared (each record compared only with the predicate it was written for)
   scoring 3.5 points below E, although it has more information. X3 tests one explanation: that
   the learned refiner depends on how many readings a predicate has. It is exploratory and is
   reported as such; no pre-registered verdict depends on it.

9. **Changes made after the experiments exposed a weakness, and one more exploratory experiment.**
   The WIDE results showed one of three refiners (seed 3) losing 6.8 points against calibrated
   rules, and X3 showed the refiner degrading when a predicate has few readings. The consolidation
   gate of H7 checked candidates only on single readings, so it could not have caught either.
   After those results, and before any conclusion was drawn from H7:
   * the gate gained *set suites*: whole predicates from validation episodes of the Stage B pool,
     in three sizes (`scripts/build_set_suites.py`). On them, seed 3 scores 77.3% against 88.9%
     for seed 1 on the large-task suite, while the standard suite cannot tell them apart
     (`runs/results/set_suites_check.json`). No test data is involved;
   * fine-tuning during consolidation no longer changes how the refiner reads the rest of a set,
     unless explicitly allowed;
   * calibrated rules can be written as a refiner with a zero correction and no context
     (`ease.aggregate.rule_refiner`), so they can be the starting point of self-evolution.
   H7 was run by the frozen code, without any of these. **X4** (exploratory) repeats the H7
   procedure with them, starting from calibrated rules, on the same streams plus MNLI-mismatched,
   whose genres are absent from training and whose labels follow the same conventions as training.


10. **Diagnostics run after all results were known, to decide what to do next (X5).** None was
    pre-registered and none changes a verdict or the released model.
    * *Replay with corrected readings* (`scripts/diagnose_headroom.py`): the STANDARD and WIDE test
      episodes, already evaluated, run again with some of the edge model's readings replaced by
      the generator's labels. It uses test labels, so it is analysis of the reported runs, not a
      new test, and nothing was tuned on it.
    * *Fit check*: the released model read a sample of its training splits.
    * *More training* (`configs/diag_more_training.yaml`, `scripts/compare_models.py`): the
      released model trained for 200,000 more examples and compared with the released model on
      the development sets only. The continued model was never run on a test split and is not
      released.
    * *Running without a GPU* (`scripts/eval_cpu.py`): update time of the shipped configuration
      on the CPU. Timing only.
    Results: section 8j and 8k.

11. **A fault found by using the service, and fixed.** The planner proposed questions only about
    requirements the model considered open (at least 0.2 probability of being unresolved). A
    requirement read as satisfied at about 0.86 is not open, so an action could sit below its
    threshold with no question proposed. Candidates are now all requirements the model is not
    sure of, and the value-of-information calculation decides among them
    (`ease.planner.askable_predicates`, two tests). No reported measurement used the planner's
    choice of question, so no figure in this report changes.

12. **The released aggregator file was rebuilt so that packaging is repeatable.** Calibrated rules
    are stored as a refiner whose last layer is zero. Its hidden layers, which do not affect the
    output, were drawn from the global random generator, so packaging the same release twice gave
    two different files and version identifiers. They are now drawn from a fixed seed
    (`ease.aggregate.rule_refiner`, one test). The output is unchanged: exactly the calibrated
    rules. X4 started from a refiner built the old way; its results are reported as measured.

13. **A second planner fault found by using the page, and fixed.** "Unnecessary" questions were
    certified one predicate at a time: a requirement whose answer changes nothing *on its own*
    (an AND with another open requirement) was reported as unable to change the proposal, while
    the same requirement appeared in the question worth asking. A predicate is now unnecessary
    for an action only if its answer cannot change the proposal under every settlement of the
    other open predicates (`ease.planner.unnecessary_questions`, one test). No reported
    measurement used this function.

14. **The larger-encoder addendum, run as planned.** `docs/PREREGISTRATION_LARGE.md` was hashed
    before training; its hash is unchanged. The pilot found batch 32 fits, so the main setting
    was used. Three things were done that the plan did not spell out, none of which touches a
    verdict: the Stage B script fitted one refiner seed as a by-product of fitting the rules'
    calibration (the refiner is not used; calibrated rules ship, and `package_release.py
    --force-rules` makes that choice explicit); the update-time comparison of H12 used
    `scripts/eval_cpu.py` with each model's calibrated rules, both models in one run, rather than
    the H2 script, which needs the dense baseline; and the fit and headroom diagnostics of X5
    were repeated for the large model (exploratory). The base model is published alongside as
    `ease-delta-base`, for machines where 2.7× the update time matters.

## 11. Limitations

These were written before the results were known and are not excuses for them. Items marked *(updated)* were revised afterwards to replace an expectation with what was measured.

**What was not measured**

* *People.* The blueprint's primary measure was human correction and completion time per verified
  task. No person took part in any experiment here. Whether this saves anyone time is unknown.
* *Other products.* No hosted model was called. Nothing here says how this compares with Jev, Kev,
  Nimble or any assistant.
* *Energy.* Not measured.
* *Languages.* English only. The backbone was pretrained on English and code.

**Where the evaluation is easier than real use**

* *Structure is generated.* Text in the episodes is human-written, but which record concerns which
  predicate, and how requirements combine, were produced by a generator. Real tasks are messier.
* *(updated)* *Distractors are off-topic.* Records unrelated to a predicate come from different Wikipedia pages
  or different NLI prompts. In a real project the distractors are other messages about the same
  project. The only same-topic hard negatives are VitaminC evidence versions labelled NEI, and
  on those the reader is much weaker: passages about a claim's subject that do not settle it were
  read as settling it 22% of the time, against 0.8% for passages on another subject (results,
  section 8j). Expect cross-talk between the requirements of one project. In a hand check of the
  service, an email withdrawing approval of a design visibly lowered the belief that the
  delivery date had been confirmed. Saying which requirement a record concerns (`about`) removes
  this; a person's correction of a single reading takes effect at once.
* *(updated)* *One record per question, and references such as "the latest version".* In the test sets each
  predicate is settled by single records. A requirement that needs two records read together is
  read **unsafely**, as measured in X1 (results, section 8b): for "the client has approved the
  latest version" with records "approved version 4" and "the latest is version 5", reading each
  record separately answered *satisfied* 82% of the time. Reading the two together
  (`join=True`) lowered that to 26%; the dense reader answered satisfied 54% of the time. X1 uses
  templated wording on one pattern, so the numbers are indicative, but the direction is clear.
  **Write requirements that name the specific thing** ("the client has approved version 5"), and
  let exact code resolve references such as "latest" or "current" before a requirement is read.
* *Domain.* Training text is Wikipedia sentences and NLI corpora. Email and contract language is
  out of distribution. The demo shows behaviour on such text; it is one scenario, not a test.
* *Attribution.* The reader sees a record's text and nothing else. A passage that does not say who
  is speaking cannot establish who approved what. Ingestion keeps an attribution in each passage
  only if it is given one.
* *Passages.* Documents are cut at paragraph and sentence boundaries. A statement and the
  condition that qualifies it can land in different passages.

**Assumptions that real data can break**

* A1 (predicates independent given the evidence) and A2 (reading errors independent across
  edges), `docs/PROOFS.md`. Two predicates settled by one sentence violate A1. Two records quoting
  the same sentence violate A2.
* The ledger assumes each source numbers its revisions consistently and that record identity is
  stable. It cannot detect that two differently named records are the same document.
* The endorsement bound (T5) assumes verdicts are truthful.
* The missed-link bound (T6) assumes new tasks resemble calibration tasks, and covers only links
  of the kind present in the calibration data.

**What the model is weak at by construction**

* *(updated)* Numbers and dates are read as text by a neural encoder, not compared by code. Measured on
  VitaminC test (results, section 2), claims containing a numeric comparison were read at least
  as accurately as claims without numbers, so this is not the weakness it was expected to be on
  that corpus. VitaminC's comparisons follow a few patterns ("more than 540,000" against
  "545,500"); other formats, units, arithmetic and date reasoning were not tested and should be
  handled by exact code when they matter.
* Consequences that need world knowledge ("broke my leg" against "cycles to work") are not taught
  by any training corpus used here.

**Found by the experiments** *(added after measurement)*

* *The learned refiner is fragile outside the conditions it was trained in.* One of three refiners
  trained by the same procedure lost 6.8 points on larger tasks; all three lose to calibrated rules
  when a predicate has few readings. The release therefore ships calibrated rules. The refiner is
  kept as a research component.
* *Self-evolution learned nothing it could verify as safe from calibrated rules.* The exact
  corrections and the threshold tracker work as specified; gated consolidation adopted no update
  in X4. Its value in real use is unmeasured.
* *Propagated conflicts are mostly missed.* The edge model recognised 35% of conflicts that follow
  from a consequence, against 77% of direct ones (STALE, at 5% false positives).
* *All measured error is reading error, and more of the same training does not reduce it.* With
  every reading correct and confident, the exact parts gave the right answer for every action at
  every step of every test task (section 8j). The edge model reads the record written for a
  requirement correctly 89% of the time (base) or 90.5% (large), and that figure limits the
  system. Training the base model for 200,000 more examples on the same corpora made it slightly
  worse on development data. A larger encoder, trained under a second plan, gained 2.0 and 2.2
  points on the two task sets (section 8l), close to the estimate made beforehand; it is what
  ships.
* *It is cautious on text unlike its training data.* In the hand-off demo, on email-like text, no
  action reached the default endorsement threshold of 0.9 (section 8h). That is the safe
  direction. It also means a person is still asked to confirm.

**What one run cannot show**

* Each edge model (149M and 395M parameters) and the dense baseline were trained once, with one
  seed, because each takes hours on this machine. Their run-to-run variation is unknown. The
  small aggregators were trained with three seeds.
* All timings are from one laptop with other applications running.

**Deployment**

* The HTTP API has no authentication and is meant for the local machine.
* Run on macOS with Apple silicon (Python 3.12) and, through the Dockerfile, on Linux (aarch64,
  CPU) where the test suite passes and the report regenerates. Windows, x86-64 and other Python
  versions are untested. Without a GPU, an update took about 0.1 to 0.2 seconds with the base
  reader and 0.2 to 0.4 seconds with the shipped larger reader on this machine's processor, with
  2.5 and 4.8 GB of memory (sections 8k and 8l). Slower processors were not measured.
* With `--multi`, workspaces are separated by key and by files; the model and its reading cache
  are shared. One process serves requests that need the model one at a time. Accounts,
  passwords and billing do not exist. None of this has been load-tested or audited.
* "Production-ready" here means: persistent, crash-safe storage, input validation, versioned and
  reversible model updates, an audit log, and tests. It does not mean audited for security or
  validated with users.

