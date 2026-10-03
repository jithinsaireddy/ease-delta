# EASE-Delta: results

Generated 2026-09-30 06:51 UTC by `scripts/make_report.py` from `runs/results/*.json`. No figure in this document was typed by hand.

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

**Architecture claim (pre-registration, section 6): supported on one test set and not the other.** See sections 4 and 5.

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

400 conflict pairs and 2000 control pairs. Threshold on P(REFUTES) = 0.2071, set for 5.0% false positives on controls. **Accuracies reported in the stale paper (different task, different scoring).**

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

## 11. Limitations

These were written before the results were known and are not excuses for them.

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
* *Distractors are off-topic.* Records unrelated to a predicate come from different Wikipedia pages
  or different NLI prompts. In a real project the distractors are other messages about the same
  project. The only same-topic hard negatives are VitaminC evidence versions labelled NEI.
* *One record per question.* In the test sets each predicate is settled by single records. A
  predicate that needs two records read together ("approved version 4" and "the latest is
  version 5") cannot be settled by reading each record separately. What the system does then,
  leave it open or wrongly call it satisfied, was not assumed: it was measured in the exploratory
  experiment X1 (results, section 8b), together with the alternative of declaring the predicate
  `join=True` so that its records are read in one pass. X1 uses templated wording on one pattern.
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

* Numbers and dates are read as text by a neural encoder, not compared by code. Measured on
  VitaminC test (results, section 2), claims containing a numeric comparison were read at least
  as accurately as claims without numbers, so this is not the weakness it was expected to be on
  that corpus. VitaminC's comparisons follow a few patterns ("more than 540,000" against
  "545,500"); other formats, units, arithmetic and date reasoning were not tested and should be
  handled by exact code when they matter.
* Consequences that need world knowledge ("broke my leg" against "cycles to work") are not taught
  by any training corpus used here.

**What one run cannot show**

* The 149M-parameter edge model and the dense baseline were each trained once, with one seed,
  because each takes hours on this machine. Their run-to-run variation is unknown. The small
  aggregators were trained with three seeds.
* All timings are from one laptop with other applications running.

**Deployment**

* The HTTP API has no authentication and is meant for the local machine.
* "Production-ready" here means: persistent, crash-safe storage, input validation, versioned and
  reversible model updates, an audit log, and tests. It does not mean audited for security or
  validated with users.

