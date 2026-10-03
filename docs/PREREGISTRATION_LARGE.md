# Pre-registered addendum: a larger encoder

Written 2026-10-02, before `answerdotai/ModernBERT-large` had been trained on anything here (its
download was in progress) and before any test split had been read by any model but the released
one. The SHA-256 of this file is recorded in `runs/prereg_large_hash.txt`. Deviations are logged
in `docs/DEVIATIONS.md`.

## 1. Why

Results section 8j (X5) found that every error of the released system on the test tasks is a
reading error of the edge model, that the exact parts are right whenever the readings are, and
that training the released model further on the same corpora does not help. The published gain
from the same family's larger encoder is MNLI 89.1 → 90.8 (Warner et al., 2024, Table 5), a
16% reduction in errors. Extrapolating through the replay in 8j, that would be worth about
+1.5 points of disposition accuracy on STANDARD and +1.5 to +2 on WIDE. This addendum tests that
extrapolation. Everything else about the system stays as released.

## 2. What is trained

One edge model, **E-L**, with `answerdotai/ModernBERT-large` (395M parameters) in place of
`answerdotai/ModernBERT-base`, by the released recipe (`configs/stage_a_large.yaml`): the same
corpora, mixture, 900,000 examples, sequence length, warm-up, decay and seed. Learning rate
3e-5 and the head's 5× ratio follow the paper's MNLI setting for the large model (Table 6).
Batch 32; if a 50-step pilot on training data shows that batch 32 does not fit in memory, batch
16 with the same number of examples, decided before the run starts and recorded here by the
deviations log. One seed, because a run takes most of a day on this machine; this is a stated
limitation, not a choice to be revisited after seeing the result.

Downstream of the edge model nothing is retrained except what the release recipe always fits
from the Stage B pool: the temperature and class bias of the calibrated rules. The learned
refiner is not retrained; the release ships calibrated rules and so does this comparison.

The development gate of `scripts/run_after_training.sh` applies unchanged (VitaminC dev ≥ 0.83,
MNLI dev ≥ 0.86, unrelated dev ≥ 0.99, no label-mix drift warnings). A model that fails it is
not evaluated on any test split and the addendum ends with "not evaluated".

## 3. Comparison

The released base system (**B1-soft**, calibrated rules on the base edge model) against the same
system with E-L. Same test sets as the main plan: 1,000 STANDARD and 300 WIDE episodes, SMALL
(X2) and the edge-model test sets. Differences are paired: by item for the edge model, by episode
for tasks. Uncertainty: 95% percentile bootstrap, 2,000 resamples, as in the main plan.

## 4. Hypotheses and decision rules

**H10 Reading.** E-L reads the test pool more accurately than the base model.
*Supported iff* the paired accuracy difference (large − base) has a 95% interval above 0 on both
`vitaminc.test` and `mnli_mm.test`. Reported alongside: `wanli.test`, SNLI, ANLI, the
false-decisive rate on `unrelated.test` (a rise of more than 0.5 points counts against shipping,
below), the real/synthetic split of VitaminC, and ECE after temperature scaling.

**H11 Decisions.** With calibrated rules, disposition accuracy on STANDARD is higher with E-L.
*Supported iff* the 95% interval of the paired difference (large − base) lies above 0 on STANDARD.
WIDE and SMALL are reported with the same statistic. Stated prediction: +1.5 points on STANDARD,
+1.5 to +2 on WIDE. Also reported: stale-decision rate, false-READY rate, status NLL, and the
replay of 8j (reading accuracy on linked and unrelated records) for E-L.

**H12 Cost.** Encoder tokens per update are identical by construction; wall-clock time is not.
*Supported iff* the median update time of E-L on STANDARD (canonical mode, Apple GPU, idle
machine) is at most 3.5× the base model's, measured in the same run. Absolute milliseconds on
the GPU and on the CPU are reported either way.

**Which model ships.** E-L replaces the base model in the release iff all of the following hold:
H11 supported on STANDARD; on WIDE and on SMALL the lower end of the accuracy interval is above
−1 point; the false-READY rate on STANDARD and WIDE is not significantly higher (interval of the
difference not entirely above 0); the false-decisive rate on `unrelated.test` is not more than
0.5 points higher; the median update time on STANDARD on the Apple GPU is at most 100 ms.
Otherwise the base model stays, and E-L is reported as a measured option. This rule is fixed
now; the numbers above are not adjusted after the results are seen.

## 5. Not under test

Anything about Jev or any hosted model (none is called); benefit to people (no study is run);
other encoders (none is trained in this addendum). A second seed, other learning rates or more
examples are not tried; if E-L fails, the conclusion is "this recipe at this size did not help",
not "larger models do not help".

## 6. Reported regardless of outcome

Every number in sections 3 and 4 for both models; the training cost in examples, time and peak
memory; the pilot's memory figure that decided the batch size; the dev-gate values; and the
fit check of section 8j for E-L.
