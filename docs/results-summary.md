# Results at a glance

Measured on an Apple M4 Max against plans written before any test data was read
([PREREGISTRATION.md](PREREGISTRATION.md), [PREREGISTRATION_LARGE.md](PREREGISTRATION_LARGE.md)). Tasks are
generated; their text is human-written (Wikipedia revisions and NLI corpora). Brackets are 95% intervals over
episodes. The figures are copied from the generated [RESULTS.md](RESULTS.md), which is the authority; if they ever
disagree, it is right and this page is wrong.

The release ships the larger reader (ModernBERT-large, 395M). Rows marked *base* were measured with the 149M
reader and are kept as reported.

## The system

| | Result |
|---|---|
| Correct dispositions, shipped system | 91.55% on 1,000 standard tasks, 86.83% on 300 larger tasks, 92.00% on 1,000 small tasks |
| Larger reader against base (H10-H12, all supported) | +2.03 points [+1.06, +2.94] on standard tasks, +2.20 [+0.68, +3.61] on larger ones, +0.70 [-0.25, +1.61] on small ones; stale decisions 13.05% to 11.10% on standard tasks; update time 2.68 times longer |
| Exactness (H1, supported) | 0 of 195,434 cached values differed from an independent full rebuild, with either reader |
| Cost of an update (H2, supported; base) | 16 ms median on tasks of 5 requirements and 11 records (full re-reading: 167 ms); 36 ms on tasks of 13 requirements and 34 records (960 ms); 2.0% to 6.5% of the tokens of re-reading |
| Update time, shipped reader | 59 ms (standard) and 118 ms (larger) median on the Apple GPU; 192 ms and 576 ms on its CPU; 4.8 GB of memory |
| Against a dense reader with the same backbone, training and wrapper (H3 supported, H4 not; base) | learned refiner: +3.6 points [+2.1, +5.1] and 3.9 fewer stale decisions [1.4, 6.6] on standard tasks, at 6.5% of the tokens; on larger tasks +5.4 [+2.0, +8.7] but the stale-decision difference was not significant. Calibrated rules: +2.2 [+0.6, +3.9] and 3.6 fewer stale decisions [1.0, 6.2]; +5.5 [+2.3, +8.7] on larger tasks |
| Against a public NLI model in the same wrapper (base) | about +19 to +20 points; it reads 45% of unrelated passages as evidence, this reader 0.8% (large: 0.6%) |
| Records that say which requirement they concern | standard 91.55% to 92.47%, larger 86.83% to 90.69% (large reader) |
| Where the error comes from | with every reading correct, the exact core was right at every step of every test task, with either reader; the reader is correct on the record written for a requirement 88.9% (base) and 90.5% (large) of the time |
| More training of the same model | 200,000 more examples on the same corpora made it slightly worse on development data (VitaminC -0.48, MNLI -0.39 points); the larger encoder is what helped |

## Components

| | Result |
|---|---|
| Learned refiner (H5, supported narrowly) | small average gains; one of three seeds lost 6.8 points on larger tasks, all lose to calibrated rules on small ones; **the release ships calibrated rules** |
| Learning a policy instead of executing it (H6, not supported) | a recurrent network did as well as the executed policy |
| Self-evolution (H7, not supported) | exact corrections and the readiness bar work as specified; gated consolidation adopted no change it could verify as safe |
| Dependency proposer (H8, not supported) | within its miss bound on the tasks it was calibrated for (4.56%), just outside on larger ones (5.11%); off by default |
| Conflicts that follow from a consequence (H9, supported as predicted) | recognised 35% (base) and 45% (large) of the time, against 77% and 88.5% for direct conflicts |
| "The latest version" (X1) | read unsafely: 82% false "satisfied" when two records must be read together; name the specific version |
| Same-topic passages that settle nothing | read as decisive 22.3% (base) and 20.2% (large) of the time, against 0.8% and 0.6% for unrelated ones |

## The reader alone (test sets)

| | Large | Base |
|---|---:|---:|
| VitaminC | 91.53% (real 89.90%, synthetic 94.25%) | 90.20% (88.65%, 92.78%) |
| MultiNLI, mismatched | 90.21% | 88.56% |
| WANLI | 77.42% | 74.74% |
| SNLI (not trained on) | 84.39% | 80.33% |
| ANLI r1 / r2 / r3 (not trained on) | 53.70 / 38.10 / 36.00% | 46.40 / 33.30 / 34.75% |
| Unrelated passages read as decisive | 0.58% | 0.82% |
