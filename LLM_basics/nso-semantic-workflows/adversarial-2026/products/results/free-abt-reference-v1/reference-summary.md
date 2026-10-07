# Free Abt–Buy conventional reference

The fixed primary lexical and Fellegi–Sunter rules give the following complete target-set results. This is an intact-source conventional reference, not a semantic replication.

| Fixed rule | Exact sets / 1,081 | False edges | Missed edges | Automatic decisions |
|---|---:|---:|---:|---:|
| lexical_all_pairs | 341 | 13 | 741 | 1081 |
| splink_all_pairs | 310 | 455 | 611 | 1081 |
| lexical_same_candidates | 341 | 13 | 741 | 1081 |
| splink_same_candidates | 310 | 451 | 611 | 1081 |

The full prior grid below uses the fixed FS threshold 0.9 and all-pairs predictions. It changes prior odds at fixed fitted likelihoods; it does not refit or choose a preferred setting.

| Prior assumption | Exact sets | False edges | Missed edges |
|---|---:|---:|---:|
| assumed_recall=0.5 | 339 | 479 | 562 |
| assumed_recall=0.8 | 310 | 455 | 611 |
| assumed_recall=1.0 | 289 | 446 | 641 |
| links_per_left=0.1 | 251 | 288 | 736 |
| links_per_left=0.5 | 310 | 455 | 611 |
| links_per_left=1.0 | 344 | 491 | 549 |
| links_per_left=2.0 | 350 | 560 | 506 |

The publisher graph has 1097 edges, 0 NIL queries and 16 multiple-target queries. Uncertainty resamples 459 actual query families, using 2,000 replicates and seed 20261007. Full lexical and FS grids, candidate-restricted results, raw query outcomes and pointwise family intervals are saved alongside this table.

Candidates retain 1093 of 1097 reference edges and every target for 1077 of 1081 queries. No target was inserted from truth. All identity outcomes were scored after the question and prediction hashes were frozen. No new fitting, paid calls, source acquisition or Amazon evaluation outcomes were used.
