# Experimental protocol for error-guided linkage

Draft for methodological alignment · 6 October 2026 · No repair benchmark executed

## Question and scope

**Does diagnostic clustering help select reusable linkage repairs that reduce errors on unseen entities, beyond equally funded review strategies with the same repair capabilities?**

Stage one tests acquisition and selection from a bounded repair library. Later stages can test invention of new repairs, candidate-generation changes, native text, and operational cost. This separation makes a small first experiment interpretable without claiming that a synthetic benchmark answers the entire research question.

The target relation is person identity across two files. A true counterpart may be left unresolved; complete assignment is not forced. The standard FEBRL4 files have complete overlap by construction, so absent-counterpart behavior remains a separate challenge condition. [FEBRL documentation](https://recordlinkage.readthedocs.io/en/latest/ref-datasets.html)

## Stage zero: establish that the question is testable

Use the official Splink FEBRL4a/FEBRL4b data, 5,000 records in each file, with the pinned source hashes and licenses already recorded in the [tutorial manifest](../../LLM_basics/record-linkage-tutorial/data/manifest.json). Preserve the tutorial and its measured result. Its 225-pair slice has no Splink errors and is unsuitable as the repair experiment's test set.

Fit a separate Splink baseline using unlabeled estimation, suitable comparisons, term-frequency adjustment where justified, missing-value treatment, and documented blocking rules. EM convergence is a fitting result, not proof of correct probabilities. Predeclare the operating threshold or select it using a charged common calibration allocation. Record priors, estimation rules, model parameters, versions, and seeds. Do not remove effective comparisons to create headroom.

The tutorial estimated match parameters from labels on 3,486 entities. Reusing that fit would instead define a study of **limited additional review after supervised training**. That is a valid alternative, but its existing label access must be disclosed and shared fairly. It cannot be called an initially unlabeled baseline.

Use a non-test development audit to assess baseline errors and whether the admissible repair family could correct any of them. Some FEBRL true pairs contain insufficient observed evidence; retain these in the evaluation denominator. A strong baseline with little correctable error is a ceiling result, not a reason to weaken it. A null under negligible repair headroom is inconclusive about clustering's broader value. [Splink FEBRL example](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl4.html)

Freeze the experimental specification before accessing final-test outcomes. The development audit is part of the common label ledger; it cannot supply one method with free error labels.

## Data separation and label access

Split by true entity before forming pairs. Keep both endpoints of a pair inside the same partition. Fit normalizers, diagnostic scalers, clustering, and learned representations on development data only. Original FEBRL identifiers encode truth and must be replaced by opaque IDs; exclude truth, generator metadata, and split identifiers from features and prompts.

| Partition | Permitted use | Prohibited use |
|---|---|---|
| Baseline development and discovery pool | Unlabeled fitting; common seed; budgeted label acquisition | Reading unqueried truth to choose examples or repairs |
| Repair validation | A fixed, capped proposal-selection procedure | Unlimited error mining or free row-level feedback |
| Calibration | The prespecified score/threshold fitting procedure | Inventing or choosing additional repairs |
| Final evaluation | One evaluation after configurations are frozen | Selecting models, thresholds, features, or a favorable subset |

Choose partition sizes from available error counts and the minimum useful effect rather than arbitrarily dividing 5,000 entities into tiny samples. Keep a manifest of previous tutorial label exposure. A fresh shuffle does not erase prior exposure; use previously uninspected entities where feasible, and describe this public-data pilot as exploratory. Confirmation of a selected method requires a new untouched corpus or a prospectively locked evaluation.

Known benchmark truth simulates a faithful labeling oracle. It measures query efficiency, not the reliability, time, or cost of human adjudication. Do not interpret every absent annotation in a future sparse benchmark as a verified nonmatch. FEBRL's complete entity mapping supports that inference here; another corpus may not.

## Frozen diagnostic representation

Start with cheap, interpretable candidate-pair features:

- Comparison levels, field-level score contributions, missingness, and disagreement between exact, lexical, and phonetic evidence.
- Parse validity and normalization flags derived from the observed values.
- Candidate counts, top-score margins, retrieval routes/ranks, and permissible assignment conflicts.
- Source identifiers only when meaningful; FEBRL's file labels are not evidence of deployment-source transfer.

Use the same feature values and raw record context in every arm. Fit scaling and feature weights on development data. Treat raw-text embeddings, fuzzy c-means, and alternative distances as later ablations. Start with a simple clustering method and a small prespecified range of cluster counts; select hyperparameters without final-test labels.

A diagnostic cluster may mix matches, nonmatches, errors, and correct decisions. The procedure must choose several members, including boundary cases, rather than propagate a centroid's label. Record cluster size and selection coverage, but do not optimize silhouette score as if it were linkage utility.

## Stage one: four selectors with identical repair powers

All arms receive the same initial seed labels, observation features, candidate set, repair library, update procedure, validation interface, and resource ceilings. Subsequent acquired pairs differ by policy; that difference is the treatment.

| Arm | Acquisition policy | Purpose |
|---|---|---|
| A | Random sampling within fixed score strata | Inexpensive review baseline with coverage beyond uncertainty |
| B | Uncertainty plus noncluster diversity and random coverage | Main alternative that already avoids redundant review |
| C | Diagnostic-cluster coverage plus the same random quota | Proposed clustering contribution |
| D | Regularized supervised prediction of baseline error risk, plus the same random quota | Tests whether labels make direct risk prediction more useful than clustering |

Noncluster diversity can use farthest-first selection on the same standardized diagnostics. For arm D, predict errors of the same frozen baseline. If the common seed contains no errors, use a prespecified fallback to stratified random acquisition; do not search for a favorable seed.

Use a constant query unit, initially one candidate-pair label. A record-level query that reveals many links has a different information yield and belongs in a separately costed experiment. Set a small number of fixed acquisition budgets; **100 common seed labels plus 200 additional pair queries per arm** is an illustrative feasibility budget, not a power calculation. Repair validation and calibration allocations are additional disclosed costs, available equally to each arm. Final-evaluation truth is a shared experimental cost, inaccessible to the methods.

For the cleanest first comparison, freeze acquisition scores and clustering after the common seed and acquire one batch. Use five prespecified random seeds, paired across arms. Repeated seeds assess selection variability within this corpus; they are not five independent datasets. Adaptive multi-batch acquisition is a later policy experiment because repairs can then change subsequent acquisition histories.

Replay each acquired sample through two update branches:

1. **Ordinary update:** one common, regularized classifier or calibration update, using the original comparison representation.
2. **Reusable repair plus update:** choose at most a fixed number of additional transformations or comparisons from the same predeclared library, then apply the identical update procedure.

Keep the frozen baseline as a reference. The paired branches show whether a selector helps ordinary learning, reusable repair, or both. Include the development-only cost of defining the repair library in the study ledger.

A narrow first library might offer a small number of additional derived comparisons while preserving original values. It must be specified before acquisition, with executable applicability conditions and counterexamples. Every arm can select any library entry. This tests **repair selection and scope**, not automatic invention of previously unknown mechanisms. It also avoids a human learning a repair in one arm and carrying that knowledge into the next.

Give each arm the same capped validation interface and maximum number of proposals, retaining rejected proposals. A validation label revealed during inspection counts as newly acquired information. Calibration follows repair selection and cannot become another free discovery set. Prespecified repeated validation can still overfit; the final holdout is essential.

## Outcomes and decision rule

Select one primary decision loss before unblinding:

```text
L = c_FP × number of false links + c_FN × number of missed true links
```

Fix the population and units of that loss, and count unresolved true links as misses for recall while also reporting abstention separately. Choose false-positive and false-negative costs for the intended use; there is no universal official-statistics ratio. Report unweighted counts as well, so the result can be interpreted under other costs.

Primary comparison: held-out reduction in this loss for arm C against the strongest prespecified noncluster arm under the same resource ceilings. Do not select the comparator using the final test. Either use development data to nominate it or retain all prespecified pairwise comparisons with multiplicity-aware uncertainty. Report the acquisition-by-repair interaction as a secondary diagnostic, rather than interpreting a favorable row in isolation.

Also report:

- True positives, false positives, false negatives, precision, recall, and abstentions on the complete target population.
- Candidate recall and conditional matching recall; omitted true candidates remain end-to-end misses.
- Errors corrected and errors introduced, both inside and outside each repair's applicability region.
- Improvement on unqueried evaluation entities, excluding the direct benefit of adjudicating discovery pairs.
- Calibration, candidate counts, cluster/assignment errors where applicable, and results by available meaningful strata.
- Oracle queries, computation, peak memory, failed proposals, and measured inference tokens. Human minutes enter the ledger only when actually observed.

Use entity-aware paired uncertainty calculations that reflect shared records. Report the effective evaluation denominators and detectable effect; do not infer production precision from a small synthetic test. Source/time transfer is outside stage one's evidence boundary.

Set the smallest useful loss reduction and maximum tolerated increase in false links before results. Continue only when the estimated gain and its uncertainty support further investment. High error yield alone is not success: a selector can find fewer but more repairable errors. Conversely, finding many errors with no safe transfer does not justify the method.

Useful negative controls are random reassignment of diagnostic groups, source/score-only grouping, and the same repair library offered to every selector. An oracle that sees all error labels can estimate development repair headroom, but it must be reported as infeasible and cannot guide the real arms for free.

## Stages two and three

**Stage two: repair invention and native evidence.** Choose a standard corpus whose target is entity identity and whose raw records and label completeness can be audited. Preserve structured and native-text conditions. Compare bounded human or LLM proposals with ordinary updates, selective LLM pair review, and recordwise extraction followed by Splink. Isolate proposal sessions, use the same prompts/context/tool access, cap attempts, and count authoring, failed proposals, review, validation, and inference. Test a held-out source or period only where those metadata actually exist. Controlled corruptions are a separate mechanism experiment with an unmodified control.

**Stage three: candidate repair and scale.** Permit changes to retrieval under identical rules for each arm. Add record-level diagnostics for no-candidate cases and candidates found only by independent retrieval. Evaluate complete known links on benchmarks; in deployment, use an entity-centric probability audit with a search beyond the production blocker. Charge index rebuilds, candidate growth, forward/reverse neighborhood changes, rescoring, and assignment updates. Measure scaling over progressively larger corpora before making throughput claims. [Binette et al., §3.2](https://arxiv.org/html/2404.05622v1)

With LLM signals, compare pooled Bayesian updating, a regularized joint model conditional on baseline diagnostics, and the baseline alone on the same routed population. Use independent identity labels as the final target. Agreement with the LLM reference measures a different outcome.

## Interpretation of possible results

| Result | Supported conclusion |
|---|---|
| Too few baseline errors or no admissible corrective change | The selected pilot has insufficient headroom |
| Clusters improve ordinary updates, with no additional repair benefit | Acquisition helps; explicit repair is not yet the explanation |
| Every selector benefits similarly from the repair library | The repair capability is useful; clustering adds little at this budget |
| Cluster-guided repairs transfer at lower measured cost | Evidence for the tested diagnostic representation and policy |
| Gains vanish on independent entities or new sources | The proposed repair does not support the intended transfer claim |
| Candidate growth or new false links offsets corrections | The end-to-end intervention does not meet the chosen objective |

This protocol is ready for a decision about the target loss, label budget, and allowed repair family. Those choices define the scientific question and should be agreed before implementation and final evaluation.
