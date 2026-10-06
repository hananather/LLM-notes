# Learning from linkage errors

Research review and proposed experiment · 6 October 2026

The [record-linkage handbook](../../LLM_basics/record-linkage-with-semantic-operators.ipynb)
brings this review together with the benchmark, worked examples and diagrams.
This folder retains the supporting research notes and experiment protocol.

**Clustering can help improve record linkage when it helps discover a correction that transfers to unseen records.** Its value is in directing investigation and defining where a repair applies. Whether that adds anything beyond uncertainty sampling, diverse review, or a supervised error-risk model is an empirical question.

The useful research objective is to turn a limited review budget into reusable improvements: a better parser, an additional comparison, a corrected extraction rule, a new candidate-retrieval route, or a better allocation of expensive inference. The strongest test gives competing review strategies the same ability to make those improvements.

This review connects that objective to Fellegi–Sunter linkage, Splink, LLM-assisted linkage, and LOTUS. The accompanying [experimental protocol](experiment-protocol.md) specifies a staged pilot; the [reading guide](sources.md) identifies the closest primary literature. The new repair experiment has **not been run**.

The supplement [Can an LLM teach a better similarity function?](llm-similarity-for-clustering.md) examines direct clustering precedents, efficient teacher-guided learning, and a controlled test of transfer to unseen entities.

## How the approaches fit together

Fellegi–Sunter linkage weighs comparison evidence according to how often it occurs among matches and nonmatches. String distances are possible ingredients; the framework also accommodates exact agreement, missingness, derived features, and other comparison functions. Splink makes those comparisons and their levels explicit. A semantic feature can therefore strengthen a Fellegi–Sunter model when its additional information and dependence are handled appropriately. [Splink comparison documentation](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparisons_and_comparison_levels.html)

Hanan Ather's *LLM-assisted record linkage: A framework for official statistics* already combines a classical linkage score with selective LLM review and Bayesian updating using estimated LLM error rates. That selective combination is an established starting point for this study. The public article appeared in the **Statistical Journal of the IAOS**, volume 42, issue 1, pages 211–217, in 2026. [Ather, 2026](https://doi.org/10.1177/18747655261422068)

LOTUS provides semantic operators, including joins, and execution optimizations. Its operator interface can express a matching predicate, but the predicate still needs to define the relation correctly. Its approximation guarantees concern agreement with a reference operator; that reference's answers can differ from independent entity truth. The paper's FEVER pipeline maps claims to search queries, retrieves Wikipedia passages, and filters claims using that evidence. Its BioDEX pipeline joins patient articles to drug-reaction labels. These tasks demonstrate semantic relations whose labels differ from same-entity identity. [Patel et al., 2025, §§2–4 and §§5.1–5.2](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf)

| Component | Role in a linkage workflow | Question for this study |
|---|---|---|
| Splink / Fellegi–Sunter | Combine inexpensive comparison evidence | Which recurring deficiencies remain after a well-configured baseline? |
| LLM comparison or extraction | Interpret evidence that simpler comparisons miss | Does the extra evidence improve decisions enough to justify its cost? |
| LOTUS | Express and execute semantic operations | Which operations are useful, and what does their reference predicate mean? |
| Diagnostic clustering | Organize similar patterns of potential failure | Does this organization lead to better repairs per unit of effort? |
| Independent evaluation | Measure the resulting linkage against its target | Do improvements transfer, including introduced errors and missed candidates? |

The hypothesis that semantic methods have more to offer on unstructured data is reasonable, with a more precise mechanism: **they may recover identifying evidence that the current representation discards**. Clean column names alone do not ensure easy linkage. Structured cells can contain aliases, multilingual names, historical changes, or ambiguous organizational roles. Conversely, text may contain enough extractable identifiers to support an inexpensive structured linkage pipeline. These are hypotheses to distinguish experimentally, not a ranking implied by the file format.

There are three useful semantic interventions to compare: direct pair adjudication, recordwise extraction followed by ordinary linkage, and an additional semantic comparison within the baseline. The second can reuse an extracted value across many pairs; the third can preserve the existing decision framework. Their relative cost depends on how often records and difficult pairs recur.

The existing [tutorial](../../LLM_basics/record-linkage-with-semantic-operators.ipynb) is a small structured-data check. On its fixed 225-pair slice of Splink's FEBRL data, Splink found all 10 true links; LOTUS with the recorded LLM configuration found eight. Neither produced a false link. That is evidence about those configurations and that slice. Because the slice contains no baseline errors, it cannot demonstrate a benefit from repairing them. [Local execution record](../../LLM_basics/record-linkage-tutorial/results/verification.json)

## What can clustering tell us without labels?

A **diagnostic cluster** groups candidate pairs with similar observed properties: field agreements, missing values, score contributions, parsing flags, retrieval routes, or competing candidates. Those pairs can describe entirely unrelated people or businesses. An **entity cluster**, in contrast, is a proposed set of records belonging to one real-world entity. The two objects serve different purposes.

Suppose several pairs disagree on birth date but agree on other fields. Grouping them can reveal a pattern worth investigating. It cannot establish whether the dates were transposed, the identities differ, or both phenomena occur in the group. A shared symptom does not establish a shared cause or a shared match label.

Let `z` be the observed diagnostic features, `Y` the true match status, and `E` an indicator that the current linkage decision is wrong. An unsupervised cluster `C = g(z)` is observable. The error rate `P(E = 1 | C)` depends on information about `Y`. Two different underlying identities can sometimes produce the same observed records and the same clusters. In that situation, geometry cannot choose the correct identity.

| Available information | What can be learned | How it could improve linkage |
|---|---|---|
| No labels; observable diagnostics only | Repeated formats, ambiguity, missingness, disagreement, drift | Select cases to investigate; propose a parser or retrieval change |
| A small set of faithful labels | Which suspicions are actual errors; examples and counterexamples | Learn a cheap error-risk model; test a reusable repair |
| An identifiable statistical model | Model-based match probabilities or error rates under its assumptions | Estimate parameters and allocate decisions or review |
| An independent probability audit | Error totals or rates for a defined population, with uncertainty | Decide whether an adopted repair improved the system |

Unlabeled estimation is possible under suitable assumptions. Winkler describes estimating linkage parameters without training labels and modeling dependencies. Dasylva, Goussanou, and Nambeu estimate linkage accuracy without clerical review through models of link counts, under explicit source and linkage assumptions. Neither result makes arbitrary cluster membership a certificate of error. The defensible distinction is between **observable patterns** and **error estimates supported by labels or an identified model**. [Winkler, 2002](https://www.census.gov/content/dam/Census/library/working-papers/2002/adrm/rrs2002-05.pdf); [Dasylva et al., 2024](https://www150.statcan.gc.ca/n1/pub/12-001-x/2024002/article/00007-eng.pdf)

Fuzzy c-means does not remove this distinction. Its memberships describe relative geometric membership under a chosen distance and fuzzifier. A membership of 0.8 does not mean an 80% probability of a linkage error. Soft membership might support routing to several specialist models, but their predictions and the routing policy still need evaluation. [Bezdek, Ehrlich, and Full, 1984](https://doi.org/10.1016/0098-3004(84)90020-7)

The most promising setting is therefore **initially unlabeled data with a limited opportunity to obtain informative review**, rather than a promise that no information about truth will ever be needed. Where review remains inconclusive, retain that uncertainty instead of turning an LLM answer into a gold label.

## Turning a pattern into a correction

Consider a hypothetical provider whose date strings sometimes follow day/month/year while another uses month/day/year. A diagnostic group contains pairs with disagreement on parsed dates, strong name agreement, and the same source combination. A reviewer investigates several examples and nearby counterexamples. Source documentation or unambiguous dates support a parsing defect. A source-specific parser is then tested on fresh records, including nonmatches that the change might accidentally join.

The repair can benefit thousands of records, but that benefit comes from correcting the parser. Clustering earns credit only if it helps discover or delimit the correction more efficiently than other review strategies. If the strings remain intrinsically ambiguous, forcing a date interpretation can introduce errors; preserving alternative interpretations may be the better feature.

The same reasoning supports several interventions:

| Observed pattern | Candidate intervention | Evidence needed before adopting it |
|---|---|---|
| Repeated formatting or parsing anomaly | Preserve raw values and add a validated normalization or parser | Improvement on new examples and nearby nonmatches |
| Aliases, transliteration, or name-order variation | Add an alias, phonetic, transliteration, or semantic comparison | Incremental value beyond existing comparisons; false-link effects |
| Long text hides identifying facts | Extract names, dates, locations, and roles once per record | Source-supported extraction accuracy and downstream linkage gain |
| True counterparts absent from candidates | Add or modify a blocking/retrieval route | Higher candidate recall, with added candidate and scoring costs |
| Several plausible counterparts | Improve comparisons, calibrate ambiguity, or use valid assignment constraints | Correct relation cardinality; effects on final assignments |
| A bounded region defeats cheap comparisons | Route selectively to an LLM or specialist reviewer | Conditional error rates, cost, and performance on that region |
| Many examples of a recurring failure | Train a cheap classifier or distill validated weak rules | Transfer to unseen entities and sources |
| Identifying evidence is absent | Abstain or acquire additional evidence | A defensible unresolved outcome |

An LLM can help summarize reviewed examples, propose a falsifiable mechanism, generate an extraction rule, or classify a difficult pair. Each role has a different validation target. A convincing explanation is a hypothesis. A useful repair has an executable applicability rule, preserved evidence, an independently measured benefit, and a way to reverse it.

One practical loop is:

1. Generate and score candidates with the baseline; retain cheap diagnostic summaries.
2. Select a diverse set of suspicions, including confident decisions and randomly sampled cases.
3. Review representatives, boundary cases, and counterexamples; retain unresolved judgments.
4. Specify the suspected mechanism and one bounded repair before testing it.
5. Measure corrected **and newly introduced** errors on separate validation records.
6. Freeze the accepted change, recalibrate if necessary, and evaluate on untouched entities.

Labels should not spread automatically across a diagnostic cluster. A date-parsing defect can affect both true matches and nonmatches. Likewise, a repair may have an applicability condition that cuts across several clusters.

## What the literature already establishes

The ingredients have substantial prior art. ALIAS explicitly allowed users inspecting discrepancies to modify or add similarity functions in 2002. Snorkel connects inspected errors to changes in reusable labeling functions. These precedents support the workflow while narrowing any novelty claim. [Sarawagi and Bhamidipaty, 2002, §2](https://www.cse.iitb.ac.in/~sunita/papers/kdd02.pdf); [Ratner et al., 2017, Appendix C](https://arxiv.org/pdf/1711.10160)

Several closer methods sharpen the comparison:

| Prior work | Closest overlap | Boundary that matters here |
|---|---|---|
| [Domino, 2022](https://arxiv.org/html/2203.14960v2) | Finds systematic failure slices using embeddings, predictions, and outcome labels | Unknown subgroup membership does not mean the method has no outcome labels |
| [D-diversity, 2026](https://doi.org/10.1109/ACCESS.2026.3708863) | Clusters record-pair embedding differences and labels representatives for LLM demonstrations | Direct precedent for pair-pattern diversity; does not establish reusable repair transfer |
| [GraphCR, 2025](https://doi.org/10.1145/3735511) | Active labeling and graph features repair proposed entity components | Entity-component repair differs from grouping failure signatures across unrelated entities |
| [FDJ, 2025 preprint](https://arxiv.org/html/2512.05399v1) | Uses difficult examples to generate or repair reusable feature extractors | Guarantees use a reference LLM predicate; independent identity truth remains a separate target |
| [Lam et al., 2026 preprint](https://arxiv.org/abs/2608.01401) | Allocates clerical review using match weights, comparison patterns, and ambiguity | Sampling for estimation differs from demonstrating repair benefit |

Adverse results matter. GraphCR's Table 4 contains a setting where input F1 is 0.89 and even perfect-oracle repair lowers it to 0.835. More accurate review labels do not ensure a beneficial downstream update. The decision rule and graph operation matter too. [Christen et al., 2025, Table 4](https://dbs.uni-leipzig.de/files/research/publications/2025-6/pdf/graph_metrics_driven_cluster_repair.pdf)

The research question should consequently be: **does similarity in diagnostic features predict shared, transferable repair benefit?** Ordinary active learning already improves unreviewed cases by learning from labels. The proposed study needs to show additional value from organizing diagnosis or repair scope. No claim of novelty is established by this review.

## Combining LLM evidence with an existing score

An LLM may recognize an alias or relationship that a simple string comparator discarded. It can add useful evidence even though it reads the same records. However, the reliability of its answer can depend on the baseline score and the reason the pair was routed to it.

Let `S` denote the baseline representation, `D` the LLM decision, and `R` the routing event. The conditional odds identity is:

```text
odds(Y = 1 | S, D, R)
  = odds(Y = 1 | S, R)
    × P(D | Y = 1, S, R) / P(D | Y = 0, S, R)
```

A global sensitivity/specificity adjustment approximates the conditional likelihood ratio with pooled quantities. Ather's §2.1.4 makes the corresponding independence simplification explicit. Diagnostic groups could reveal where that approximation changes, but fitting a separate model to every small cluster could be unstable. Compare pooled updating with a regularized joint model on separate calibration data, and judge both on untouched cases. This is a proposed empirical extension, not an assertion that the existing update is universally invalid. [Ather, 2026, §2.1.4](https://doi.org/10.1177/18747655261422068)

The cheaper alternative deserves equal attention: add a reusable extracted or semantic feature to the baseline, then recalibrate. Cluster-specific LLM routing is useful only if it beats that alternative at the intended operating point.

## What changes at 100 million records per source

The full cross-product contains **10 quadrillion pairs: 10^16**. Clustering cannot make materializing or asking an LLM about that product affordable. A plausible architecture retrieves a sparse candidate set, scores it cheaply, diagnoses sampled failures, and applies validated changes to the affected parts of the pipeline.

First define the relation. Linking an archive document to a Wikipedia entity can mean that the document mentions the entity, concerns an event involving it, or shares a topic. Those are different labels and different cardinalities. An archive can contain many documents about one person. A one-to-one identity constraint would then discard valid links. Document-to-entity resolution may also require extracting several mentions from each document before linkage.

The following calculations are illustrative, in decimal units; they are not measured throughput or cost estimates.

| Assumption | Consequence |
|---|---:|
| 100M left records × 20 retrieved candidates each | 2 billion candidate edges before unions or deduplication |
| 24 bytes per edge: two 64-bit IDs, score, flags | 48 GB of raw edge payload |
| 32 additional float32 diagnostic features per edge | 256 GB of additional numeric payload |
| 384-dimensional float16 embeddings for 200M records | 153.6 GB of vector payload before index overhead |
| LLM review of 0.01% of 2B candidates | 200,000 pair evaluations |
| Extraction from all 200M records at 500 input + 50 output tokens | 100B input + 10B output tokens |

Storage figures exclude replication, indexes, text, shuffle space, and temporary copies. Linear extraction can still dominate cost. Selective extraction and caching by record content and extractor version should therefore compete against selective pair adjudication; neither is automatically cheaper.

Candidate generation can combine lexical blocking and identity-oriented embedding retrieval. DeepBlocker reports complementary neural and non-neural blocking. Approximate nearest-neighbor systems such as Faiss provide relevant infrastructure, but recovering vector neighbors is different from recovering true entity counterparts. [Thirumuruganathan et al., 2021](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf); [Douze et al., inspected v4](https://arxiv.org/html/2401.08281v4)

A useful warning comes from SC-Block: a 99.5% validation recall target corresponded to 89.5% test candidate recall on its largest benchmark. That is a result for that dataset and configuration, not a general failure rate. It demonstrates why candidate recall must be measured separately after tuning. [Brinkmann, Shraga, and Bizer, 2024, Tables 4, 8–9](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf)

When final links are drawn from the candidate set, pair recall decomposes exactly as:

```text
end-to-end recall
  = candidate recall × final recall conditional on being a candidate
```

For example, 98% candidate recall and 97% conditional final recall yield 95.06% end-to-end recall. Improving scores cannot recover true links that were never candidates. If a later stage expands the graph, the candidate set in this identity must include those added pairs.

At this scale, retain whole-population counters and stratified diagnostic samples, rather than a dense matrix of every pair's features. A repair may require re-extraction, index updates, new forward and reverse candidate neighborhoods, rescoring, and revised assignment. Global weight or threshold changes can require broad recomputation. Those costs belong to the repair.

The reviewed semantic-join and graph papers do not establish this 100M-by-100M workflow. SemJoin's reported workloads are small semantic-relation tasks; Alper's inspected experiments reach 50,797 records. Their mechanisms are relevant research inputs, with scale validation still to be done. [SemJoin, 2026 preprint, Table 1](https://arxiv.org/html/2606.29532v1); [Alper, inspected 2026 technical report, Table 1](https://arxiv.org/html/2605.25814v1)

## Evaluation must reach beyond the reviewed pairs

There are two different uses of labels: discovering repairs and measuring a frozen system. Discovery intentionally favors interesting cases. Its error fraction is not the population error rate. Keep a separate probability audit, or an untouched benchmark with complete relevant truth, for the latter purpose.

Auditing only production candidates misses the blocker's failures. In an operational study, sample records or entities, including those with no candidates, and investigate counterparts beyond the production search. Binette et al. formalize an entity-centric approach based on sampled records and recovery of their true entity clusters. The credibility of recall estimates still depends on how completely that recovery is performed. [Binette et al., 2024, §3.2](https://arxiv.org/html/2404.05622v1)

Uniformly sampling the enormous pair product is usually inefficient. Under an illustrative one-to-one scenario with 100M true links among 10^16 pairs, a million uniformly sampled pairs contain only **0.01 expected true links**. Sampling anchor records and searching for their counterparts can concentrate the audit on informative units without pretending the production blocker is complete.

Precision claims also need adequate denominators. Under independent representative Bernoulli sampling, zero false links among 300 audited accepted links gives an approximate 95% upper false-link bound of 1%; 3,000 gives about 0.1%. Repeated entities, unequal sampling weights, incomplete adjudication, or adaptive evaluation change that calculation. A handful of clean cluster representatives cannot certify a national linkage system.

## The experiment worth starting

Start with a **controlled feasibility study on standard Splink data**, hiding the benchmark labels from the selection and repair procedures. Preserve the existing tutorial. Use a fresh baseline estimated without truth labels if the question is initially unlabeled linkage; the tutorial's supervised parameter estimation is a different starting condition.

The first comparison should keep candidate generation fixed and allow the same bounded repairs in every arm. Compare diagnostic clusters against stratified random review, uncertainty plus diversity, and a supervised error-risk selector trained on the same seed labels. Use the same diagnostics and raw context. Replay a shared update-only procedure on each acquired label set to distinguish better example selection from better repair capability. This pilot tests selection and scoping of predefined repairs; discovering new repair mechanisms is a later study.

FEBRL is suitable for checking that machinery. Its synthetic corruptions and known links do not establish the prevalence of repairable defects in administrative sources, source/time transfer, or performance on native documents. If the development pool has too few consequential errors, stop the feasibility claim there; do not weaken Splink to create a favorable demonstration.

Advance only if a repair benefits untouched entities and clustering improves the cost–quality frontier against the strongest noncluster alternative. The next stages can test candidate-generation repairs and a native text identity task. A topical semantic join should remain a separately labeled task.

The distinctive positive finding would be that **diagnostic organization helps discover transferable corrections earlier, safely, and at lower total cost**. An equally useful negative finding would be that simple stratification, a learned error-risk score, or existing active learning captures the same benefit. The [protocol](experiment-protocol.md) makes those alternatives explicit before results exist.
