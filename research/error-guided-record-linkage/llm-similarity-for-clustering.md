# Can an LLM teach a better similarity function?

Research assessment and proposed pilot · 6 October 2026

**Yes. LLM-guided clustering is feasible, and several papers already study it directly.** The strongest approach uses a limited number of task-specific judgments to improve a reusable representation, inexpensive comparator, or clustering constraints. Whether that improves record linkage at an acceptable total cost remains an empirical question.

The research opportunity is to establish **when semantic supervision produces better linkage decisions on unseen entities**, or helps discover repairs that transfer to new cases. Using an LLM to supply clustering judgments, by itself, is established prior work. A selective literature review cannot establish that every narrower contribution has already been explored.

This supplement extends [Learning from linkage errors](README.md). It proposes a staged test; no new model inference or linkage benchmark was run for this assessment. The numerical bridge example below is a constructed counterexample, not an empirical result.

## What “LLM similarity” can mean

An LLM might recognize that two differently worded descriptions denote the same organization, or that two problematic record pairs involve transliteration. Those are useful capabilities if they add information beyond the existing comparisons. There are several ways to reuse that information:

| Route | What the expensive model supplies | What runs on the remaining data |
|---|---|---|
| Recordwise features | Names, roles, dates, aliases, or diagnostic tags | Ordinary comparisons and linkage |
| Task-conditioned embeddings | A vector representation for each record | Indexed retrieval and vector comparisons |
| Teacher-guided learning | Selected pair, triplet, or small-set judgments | A smaller learned comparator or encoder |
| Soft clustering constraints | Evidence that selected items belong together or apart | A clustering objective that can tolerate conflicting evidence |
| Direct set clustering | A partition of a small block of records | Further block processing and reconciliation |

The third route is particularly attractive when the same kind of semantic distinction recurs. A teacher answers a limited set of questions; a student learns a cheaper approximation. Learning an embedding is useful when retrieval or geometric clustering needs reusable vectors. A pair classifier can be a cheaper first test when the immediate task is deciding which candidate links to accept.

The query must define the target. “Which description resembles A more, B or C?” does not establish that either refers to A's entity. Relative comparisons need appropriate tie, neither, and insufficient-evidence options. A numerical answer between zero and one is an affinity score until it has been calibrated for a defined event and population.

This fits the Fellegi–Sunter perspective: improve the comparison evidence and assess its incremental value. Adding a learned score alongside the fields from which it was derived can double-count evidence if dependence is ignored. Its integration therefore needs fitting and calibration; changing the similarity function does not remove that responsibility. The [earlier review](README.md) connects this to Splink, Ather's selective LLM-assisted linkage framework, and LOTUS.

## Papers that directly answer the question

These papers are the most useful starting points. Their results concern their own datasets, supervision, models, and budgets.

| Paper | Mechanism and evidence | Boundary relevant to this project |
|---|---|---|
| **[ClusterLLM](https://aclanthology.org/2023.emnlp-main.858/)** — Zhang, Wang and Shang, EMNLP 2023 | Uses selected LLM triplet comparisons to fine-tune smaller embeddings, then pair judgments to choose clustering granularity. The experiments use 1,024 triplets per run. | Direct precedent for amortizing semantic judgments. Main embedding evaluation uses the true cluster count; a larger teacher is not uniformly better. |
| **[Large Language Models Enable Few-Shot Clustering](https://aclanthology.org/2024.tacl-1.18/)** — Viswanathan et al., TACL 2024 | Compares generated keyphrases, pair constraints, and post-clustering correction. Includes entity canonicalization as well as intent and topic tasks. | Keyphrase expansion wins on three of five datasets. Bank77 reassignment accuracy is only 41.7%; the method also obtains some demonstrations from test labels. Its full evaluation is not blind unsupervised learning. |
| **[Cequel](https://doi.org/10.1145/3746252.3761074)** — Wang et al., CIKM 2025 | Selects informative pairs or triangles under a token budget and uses the resulting constraints for clustering. [Inspected technical report, v2](https://arxiv.org/html/2504.15640v2). | Six text datasets contain 2,225–11,514 items and 5–89 categories. Efficient query selection does not establish efficient whole-pipeline identity clustering. |
| **[Optimized Algorithms for Text Clustering with LLM-Generated Constraints](https://ojs.aaai.org/index.php/AAAI/article/view/39379)** — Jia et al., AAAI 2026 | Elicits constraints over sets and combines constraint handling with local search. Reports more than twentyfold fewer queries than its pair-generation comparator. | Query count is not token or dollar cost. More constraints can introduce noise; quality often peaks or plateaus before all items are constrained. |
| **[In-context Clustering-based Entity Resolution with Large Language Models](https://doi.org/10.1145/3749170)** — Fu et al., PACMMOD 2025, associated with SIGMOD 2026 | LLM-CER directly clusters blocked record sets and merges their results. [Inspected June 2025 version](https://arxiv.org/html/2506.02509v1) evaluates nine datasets and a 50,000-record scale test. | This is direct identity-clustering prior art. On Music, slightly fewer calls than BoostER still cost more: $0.19 versus $0.02 in that experiment. |

Two related lines matter for choosing a baseline. **[INSTRUCTOR](https://aclanthology.org/2023.findings-acl.71/)** trains instruction-conditioned text embeddings, so a suitable existing encoder is a serious alternative to new LLM supervision. **[DistillER](https://arxiv.org/html/2602.05452v1)** studies entity-resolution distillation directly. Its 8B student retains the teacher's reported average F1 of 0.85 but has similar inference time; much faster RoBERTa and MiniLM students score 0.69 and 0.61. Reuse and cheap inference are separate achievements.

A June 2026 study, **[Labeling Training Data for Entity Matching Using Large Language Models](https://arxiv.org/html/2606.28823v1)**, further illustrates the trade-off. Its selected Ditto student trained on machine labels achieves 72.17 F1 on WDC against 87.20 for the strongest direct teacher, while students outperform teachers on its bibliographic benchmarks. Its lowest teacher-token-cost acquisition strategy retrains five Ditto models each round. Total cost can rank strategies differently from API expenditure.

## Define what belongs together

Three applications can use similar machinery while answering different questions:

| Application | Objects being grouped | Evidence of success |
|---|---|---|
| Identity linkage or deduplication | Records about people, households, businesses, or other defined units | Independent match truth and correct entity groups |
| Diagnostic organization | Candidate pairs and their observed comparison patterns | More effective investigation or repair selection |
| Semantic document organization | Texts sharing a specified topic or relationship | Independent labels for that particular relationship |

A Wikipedia page and an archive item may concern the same topic, describe the same historical event, or identify the same person. Each predicate supports a different join. A topic-clustering result does not establish identity accuracy.

For diagnostic grouping, suppose case A benefits from normalization, case B from both normalization and transliteration, and case C from transliteration. A and B share a useful repair; B and C do too; A and C need not. This relationship can overlap. A single partition would force a distinction that the repair problem does not require. Fuzzy membership allows graded assignments, but its values do not automatically become probabilities of an error or of repair success.

This motivates a competing diagnostic approach: predict a set of applicable repairs, or a benefit for each repair. With a frozen repair library and independent labels, define:

`benefit_r(case) = loss(baseline, case) − loss(repair_r, case)`

Similar benefit vectors provide a decision-based meaning of similarity. This is a proposed operational target for the experiment, not a claim that those benefits can be observed without labels. For graph-wide repairs, measure the loss change on the affected entity groups as well; an individual pair's apparent improvement can conceal damage elsewhere.

There is already nearby work on diagnostic categories. **[Peeters, Steiner and Bizer, EDBT 2025](https://arxiv.org/html/2310.11244v4)** generate error classes and assign known matching errors to multiple classes. Their diagnosis receives the correct and predicted labels. That supports label-informed organization, not discovery of actual errors from unlabeled patterns alone. Independent review remains necessary to distinguish a useful explanation from a plausible one.

## Match the algorithm to the information

Ordinary k-means and fuzzy c-means operate on vectors and centroid distances. Replacing a distance calculation with a verbal LLM score does not define how to compute a centroid or compare it with a record. A learned embedding supplies a coherent vector representation, but it is a new approximation to evaluate.

Soft constraints or graph methods are alternatives when the natural output is pair evidence. **[Bilenko, Basu and Mooney, ICML 2004](https://icml.cc/Conferences/2004/proceedings/papers/119.pdf)** combine metric learning with constraint penalties. **[Bansal, Blum and Chawla, 2004](https://www.cs.cmu.edu/~shuchi/papers/clusteringfull.pdf)** formulate correlation clustering around agreement with positive and negative edges. These approaches give inconsistent evidence an explicit objective; optimizing that objective still depends on the evidence being relevant and accurate.

Noisy-oracle theory provides conditional results, rather than an automatic guarantee for an LLM. **[Kuroki et al., NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/fb7214d2fdfd84165b08539d59c92e07-Paper-Conference.pdf)** assume independently sampled feedback around fixed edge affinities; their experiments use synthetic oracle noise. **[Pradhan et al., AISTATS 2026](https://proceedings.mlr.press/v300/pradhan26a.html)** assume an underlying metric and an expensive oracle that returns exact distances. Repeated prompting need not correct a persistent semantic misconception, and an LLM identity judgment is not an exact distance oracle.

True identity is transitive once the entity and time policy are defined. Estimated pair decisions can be inconsistent. General resemblance need not be transitive even under a valid metric: nearby neighbors can have distant endpoints. These distinctions matter when deciding whether to impose hard links, soft penalties, or overlapping groups.

## A single false bridge can dominate the outcome

Consider two real entities, each represented by 50 records. Accept all 2,450 true within-entity pairs and accidentally accept one cross-entity pair. Connected components then combine the entities into one group.

| Quantity | Before transitive closure | After connected-components clustering |
|---|---:|---:|
| Predicted matching pairs | 2,451 | 4,950 |
| Incorrect matching pairs | 1 | 2,500 |
| Pair precision | 99.959% | 49.495% |
| Pair recall | 100% | 100% |

This constructed example shows why excellent edge precision can coexist with damaging entity clusters. Joining components of sizes `a` and `b` creates `a × b` cross-component pairs. Source constraints or a more cautious graph rule can prevent some such merges, so the example is not a prediction for a particular implementation.

The concern also appears in empirical work. **[Graph Metrics-driven Record Cluster Repair meets LLM-based active learning](https://dbs.uni-leipzig.de/files/research/publications/2025-6/pdf/graph_metrics_driven_cluster_repair.pdf)** reports a Dexter setting where input F1 is 0.85, GPT-4-assisted repair gives 0.83, and a perfect-oracle version gives 0.88. In another setting, even the perfect-oracle repair falls below the input. Both the information source and the update procedure must earn their place.

## What efficiency means at 100 million records per side

The following are arithmetic illustrations, not measured capacity estimates:

| Operation or representation | Assumption | Size |
|---|---|---:|
| Unrestricted cross-file comparisons | 100M × 100M | 10 quadrillion pairs |
| Retained candidate pairs | 20 per left record | 2 billion pairs |
| Dense cross-file affinity matrix | One float32 per pair | 40 petabytes |
| Reusable vectors for both files | 200M × 384 dimensions × 2 bytes | 153.6 gigabytes |
| Selected teacher judgments | 5,000 prompts; 1,200 input and 20 output tokens each | 6M input and 100,000 output tokens |

Storage uses decimal units and excludes indexes, raw text, working memory and replicas. Real prompt budgets must include instructions, examples, retries and repeated checks. A recordwise generative feature extractor avoids quadratic calls, but 200 million generations can still be expensive.

The plausible workflow is inexpensive candidate construction, a capped teacher budget, reusable learned evidence, and sparse linkage decisions. Its total cost includes encoding, indexing, acquisition, training, candidate scoring, graph inference, calibration and review. The teacher budget can be small while the remaining two billion candidate decisions dominate runtime.

Identity clustering also has a different size regime from ordinary topic clustering. Many entities have only a few records, so the number of clusters can grow with the number of records. **[Microclustering research](https://jwmi.github.io/publications/flexible-models-for-microclustering.pdf)** makes this distinction explicit. A method with work proportional to records times clusters can become approximately quadratic in that regime. None of the inspected empirical studies establishes this entire workflow at 100 million records per source.

## The smallest experiment worth running

The first hypothesis should be narrow: **selected LLM judgments contain additional identity information, and an inexpensive student preserves enough of it to improve untouched entities.** Diagnostic grouping is a second hypothesis with a different outcome. This staged design is a proposed extension to the [existing protocol](experiment-protocol.md), not a replacement for it.

### 1. Establish information value and transfer

Use a standard benchmark with independent entity identifiers and measurable baseline errors. Keep the existing Splink FEBRL example as a structured-data control. Its 225-pair slice has no Splink errors, so it has no repair headroom. Select a larger, prespecified split before inspecting its errors; do not choose the final test set because the baseline happens to fail there. A multi-record benchmark is additionally needed to study large erroneous components.

Split whole entities into training, calibration and locked test sets before forming pairs. Keep prompts, selection rules, thresholds, candidate rules and graph rules fixed during the test. Source or time holdouts can test further transfer where real metadata permits. Standard public benchmarks provide reproducibility, but possible model exposure and imperfect labels remain part of their evidence boundary.

Choose a fixed training-query set with random coverage, uncertain cases and diverse patterns. Keep the student architecture, inputs, tuning effort and semantic target identical when comparing label sources. Compare:

1. Tuned Splink and an inexpensive semantic/lexical baseline, with all common labels disclosed.
2. A small student trained on teacher judgments for the selected training pairs.
3. The same student and training pairs with cheap baseline-derived answers, to check whether adaptation alone explains the gain.
4. The same student trained with independent correct labels on those pairs, as a diagnostic of attainable supervised performance at that sampling budget—not a guaranteed upper bound.

If the claimed mechanism is better geometric clustering, use a learned encoder and keep the downstream clustering algorithm identical across representation arms. A pair-classifier pilot answers the narrower information-transfer question; it cannot by itself establish a clustering improvement.

Audit teacher answers against independent truth, including a frozen test sample after model selection. Count that audit separately. Preserve abstentions and invalid outputs. The primary student result uses unqueried test entities. This prevents improvement on directly corrected examples from masquerading as learned generalization.

### 2. Compare ways to spend the budget

If the teacher and student pass that gate, compare deployment policies: baseline alone, student trained from development queries, selective direct teacher decisions on test candidates, and reusable extracted features where relevant. All policies see the same underlying records and permitted information; test truth stays hidden.

Their query locations differ deliberately. Training on development entities and directly correcting deployment entities cannot use exactly the same questions while preserving an unseen-entity claim. Match total operating budgets and predeclare the deployment volume and refresh horizon over which training is amortized. Freeze acquisition first; compare active versus random selection later if acquisition efficiency is the next claim.

For diagnostic grouping, run a separate comparison using the same fixed repair library and review budget: cheap diagnostic clustering, teacher-guided grouping, and direct overlapping repair tags or per-repair benefit prediction. Evaluate the selected repairs on untouched entities. Cluster names and purity are secondary; corrected errors and introduced errors determine operational value. Check whether any allowed repair can improve the evaluated cases using an explicitly inaccessible truth-informed diagnostic. If the library contains no helpful repair, that result cannot determine whether the representation is useful. When comparing label sources, retain the same target: a teacher's mechanism label and a measured repair benefit are different outcomes.

### 3. Decide with independent outcomes

Measure precision and recall at prespecified operating requirements, false merges, false splits, and damage to entity components. Count missed candidates separately: a fixed-candidate experiment estimates improvement conditional on those candidates. Recovery of excluded true matches requires a later retrieval experiment. Report uncertainty in a way that respects dependence among pairs sharing records or entities.

Record all teacher tokens, selection compute, training, indexing, inference and review. Report performance on queried and unqueried cases separately, and across relevant source and missingness groups. Order reversals and prompt perturbations diagnose instability; they cannot certify correctness. Teacher agreement, silhouette, attractive cluster names and model-reported confidence do not replace independent outcomes.

Stop or simplify when the teacher adds no useful information beyond the baseline, the student loses the gain, the result depends on unavailable labels or cluster counts, or improvements disappear at comparable total cost. A ceiling result or an imprecise estimate is inconclusive. A gain that survives these checks justifies a larger test; it does not yet establish national-scale performance.

## Decision

**Proceed with a small, controlled pilot of reusable semantic supervision.** Keep the strongest conventional and embedding baselines, isolate information value from generalization, and test the budget against selective direct inference. For the error-clustering project, compare grouping with overlapping repair predictions and judge both by held-out linkage improvement.

The potential contribution is a demonstrated condition under which a limited semantic-review budget buys reusable linkage accuracy. The literature supplies several mechanisms to try. The experiment must establish which one earns its cost for this task.
