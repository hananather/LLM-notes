# Reading guide and evidence map

Primary-source review · 6 October 2026

This is a targeted literature review of diagnosis, active learning, repair, semantic joins, and evaluation. It is not a systematic review or proof of novelty. Published papers and inspected preprint versions are distinguished below. Results cited from these studies have not been independently reproduced here.

Start with Ather, D-diversity, FDJ, GraphCR, and the two audit papers by Dasylva and Binette. Together they address the existing hybrid linkage method, the closest clustering overlap, reusable extraction repairs, graph repair, and evaluation without assuming the candidate set is complete.

## Linkage foundations and the semantic connection

1. **Fellegi, I. P., and Sunter, A. B. (1969). “A Theory for Record Linkage.” JASA 64(328), 1183–1210.** [DOI](https://doi.org/10.1080/01621459.1969.10501049). Foundational likelihood-ratio decision framework. A comparison representation and its probability model are separate choices; string distance alone does not define the framework.

2. **Winkler, W. E. (2002). “Methods for Record Linkage and Bayesian Networks.” US Census Bureau RRS2002-05.** [Official paper](https://www.census.gov/content/dam/Census/library/working-papers/2002/adrm/rrs2002-05.pdf). Introduction and model discussion cover unlabeled estimation and dependence. Successful estimation under a model is distinct from validating it on a new population.

3. **Ather, H. (2026). “LLM-assisted record linkage: A framework for official statistics.” Statistical Journal of the IAOS 42(1), 211–217.** [DOI](https://doi.org/10.1177/18747655261422068). §§2.1.3–2.2 describe selective LLM routing and updating; §2.1.4 states the simplifying dependence assumption. This study builds on the existing hybrid method. Public published version used; no private manuscript is redistributed.

4. **Patel et al. (2025). “Semantic Operators and Their Optimization: Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS.” PVLDB 18(11), 4171–4184.** [Published PDF](https://www.vldb.org/pvldb/vol18/p4171-patel.pdf). §§2–4 distinguish semantic operator definitions from optimized execution and reference agreement. Read the predicate and reference carefully before translating an accuracy guarantee to entity linkage.

5. **Splink documentation. “Comparisons and comparison levels”; FEBRL4 example.** [Comparisons](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/comparisons_and_comparison_levels.html); [FEBRL example](https://moj-analytical-services.github.io/splink/demos/examples/duckdb/febrl4.html); [dataset repository](https://github.com/moj-analytical-services/splink_datasets). Implementation sources for the baseline and standard synthetic corpus. The existing tutorial records pinned data and package versions separately.

## Active learning, diagnostic groups, and reusable repair

6. **Sarawagi, S., and Bhamidipaty, A. (2002). “Interactive Deduplication using Active Learning.” KDD, 269–278.** [DOI](https://doi.org/10.1145/775047.775087); [author PDF](https://www.cse.iitb.ac.in/~sunita/papers/kdd02.pdf). ALIAS §2 already connects inspection of discrepancies to modified or new similarity functions. It is direct prior art for interactive comparator repair.

7. **Meduri, V. V., Popa, L., Sen, P., and Sarwat, M. (2020). “A Comprehensive Benchmark Framework for Active Learning Methods in Entity Matching.” SIGMOD, 1133–1147.** [Full paper](https://arxiv.org/pdf/2003.13114). §§3–4 define the evaluation and selectors. Its progressive evaluation includes labeled pairs; those results should not be presented as untouched, entity-disjoint transfer.

8. **Jain, A., Sarawagi, S., and Sen, P. “Deep Indexed Active Learning for Matching Heterogeneous Entity Representations.” PVLDB 15(1), 31–45.** [Published paper](https://www.vldb.org/pvldb/vol15/p31-jain.pdf). DIAL combines blocker and matcher learning. §4.7/Table 8 compare uncertainty, committee, partition, and diversity-based selectors. Diversity is already a substantive alternative to uncertainty alone. The printed reference uses 2022; the work first appeared in 2021.

9. **Eyuboglu et al. (2022). “Domino: Discovering Systematic Errors with Cross-Modal Embeddings.” ICLR.** [Inspected full text](https://arxiv.org/html/2203.14960v2). §§3 and 5.2 use outcome labels and predictions in error-aware slice discovery. Its diagnostic objective is relevant, but the reported application is not record linkage with unknown outcomes.

10. **Christen, V., Obraczka, D., Hofer, M., Franke, M., and Rahm, E. (2025). “Graph Metrics-driven Record Cluster Repair meets LLM-based active learning.” ACM JDIQ 17(2), Article 7.** [DOI](https://doi.org/10.1145/3735511); [full paper](https://dbs.uni-leipzig.de/files/research/publications/2025-6/pdf/graph_metrics_driven_cluster_repair.pdf). §5.3 examines acquisition; §5.4/Table 4 compares perfect and LLM oracles, including harmful repairs. This is entity-graph repair, not clustering unrelated error signatures. Earlier [GraphCR version](https://arxiv.org/html/2401.14992v1).

11. **Akashi, K., Ogawa, M., and Tayama, K. (2026). “Cost-Effective Entity Matching With Large Language Models and Diversity-Aware Example Selection.” IEEE Access 14, 105851–105862.** [DOI](https://doi.org/10.1109/ACCESS.2026.3708863). D-diversity §IV-B clusters pairwise embedding differences to select labeled LLM demonstrations. §VI-B also reports changed positive-example prevalence, a possible contributor to gains. Publisher full text was retrieved through its search index; direct DOI access failed during this review.

12. **Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., and Ré, C. (2017). “Snorkel: Rapid Training Data Creation with Weak Supervision.” PVLDB 11(3), 269–282.** [Published paper](https://www.vldb.org/pvldb/vol11/p269-ratner.pdf); [extended paper](https://arxiv.org/pdf/1711.10160). Appendix C describes error-pattern inspection and labeling-function revision; §3.2 addresses dependencies. Weak rules and LLM judgments remain fallible signals.

13. **Zeighami, S., Shankar, S., and Parameswaran, A. (2025). “Featurized-Decomposition Join: Low-Cost Semantic Joins with Guarantees.” arXiv:2512.05399v1.** [Inspected preprint](https://arxiv.org/html/2512.05399v1). §5 iteratively improves features/extractors using labeled difficult examples. This is close prior art for reusable semantic repairs. Reference-LLM labels and ignored non-LLM costs in its model must be reconsidered for an independently audited, billion-candidate linkage workload.

14. **Wang, H., Yang, R., Zheng, H., and Ke, X. (2026). “Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution.”** [Inspected technical report, arXiv:2605.25814v1](https://arxiv.org/html/2605.25814v1). Alper combines graph updates, propagation, and LLM queries. Table 1 and Appendix B bound the evidence and assumptions. Institutional metadata lists KDD 2026; claims here refer to the inspected technical-report version.

15. **Gou, C., Banerjee, A., Wang, J., and Liu, C. (2026). “SemJoin: Semantic Join Optimization.” arXiv:2606.29532v1.** [Inspected preprint](https://arxiv.org/html/2606.29532v1). §3 describes cluster filtering and classifier execution. Tables 1–2 use small semantic-relation workloads and per-dataset-tuned cluster upper bounds. This does not establish large-scale identity matching.

## Identification and independent evaluation

16. **Bezdek, J. C., Ehrlich, R., and Full, W. (1984). “FCM: The fuzzy c-means clustering algorithm.” Computers & Geosciences 10(2–3), 191–203.** [DOI](https://doi.org/10.1016/0098-3004(84)90020-7); [full paper](https://web-ext.u-aizu.ac.jp/course/bmclass/documents/FCM%20-%20The%20Fuzzy%20c-Means%20Clustering%20Algorithm.pdf). Membership and objective definitions explain why fuzzy geometry has no automatic match-probability interpretation.

17. **Allman, E. S., Matias, C., and Rhodes, J. A. (2009). “Identifiability of parameters in latent structure models with many observed variables.” Annals of Statistics 37(6A), 3099–3132.** [Author manuscript](https://jarhodesuaf.github.io/papers/Latent.pdf). Formal conditions for generic identifiability, up to component labeling, in latent-class models. They do not guarantee practical calibration or justify arbitrary correlated comparisons.

18. **Dasylva, A., Goussanou, A., and Nambeu, C.-O. (2024). “Models of linkage error for capture-recapture estimation without clerical reviews.” Survey Methodology 50(2), 375–408.** [Official paper](https://www150.statcan.gc.ca/n1/pub/12-001-x/2024002/article/00007-eng.pdf). §§2, 4, and 6 show a label-free modeling route under explicit assumptions about sources and linkage. This is the important counterexample to a blanket assertion that error estimation always requires gold labels.

19. **Dasylva, A. (2018). “Design-Based Estimation with Record-Linked Administrative Files and a Clerical Review Sample.” Journal of Official Statistics 34(1), 41–54.** [DOI](https://doi.org/10.1515/jos-2018-0003). §§2–3 distinguish probability review, model assistance, and coverage of blocking/nonblocking strata. Sampling design matters when extrapolating from reviewed cases.

20. **Farquhar, S., Gal, Y., and Rainforth, T. (2021). “On Statistical Bias in Active Learning: How and When to Fix It.” ICLR.** [Full text](https://arxiv.org/html/2101.11665v2). §§3.1–3.3 derive design-aware corrections for adaptive sampling. Naively inverting a per-draw acquisition probability is not a universal correction for sampling without replacement or repeated model selection.

21. **Lam et al. (2026). “Designing Ambiguity-Aware Clerical Review: A Stratified Sampling Framework for Record Linkage and Deduplication.” arXiv:2608.01401v1.** [Preprint](https://arxiv.org/abs/2608.01401). §§2.1–2.3 stratify review by scores, comparisons, ambiguity, and demographics. It studies review estimation on candidate pairs, leaving repair transfer and complete blocking recall as separate questions.

22. **Binette, O., Baek, Y., Engineer, S., Jones, C., Dasylva, A., and Reiter, J. P. (2024). “How to Evaluate Entity Resolution Systems: An Entity-Centric Framework with Application to Inventor Name Disambiguation.” arXiv:2404.05622v1.** [Inspected paper](https://arxiv.org/html/2404.05622v1). §3.2 samples records and reconstructs their true entity groups using search beyond predicted clusters. §4.4 warns about small-sample interval behavior. Reliable truth recovery remains consequential.

23. **Sadinle, M. (2017). “Bayesian Estimation of Bipartite Matchings for Record Linkage.” JASA 112(518), 600–612.** [Author preprint](https://arxiv.org/pdf/1601.06630). §§2 and 5 explain constrained matching and unresolved decisions. Bipartite identity matching assumes no within-file duplicates; archive mention relations need different cardinalities.

## Candidate generation and scale

24. **Thirumuruganathan et al. (2021). “Deep Learning for Blocking in Entity Matching: A Design Space Exploration.” PVLDB 14(11), 2459–2472.** [Paper](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf). §§2–3 and experiments motivate comparing neural and lexical candidate routes. Complementarity observed on benchmarks is a reason to measure a union, not a guarantee of deployment recall.

25. **Brinkmann, A., Shraga, R., and Bizer, C. (2024). “SC-Block: Supervised Contrastive Blocking within Entity Resolution Pipelines.” ESWC, LNCS 14664, 121–142.** [Paper](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf). §4.4 and Tables 4, 8–9 separate tuning targets from measured candidate recall and show sensitivity to retrieval orientation. The largest inspected benchmark is far smaller than the proposed cross-product.

26. **Douze et al. “The Faiss Library.” arXiv:2401.08281, inspected v4 (2025).** [Paper](https://arxiv.org/html/2401.08281v4). §§3–4 cover approximate search, refinement, and vector compression. Vector-neighbor recall and entity-match recall are distinct quantities; both need consideration in a linkage system.

27. **Gagliardelli, L., Papadakis, G., Simonini, G., Bergamaschi, S., and Palpanas, T. “Generalized Supervised Meta-blocking.”** [Inspected 2022 technical report](https://arxiv.org/html/2204.08801v1); [2024 journal version, “GSM,” Information Systems 120, 102307](https://doi.org/10.1016/j.is.2023.102307). Candidate pruning can reduce downstream work. §5.4.2 also exposes the risk to true pairs with sparse shared-block evidence. Journal metadata was verified; claims refer to the inspected technical report.
