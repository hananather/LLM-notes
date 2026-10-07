# Product source notices

Erhard Rahm's Database Group at Leipzig University publishes these datasets under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Its [dataset page](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution), checked on 7 October 2026, explicitly states this for the binary entity-resolution datasets in its January 2019 notice.

Sources:

- [Abt–Buy](https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip): 1,081 Abt records, 1,092 Buy records and 1,097 supplied product links.
- [Amazon–GoogleProducts](https://dbs.uni-leipzig.de/files/datasets/Amazon-GoogleProducts.zip): 1,363 Amazon records, 3,226 Google records and 1,300 supplied product links.

Please credit this dataset page and H. Köpcke, A. Thor and E. Rahm, *Evaluation of Entity Resolution Approaches on Real-World Match Problems*, Proceedings of the VLDB Endowment 3(1–2), 2010.

The source ZIPs are preserved byte for byte. This experiment changes column names to a common schema, assigns opaque record identifiers, and separates inference records, source metadata, partitions and reference links. Derived normalized fields and predictions are experimental outputs. These changes do not imply endorsement by the source publishers or retailers.

The supplied mappings define the benchmark relation. Some records have several correct counterparts. All Abt–Buy records have a supplied counterpart. Records without a supplied Amazon–Google counterpart are scored as unmatched under that publication's mapping convention. This does not independently establish that no real-world product equivalent exists.

Historical descriptions and prices do not represent current retailer inventories or current prices. Their suitability as a proxy for official-statistics product work requires separate operational evidence.
