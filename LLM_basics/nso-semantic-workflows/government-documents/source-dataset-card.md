---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- legal
size_categories:
- 10M<n<100M
---

LePaRD is a massive collection of U.S. federal judicial citations to precedent in context. LePaRD builds on millions of expert decisions by extracting quotations to precedents from judicial opinions along with the preceding context. Each row of the dataset corresponds to a quoted passage from prior case law used in a certain context.

The distribution of passage citation frequency is long tailed, a small number of passages appear thousands of times in the data while many are cited just once of twice. As a result, the passage retrieval task becomes harder as we consider more data. 

We provide four versions of LePaRD:
* [top_10000_data.csv.gz](https://huggingface.co/datasets/rmahari/LePaRD/blob/main/top_10000_data.csv.gz): Contains the data corresponding to the 10,000 most cited passages
* [top_20000_data.csv.gz](https://huggingface.co/datasets/rmahari/LePaRD/blob/main/top_20000_data.csv.gz): Contains the data corresponding to the 20,000 most cited passages
* [top_50000_data.csv.gz](https://huggingface.co/datasets/rmahari/LePaRD/blob/main/top_50000_data.csv.gz): Contains the data corresponding to the 50,000 most cited passages
* [all_data.csv.gz](https://huggingface.co/datasets/rmahari/LePaRD/blob/main/all_data.csv.gz): Contains data associated with all passages.

Each row of LePaRD contains the following features:
* passage_id: A unique identifier for each passage
* destination_context: The preceding context before the quotation
* quote: The text of the passage that was quoted
* court: The court from which the passage originated
* date: The date when the opinion from which the passage originated was published

Passage_ids are mapped to the passage text in [passage_dict.json](https://huggingface.co/datasets/rmahari/LePaRD/blob/main/passage_dict.json).
Note that multiple slightly different quotes can map to the same passage as judges will sometimes cite different parts of the same sentence.

In the vocabulary of information retrieval, the destination_context can be seen as a query, and the predicted passage_id (or the actual text of a passage in passage_dict.json) can be seen as the targets.

LePaRD was created by [Mahari et al.](https://arxiv.org/abs/2311.09356). More information on using LePaRD and a replication package for our paper can be found in the [LePaRD Github Repo](https://github.com/rmahari/LePaRD).