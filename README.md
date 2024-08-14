# Extracting Optimal Data Snippets for Dataset Search and Beyond

This is the source code and data of the paper "Extracting Optimal Data Snippets for Dataset Search and Beyond".

## Table of Contents

- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Running Our Snippet Extraction Method](#running-our-snippet-extraction-method)
  - [Getting Started](#getting-started)
  - [Experiment 1: Dataset Reranking](#experiment-1-dataset-reranking)
  - [Experiment 2: Snippet Quality](#experiment-2-snippet-quality)
  - [Experiment 3: Approximation Ratio](#experiment-3-approximation-ratio)
- [Other Algorithms](#other-algorithms)
- [License](#license)
- [Citation](#citation)

## Directory Structure

The `./src` directory contains all the source code, which is based on Python 3.10 and JDK 11.

- `./src/OptimalDataSnippets` contains the implementation of our method.
- `./src/DatasetReranking` contains the code for Experiment 1 (Dataset Reranking).
- `./src/SnippetQuality` contains the code for Experiment 2 (Snippet Quality).
- `./src/ApproximationRatio` contains the code for Experiment 3 (Approximation Ratio).
- `./src/Utils` contains classes and tools.

The `./data` directory contains test collection data and results for reranking experiments.

- `./data/test_collection_data/{test_collection}` contains test collection data, including qrels, queries, and top-10 BM25 results.
- `./data/results` contains all the results for reranking experiments. Result files are named as `{test_collection}_{reranking_model}_{snippet_extraction_method}_{topk}_{normalization_method}_{fusion_method}.txt` and are in TREC format. For example, `ntcir_bge_ours_top10_min-max_mixed.txt` represents the reranking results of BGE with our snippet extraction method, min-max normalization, and mixed fusion method.


```
DS1-E-1001 Q0 f8cfaa69-3f89-4ebe-96e2-d15a30173f43 1 1.1313708498984762 mixed
DS1-E-1001 Q0 7629c1d5-5da8-45b5-bc8b-58483f97921a 2 0.8333792973031102 mixed
DS1-E-1001 Q0 ef605593-f1b3-40a4-a6a1-0c1fd2a4a381 3 0.7795421099910232 mixed
```

## Requirements

- pyserini==0.23.0
- FlagEmbedding
- tqdm
- networkx
- pandas
- torch
- transformers

## Running Our Snippet Extraction Method

### Getting Started

We conducted experiments on the following two test collections for ad hoc dataset retrieval:

#### NTCIR-E

[NTCIR-E](https://ntcir.datasearch.jp/data_search_1/) is the English version of the test collection used in the NTCIR-15 Dataset Search task, including 46,615 datasets and 192 queries.

#### ACORDAR

[ACORDAR](https://github.com/nju-websoft/ACORDAR) is a test collection specifically for RDF datasets, including 31,589 datasets and 493 queries.

We used the graph-based unification method from [CDS](https://github.com/nju-websoft/CDS) to convert the data files of these two test collections into `term.tsv` and `triple.tsv`. Place them in the `./data/test_collection_data/{test_collection}/{filename}` folder.

Use the following commands to prepare for executing our algorithm in `./src/OptimalDataSnippets`:

```
python label_index_prepare.py
bash ./label_index_builder.py
python graph_init.py
```


The corresponding data and indexes will be generated in `./data/test_collection_data/{test_collection}/{filename}` and `./data/index/{test_collection}/label_index`, respectively.

Use the following command to extract our snippets for each query-data pair in qrels:



```
python extract_snippet.py
```


The resulting snippets will be in JSON files located in `./data/snippets`, formatted as follows:



```
[
  {
    "rank_num": 1,
    "triples": "{subject} {predicate} {object}",
    "text": "xxx"
  },
  ...
]
```


### Experiment 1: Dataset Reranking

We have placed all the code required for Experiment 1 in the `./src/DatasetReranking/{test_collection}` directory. First, generate the training and test sets using the following commands:


```
python generate_bge_style_train_data.py
python generate_bge_style_dev_test_data.py
```


We implemented a fine-tuned reranking model for **BGE** and **BGE-reranker** based on code from the [official GitHub repository](https://github.com/FlagOpen/FlagEmbedding). We use [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large) as initial checkpoints, respectively. We followed the official instructions to mine hard negatives and then finetune both models according to the recommended method.



```
bash ./finetune_{test_collection}_data_ours_{ranking_model}.sh
```


We then calculated the relevance score of each `<query, dataset>` pair following the official instructions.


```
python test_acordar_{ranking_model}.py
```

Run the following code to obtain the evaluation results:

```
python eval.py
```


Following the official guidelines, we directly used [SFR](https://huggingface.co/Salesforce/SFR-Embedding-2_R) and [BGE-icl](https://huggingface.co/BAAI/bge-en-icl) to calculate the relevance score of each `<query, dataset>` pair.



```
python test_acordar_{ranking_model}.py
```


Run the following code to get the evaluation results:


```
python eval_7b.py
```


### Experiment 2: Snippet Quality

Use the following commands to prepare for calculating quality metrics in `./src/SnippetQuality`:



```
python component_index_prepare.py
bash ./component_index_prepare.py
```

Run the following code to get the evaluation results:



```
python quality_metrics.py
```

### Experiment 3: Approximation Ratio

Run the following code to get the evaluation results in `./src/ApproximationRatio`:



```
python cal_approximation_ratio.py
```

## Other Algorithms

Please refer to the following repositories if you want to run other algorithms used in our experiments:

- IlluSnip: [BANDAR](https://github.com/nju-websoft/BANDAR/blob/master/code/src/snippetAlgorithm/IlluSnip.java)
- KSD: [BANDAR](https://github.com/nju-websoft/BANDAR/blob/master/code/src/snippetAlgorithm/KSDSnippet.java)
- CDS: [CDS](https://github.com/nju-websoft/CDS)

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Citation
