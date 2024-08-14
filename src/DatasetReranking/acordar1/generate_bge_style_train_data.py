
# output format: [{"query": "", "pos": ["", ""], "neg":[]}]
import os
import jsonlines
import json
import sys
from tqdm import tqdm

sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence
from .config import test_collection, rerank_data_path, snippet_max_size, snippet_base


output_path = rerank_data_path
max_length = 500
snippet_method = "ours"
max_size = snippet_max_size
data_util = DataUtil(test_collection)

def preprocess_qrels(qrels):
    split2query2dataset = dict()
    for query_id, dataset_id, rel_score, split in qrels:
        if split not in split2query2dataset:
            split2query2dataset[split] = dict()
        if query_id not in split2query2dataset[split]:
            split2query2dataset[split][query_id] = {"pos": [], "neg": []}
        if rel_score == 0:
            split2query2dataset[split][query_id]['neg'].append(dataset_id)
        else:
            split2query2dataset[split][query_id]['pos'].append(dataset_id)
    return split2query2dataset

def get_snippet():
    query_dataset_list = os.listdir(snippet_base)
    query2dataset2text = dict()
    for query_dataset in query_dataset_list:
        with open(os.path.join(snippet_base, query_dataset, 'ours.json'), 'r') as f:
            triples = json.load(f)
            lst = query_dataset.split('_')
            query_id = int(lst[0])
            dataset_id = int(lst[1])
            if query_id not in query2dataset2text:
                query2dataset2text[query_id] = dict()
            if dataset_id not in query2dataset2text[query_id]:
                query2dataset2text[query_id][dataset_id] = ' '.join([triple['text'] for triple in triples])
    return query2dataset2text


def main():
    qrels = data_util.get_qrels()
    split2query2dataset = preprocess_qrels(qrels)
    id2query = data_util.get_id_to_query()
    query2dataset2text = get_snippet()
    for split in range(5):
        print(f"ours-{max_size}-split{split}")
        result_list = []
        for query_id in tqdm(split2query2dataset[split]):
            result = {"query": id2query[query_id], "pos": [], "neg": []}
            for dataset_id in split2query2dataset[split][query_id]['pos']:
                if dataset_id not in query2dataset2text[query_id]:
                    continue
                positive_text = deal_sentence(query2dataset2text[query_id][dataset_id])
                if positive_text != "":
                    positive_text = ' '.join(positive_text.split()[:max_length])
                    result['pos'].append(positive_text)
            
            for dataset_id in split2query2dataset[split][query_id]['neg']:
                if dataset_id not in query2dataset2text[query_id]:
                    continue
                negative_text = deal_sentence(query2dataset2text[query_id][dataset_id])
                if negative_text != "":
                    negative_text = ' '.join(negative_text.split()[:max_length])
                    result['neg'].append(negative_text)
            result_list.append(result)
        output = os.path.join(output_path, f"{snippet_method}_{max_size}_split_{split}.jsonl")
        with jsonlines.open(output, 'w') as writer:
            writer.write_all(result_list)
    for fold in range(5):
        split1 = fold
        split2 = (fold + 1) % 5
        split3 = (fold + 2) % 5
        results = []
        for split in [split1, split2, split3]:
            input_path = os.path.join(output_path, f"{snippet_method}_{max_size}_split_{split}.jsonl")
            with jsonlines.open(input_path, 'r') as reader:
                for result in reader:
                    results.append(result)
        output = os.path.join(output_path, f"{snippet_method}_{max_size}_split_{split1}{split2}{split3}.jsonl")
        with jsonlines.open(output, 'w') as writer:
            writer.write_all(results)


if __name__ == "__main__":
    main()