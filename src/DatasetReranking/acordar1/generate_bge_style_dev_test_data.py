# ouput format: [{"q_id": "", "question": "", "ctxs":[{"c_id": "", "text": ""}]}]

import os
import json
import sys
from tqdm import tqdm
from collections import defaultdict

sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence
from .config import test_collection, rerank_data_path, snippet_max_size, snippet_base, bm25_path


output_path = rerank_data_path
max_length = 500
top_num = 10
snippet_method = "ours"
max_size = snippet_max_size
data_util = DataUtil(test_collection)

def read_bm25():
    q2d = defaultdict(list)
    with open(bm25_path, 'r') as f:
        for line in f.readlines():
            items = line.strip('\n').split('\t')
            if len(items) != 6:
                continue
            query_id = items[0]
            dataset_id = items[2]
            rank = items[3]
            if int(rank) > top_num:
                continue
            q2d[int(query_id)].append(int(dataset_id))
    return q2d


def preprocess_qrels(qrels):
    split2query= dict()
    for query_id, _, _, split in qrels:
        if split not in split2query:
            split2query[split] = set()
        split2query[split].add(query_id)
    return split2query

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
    split2query = preprocess_qrels(qrels)
    id2query = data_util.get_id_to_query()
    q2d = read_bm25()

    splits = [0, 1, 2, 3, 4]

    query2dataset2text = get_snippet()
    for split in splits:
        print(f"{snippet_method}-{max_size}-split{split}")
        result_list = []
        for query_id in tqdm(split2query[split]):
            result = {"q_id": str(query_id), "question": id2query[query_id], "ctxs": []}
            for dataset_id in q2d[query_id]:
                if dataset_id not in query2dataset2text[query_id]:
                    result['ctxs'].append({'c_id': str(dataset_id), 'text': ""})
                    continue
                snippet_text = deal_sentence(query2dataset2text[query_id][dataset_id])
                snippet_text = ' '.join(snippet_text.split()[:max_length])
                result['ctxs'].append({'c_id': str(dataset_id), 'text': snippet_text})
            result_list.append(result)
        output = os.path.join(output_path, f"BM25_top{top_num}_{snippet_method}_{max_size}_split_{split}.json")
        with open(output, 'w') as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    main()