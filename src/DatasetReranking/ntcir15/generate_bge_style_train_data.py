import json
import os
import jsonlines
import sys

from collections import defaultdict
from tqdm import tqdm
sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence
from .config import test_collection, rerank_data_path, snippet_max_size, snippet_base

split = "train"
output_path = rerank_data_path
max_length = 500
data_util = DataUtil(test_collection)

max_size = snippet_max_size

def get_data(qrels, data_type):
    query2positive = defaultdict(list)
    query2negative = defaultdict(list)
    for query_id, dataset_id, score, type in qrels:
        if type != data_type:
            continue
        if score == 0:
            query2negative[query_id].append(dataset_id)
        else:
            query2positive[query_id].append(dataset_id)
    if len(query2negative) > len(query2positive):
        for query in query2negative:
            if query not in query2positive:
                query2positive[query] = []
    else:
        for query in query2positive:
            if query not in query2negative:
                query2negative[query] = []
    return query2positive, query2negative

def get_relevant_snippet_if_exist(dataset2files, dataset_id, query_id):
    snippet_text_list = []
    for filename in dataset2files[dataset_id]:
        snippet_path = f"{snippet_base}/{query_id}_{filename}/ours.json"
        if not os.path.exists(snippet_path):
            print(f"snippet not exist: {snippet_path}")
            continue
        
        with open(snippet_path, 'r') as f:
            triples = json.load(f)
            snippet_text = ' '.join([triple['text'] for triple in triples[:max_size]])

        snippet_text = deal_sentence(snippet_text)
        if snippet_text != "":
            snippet_text = ' '.join(snippet_text.split()[:max_length])
            snippet_text_list.append(snippet_text)
    return snippet_text_list

def main():
    qrels = data_util.get_qrels()
    dataset2files = data_util.get_dataset_to_file()
    id2query = data_util.get_id_to_query()
    query2positive_train, query2negative_train = get_data(qrels, split)

    result_list = []
    for query_id in tqdm(query2positive_train):
        result = {"query": id2query[query_id], "pos": [], "neg": []}
        for dataset_id in query2positive_train[query_id]:            
            positive_snippet_list = get_relevant_snippet_if_exist(dataset2files, dataset_id, query_id)
            result['pos'].extend(positive_snippet_list)

        for dataset_id in query2negative_train[query_id]:
            negative_snippet_list = get_relevant_snippet_if_exist(dataset2files, dataset_id, query_id)
            result['neg'].extend(negative_snippet_list)
        
        if len(result['pos']) == 0:
            continue
        result_list.append(result)
    
    output = os.path.join(output_path, f"ours_{snippet_max_size}_split_{split}.jsonl")
    with jsonlines.open(output, 'w') as writer:
        writer.write_all(result_list)

if __name__ == "__main__":
    main()

