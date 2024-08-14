import json
import os
from collections import defaultdict
import sys
from tqdm import tqdm
sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_iri, deal_sentence
from .config import test_collection, rerank_data_path, snippet_max_size, snippet_base, bm25_dev_path, bm25_path

data_type = "dev" # "test"
top_num = 10
max_length = 500
output_path = rerank_data_path
bm25_path =  bm25_path if data_type == "test" else bm25_dev_path
delim = '\t' if data_type == "test" else ' '
data_util = DataUtil(test_collection)
max_size = snippet_max_size


def read_bm25():
    q2d = defaultdict(list)
    with open(bm25_path, 'r') as f:
        for line in f.readlines():
            items = line.strip('\n').split(delim)
            if len(items) != 6:
                continue
            query_id = items[0]
            dataset_id = items[2]
            rank = items[3]
            if int(rank) > top_num:
                continue
            q2d[query_id].append(dataset_id)
    return q2d

def main():
    query2dataset = read_bm25()
    id2query = data_util.get_id_to_query()
    dataset2files = data_util.get_dataset_to_file()
    
    result_list = []

    for query_id in tqdm(query2dataset):
        result = {"q_id": query_id, "question": id2query[query_id], "ctxs": []}
        for dataset_id in query2dataset[query_id]:
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
                result['ctxs'].append({"c_id": f"{dataset_id}____{filename}", "text": snippet_text})

        result_list.append(result)


    output = os.path.join(output_path, f"BM25_top{top_num}_ours_{snippet_max_size}_split_{data_type}.json")
    with open(output, 'w') as f:
        json.dump(result_list, f)

if __name__ == "__main__":
    main()