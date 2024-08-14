import json
import csv
from tqdm import tqdm
from FlagEmbedding import FlagModel
import sys

sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size

model_path = rerank_outputs_path
rerank_path = rerank_data_path

def cal_score(model, query, passages):
    res = []
    q_embeddings = model.encode_queries([query["question"]], batch_size=512)
    p_embeddings = model.encode([p["text"] for p in passages], batch_size=512)
    scores = q_embeddings @ p_embeddings.T
    # print(scores)
    for i in range(len(scores[0])):
        res.append((query['q_id'], passages[i], scores[0][i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res


for epoch in [10]:
    for lr in ['1e-5', '3e-5', '5e-5']:
        for split in ['test', 'dev']:
            extra = '_dev' if split == 'dev' else ''
            for fold in [15]:
                print(f'ntcir{fold} ours_{epoch}_{lr}')
                with open(f'{model_path}/BM25_data_ours_{snippet_max_size}_top10_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}.tsv', 'w+') as fout:
                    with open(f'{rerank_path}/BM25_top10_ours_{snippet_max_size}_split_{split}.json', 'r') as fin:
                        test_json = json.load(fin)
                        model = FlagModel(f'{model_path}/ours_{snippet_max_size}/lr_{lr}_bs_2_epoch_{epoch}', query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ") 
                    for qp in tqdm(test_json):
                        dataset_id_set = set()
                        for res in cal_score(model, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                            dataset_id = res[1]["c_id"].split('___')[0]
                            if dataset_id in dataset_id_set:
                                continue
                            dataset_id_set.add(dataset_id)
                            fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')