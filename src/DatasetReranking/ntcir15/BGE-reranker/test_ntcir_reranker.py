import json
import sys
from tqdm import tqdm
from FlagEmbedding import FlagReranker


sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size

model_path = rerank_outputs_path
rerank_path = rerank_data_path
snippet_method = 'ours'
max_size = snippet_max_size

def cal_score(reranker, query, passages):
    res = []
    scores = reranker.compute_score([[query["question"], p['text']] for p in passages], batch_size=512)
    # print(scores)
    for i in range(len(scores)):
        res.append((query['q_id'], passages[i], scores[i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

for epoch in [10]:
    for lr in ['1e-5', '3e-5', '5e-5']:
        for split in ['test', 'dev']:
            extra = '_dev' if split == 'dev' else ''
            for fold in [15]:
                print(f'ntcir{fold} {snippet_method}_{epoch}_{lr}')
                with open(f'{model_path}/BM25_data_{snippet_method}_{max_size}_top10_reranker_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}.tsv', 'w+') as fout:
                    with open(f'{rerank_path}/BM25_top10_{snippet_method}_{max_size}_split_{split}.json', 'r') as fin:
                        test_json = json.load(fin)
                        reranker = FlagReranker(f'{model_path}/{snippet_method}_{max_size}_reranker/lr_{lr}_bs_2_epoch_{epoch}', use_fp16=True) 
                    for qp in tqdm(test_json):
                        dataset_id_set = set()
                        for res in cal_score(reranker, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                            dataset_id = res[1]["c_id"].split('___')[0]
                            if dataset_id in dataset_id_set:
                                continue
                            dataset_id_set.add(dataset_id)
                            fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')