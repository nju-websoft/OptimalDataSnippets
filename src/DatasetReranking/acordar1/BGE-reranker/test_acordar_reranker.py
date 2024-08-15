from FlagEmbedding import FlagReranker
import json
import sys
from tqdm import tqdm

sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size

model_path = rerank_outputs_path
rerank_path = rerank_data_path

def cal_score(reranker, query, passages):
    res = []
    scores = reranker.compute_score([[query["question"], p['text']] for p in passages])
    for i in range(len(scores)):
        res.append((query['q_id'], passages[i], scores[i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

for mode in ['test', 'dev']:
    for topk in [10]:
        extra = '_dev' if mode == 'dev' else ''
        for segment_method in ['ours']:
            for max_size in [snippet_max_size]:
                for epoch in [10]:
                    for lr in ['1e-5', '3e-5', '5e-5']:
                        with open(f'{model_path}/BM25_data_{segment_method}_{max_size}_top{topk}_reranker_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}.tsv', 'w+') as fout:
                            for fold in range(0,5):  
                                print(f'acordar fold{fold} data_{segment_method}_{max_size}_{lr}_{epoch}')
                                split = (fold + 3) % 5 if mode == 'dev' else (fold + 4) % 5
                                with open(f'{rerank_path}/BM25_top{topk}_{segment_method}_{max_size}_split_{split}.json', 'r') as fin:
                                    test_json = json.load(fin)
                                reranker = FlagReranker(f'{model_path}/{segment_method}_{max_size}_reranker/fold{fold}/lr_{lr}_bs_2_epoch_{epoch}', use_fp16=True)    
                                for qp in tqdm(test_json):
                                    dataset_id_set = set()
                                    for res in cal_score(reranker, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                                        dataset_id = int(res[1]["c_id"])
                                        if dataset_id in dataset_id_set:
                                            continue
                                        dataset_id_set.add(dataset_id)
                                        fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')
            