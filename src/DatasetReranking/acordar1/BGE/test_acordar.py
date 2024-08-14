from FlagEmbedding import FlagModel
import json
import sys
from tqdm import tqdm

sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size

model_path = rerank_outputs_path
rerank_path = rerank_data_path

def cal_score(model, query, passages):
    res = []
    q_embeddings = model.encode_queries([query["question"]])
    p_embeddings = model.encode([p["text"] for p in passages])
    scores = q_embeddings @ p_embeddings.T
    for i in range(len(scores[0])):
        res.append((query['q_id'], passages[i], scores[0][i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

for mode in ['test', 'dev']:
    for topk in [10]:
        extra = '_dev' if mode == 'dev' else ''
        for segment_method in ['ours']:
            for max_size in [snippet_max_size]:
                for epoch in [10]:
                    for lr in ['1e-5', '3e-5', '5e-5']:
                        for fold in range(5):  
                            split = (fold + 3) % 5 if mode == 'dev' else (fold + 4) % 5
                            with open(f'{model_path}/BM25_data_{segment_method}_{max_size}_top{topk}_reranking{extra}_lr_{lr}_bs_2_epoch_{epoch}_fold_{split}.tsv', 'w+') as fout:
                                print(f'acordar1 fold{fold}_{mode}')
                                with open(f'{rerank_path}/BM25_top{topk}_{segment_method}_{max_size}_split_{split}.json', 'r') as fin:
                                    test_json = json.load(fin)
                                model = FlagModel(f'{model_path}/{segment_method}_{max_size}/fold{fold}/lr_{lr}_bs_2_epoch_{epoch}', query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ")    
                                for qp in tqdm(test_json):
                                    dataset_id_set = set()
                                    for res in cal_score(model, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                                        dataset_id = int(res[1]["c_id"])
                                        if dataset_id in dataset_id_set:
                                            continue
                                        dataset_id_set.add(dataset_id)
                                        fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')


