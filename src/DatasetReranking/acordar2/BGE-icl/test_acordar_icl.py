from FlagEmbedding import FlagICLModel
import json
import sys
from tqdm import tqdm

sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size, alpha

output_path = rerank_outputs_path
input_path = rerank_data_path

def cal_score(model, query, passages):
    res = []
    q_embeddings = model.encode_queries([query["question"]])
    p_embeddings = model.encode_corpus([p["text"] for p in passages])
    scores = q_embeddings @ p_embeddings.T
    # print(scores)
    for i in range(len(scores[0])):
        res.append((query['q_id'], passages[i], scores[0][i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

model = FlagICLModel('/path/to/model/bge-icl',
                    query_instruction_for_retrieval="Given a web search query, retrieve relevant passages that answer the query.",
                    examples_for_task=None,
                    use_fp16=True)
for mode in ['test']:
    for segment_method in ['ours']:
        with open(f'{output_path}/pool_data_{segment_method}_{snippet_max_size}_{alpha}_reranking_icl.tsv', 'w+') as fout:
            print(f'acordar data_{segment_method}')
            with open(f'{input_path}/pool_{segment_method}_{snippet_max_size}_{alpha}_split_{mode}.json', 'r') as fin:
                test_json = json.load(fin)
            for qp in tqdm(test_json):
                dataset_id_set = set()
                for res in cal_score(model, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                    dataset_id = int(res[1]["c_id"])
                    if dataset_id in dataset_id_set:
                        continue
                    dataset_id_set.add(dataset_id)
                    fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')


