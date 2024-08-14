import json

from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import sys

sys.path.append("..")
from config import rerank_outputs_path, rerank_data_path, snippet_max_size

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

output_path = rerank_outputs_path
input_path = rerank_data_path
model_path = "/path/to/SFR-Embedding-2_R"

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

task = 'Given a web search query, retrieve relevant passages that answer the query'
max_length = 1024

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, device_map='auto')

model.eval()

@torch.no_grad()
def cal_score(model, query, passages):
    res = []
    input_texts = [get_detailed_instruct(task, query["question"])] + [p["text"] for p in passages]
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:1] @ embeddings[1:].T)
    for i in range(len(scores[0])):
        res.append((query['q_id'], passages[i], scores[0][i]))
    res.sort(key=lambda x:x[2], reverse=True)
    return res

for mode in ['dev', 'test']:
    for topk in [10]:
        for segment_method in ['ours']:
            with open(f'{output_path}/BM25_data_{segment_method}_{snippet_max_size}_top{topk}_reranking_sfr_split_{mode}.tsv', 'w+') as fout:
                print(f'ntcir data_{segment_method}')
                with open(f'{input_path}/ntcir/data/BM25_top{topk}_{segment_method}_{snippet_max_size}_split_{mode}.json', 'r') as fin:
                    test_json = json.load(fin)
                for qp in tqdm(test_json):
                    dataset_id_set = set()
                    torch.cuda.empty_cache()
                    for res in cal_score(model, {"q_id": qp["q_id"], "question": qp["question"]}, qp["ctxs"]):
                        dataset_id = res[1]["c_id"].split('___')[0]
                        if dataset_id in dataset_id_set:
                            continue
                        dataset_id_set.add(dataset_id)
                        fout.write(f'{res[0]}\t{dataset_id}\t{res[2]}\n')



