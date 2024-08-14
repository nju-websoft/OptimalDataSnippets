from ranx import Qrels, Run, evaluate, fuse, optimize_fusion
import pandas as pd

from .config import qrel_path, save_path, rerank_outputs_path, snippet_max_size

dataset = 'ntcir15'
model = 'bge' # bge or bge-reranker
top_num = 10
points = 4
extra = "reranker_" if model == "bge-reranker" else ""
base = rerank_outputs_path

qrel_df = pd.read_csv(qrel_path,
                      sep='\t',
                      names=['q_id', 'doc_id', 'score', 'split'],
                      dtype={'q_id': str, 'doc_id': str, 'score': int, 'split': str},
                      header=None)

qrels = Qrels.from_df(qrel_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

def transfer(d: dict):
    return {k: round(v, points) for k, v in d.items()}

metadata_df = pd.read_csv(base + f'BM25_metadata_top{top_num}_{extra}reranking.tsv',
                          sep='\t',
                          names=['q_id', 'doc_id', 'score'],
                          dtype={'q_id': str, 'doc_id': str, 'score': float},
                          header=None)
metadata_run = Run.from_df(df=metadata_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

metadata_dev_df = pd.read_csv(base + f'BM25_metadata_top{top_num}_{extra}reranking_dev.tsv',
                              sep='\t',
                              names=['q_id', 'doc_id', 'score'],
                              dtype={'q_id': str, 'doc_id': str, 'score': float},
                              header=None)
metadata_dev_run = Run.from_df(df=metadata_dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

dev_qids = set(metadata_dev_df['q_id'].tolist())
dev_qrel_df = qrel_df.loc[qrel_df['q_id'].isin(dev_qids)]
dev_qrels = Qrels.from_df(dev_qrel_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

test_qids = set(metadata_df['q_id'].tolist())
test_qrel_df = qrel_df.loc[qrel_df['q_id'].isin(test_qids)]
test_qrels = Qrels.from_df(test_qrel_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

ret = transfer(evaluate(test_qrels, metadata_run, ['ndcg@5', 'ndcg@10', 'map@5', 'map@10']))
print(ret)

for epoch in [10]:
    for lr in ['1e-5', '3e-5', '5e-5']:
        for bs in [2]:
            data_df = pd.read_csv(
                base + f'BM25_data_ours_{snippet_max_size}_top{top_num}_{extra}reranking_lr_{lr}_bs_{bs}_epoch_{epoch}.tsv',
                sep='\t',
                names=['q_id', 'doc_id', 'score'],
                dtype={'q_id': str, 'doc_id': str, 'score': float},
                header=None)
            data_run = Run.from_df(df=data_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

            dev_df = pd.read_csv(
                base + f'{model}\\BM25_data_ours_{snippet_max_size}_top{top_num}_{extra}reranking_dev_lr_{lr}_bs_{bs}_epoch_{epoch}.tsv',
                sep='\t',
                names=['q_id', 'doc_id', 'score'],
                dtype={'q_id': str, 'doc_id': str, 'score': float},
                header=None)
            dev_run = Run.from_df(df=dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

            best_params = optimize_fusion(
                qrels=dev_qrels,
                runs=[metadata_dev_run, dev_run],
                norm="min-max",
                method='mixed',
                metric="ndcg@5",  # Metric we want to maximize
            )
            
            combined_run = fuse(
                runs=[metadata_run, data_run],
                norm="min-max",
                method='mixed',
                params=best_params
            )
            print(
                f"==========================lr {lr}==epoch {epoch}================================")
            ret = transfer(evaluate(test_qrels, combined_run, ['ndcg@5', 'ndcg@10', 'map@5', 'map@10']))
            print(ret)
            metadata_run.save(f'{save_path}/ntcir_{model}_metadata_top{top_num}.txt')
            combined_run.save(f'{save_path}/ntcir_{model}_ours_top{top_num}_min-max_mixed.txt')