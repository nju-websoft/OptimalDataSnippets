import pandas as pd
from ranx import Qrels, Run, evaluate, fuse, optimize_fusion

from .config import qrel_path, save_path, rerank_outputs_path, snippet_max_size

model = 'bge' # bge or bge-reranker
top_num = 10
points = 4
extra = "reranker_" if model == "bge-reranker" else ""
base = rerank_outputs_path

def transfer(d: dict):
    return {k: round(v, points) for k, v in d.items()}

qrel_df = pd.read_csv(qrel_path,
                      sep='\t',
                      names=['q_id', 'doc_id', 'score', 'split'],
                      dtype={'q_id': str, 'doc_id': str, 'score': int, 'split': int},
                      header=None)

qrels = Qrels.from_df(qrel_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

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

ret = transfer(evaluate(qrels, metadata_run, ['ndcg@5', 'ndcg@10', 'map@5', 'map@10']))
print(ret)


def get_split_df(df, split):
    filtered_df = df[df['split'] == split]
    return filtered_df


for epoch in [10]:
    for lr in ['1e-5', '3e-5', '5e-5']:
        data_df = pd.read_csv(
            base + f'BM25_data_ours_{snippet_max_size}_top{top_num}_{extra}reranking_lr_{lr}_bs_2_epoch_{epoch}.tsv',
            sep='\t',
            names=['q_id', 'doc_id', 'score'],
            dtype={'q_id': str, 'doc_id': str, 'score': float},
            header=None)
        data_run = Run.from_df(df=data_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

        dev_df = pd.read_csv(
            base + f'{model}\\BM25_data_ours_{snippet_max_size}_top{top_num}_{extra}reranking_dev_lr_{lr}_bs_2_epoch_{epoch}.tsv',
            sep='\t',
            names=['q_id', 'doc_id', 'score'],
            dtype={'q_id': str, 'doc_id': str, 'score': float},
            header=None)
        dev_run = Run.from_df(df=dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
        split_list = []
        for split in range(5):
            qrels_split_df = get_split_df(qrel_df, (split + 4) % 5)
            qrels_split_run = Qrels.from_df(qrels_split_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
            qrels_split_dev_df = get_split_df(qrel_df, (split + 3) % 5)
            qrels_split_dev_run = Qrels.from_df(qrels_split_dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
            metadata_split_df = get_split_df(metadata_df, (split + 4) % 5)
            metadata_split_run = Run.from_df(df=metadata_split_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
            metadata_split_dev_df = get_split_df(metadata_dev_df, (split + 3) % 5)
            metadata_split_dev_run = Run.from_df(df=metadata_split_dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
            data_split_df = get_split_df(data_df, (split + 4) % 5)
            data_split_run = Run.from_df(df=data_split_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')
            data_split_dev_df = get_split_df(dev_df, (split + 3) % 5)
            data_split_dev_run = Run.from_df(df=data_split_dev_df, q_id_col="q_id", doc_id_col='doc_id', score_col='score')

            best_params = optimize_fusion(
                qrels=qrels_split_dev_run,
                runs=[metadata_split_dev_run, data_split_dev_run],
                norm="min-max",  # Default value
                method='mixed',
                metric="ndcg@5",  # Metric we want to maximize
            )
            combined_run = fuse(
                runs=[metadata_split_run, data_split_run],
                norm="min-max",
                method='mixed',
                params=best_params,
            )
            ret = transfer(evaluate(qrels_split_run, combined_run, ['ndcg@5', 'ndcg@10', 'map@5', 'map@10']))
            split_list.append((ret, metadata_split_df['q_id'].nunique()))
            combined_run.save(f'{save_path}\\acordar_{model}_ours_top{top_num}_min-max_mixed_fold{split}.txt')
        score = {'ndcg@5': 0, 'ndcg@10': 0, 'map@5': 0, 'map@10': 0}
        total = 0
        for score_dict, num in split_list:
            score['ndcg@5'] += score_dict['ndcg@5'] * num
            score['ndcg@10'] += score_dict['ndcg@10'] * num
            score['map@5'] += score_dict['map@5'] * num
            score['map@10'] += score_dict['map@10'] * num
            total += num
        print({k: round(v / total, points) for k, v in score.items()})
