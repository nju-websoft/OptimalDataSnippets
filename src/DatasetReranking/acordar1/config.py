import os

test_collection = "acordar1"

rerank_data_path = os.path.join(os.path.dirname(__file__), f"../../../data/rerank/{test_collection}")
rerank_outputs_path = os.path.join(os.path.dirname(__file__), f"../../data/rerank_outputs/{test_collection}")

bm25_path = os.path.join(os.path.dirname(__file__), f"../../../data/test_collection_data/{test_collection}/BM25F.txt")

qrel_path = os.path.join(os.path.dirname(__file__), f"../../../data/test_collection_data/{test_collection}/qrels.txt")

save_path = os.path.join(os.path.dirname(__file__), f"../../../data/results")

snippet_base = os.path.join(os.path.dirname(__file__), f"../../data/snippets/{test_collection}")
snippet_max_size = 20