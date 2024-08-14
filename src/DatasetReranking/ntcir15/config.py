import os

test_collection = "ntcir15"

rerank_data_path = os.path.join(os.path.dirname(__file__), f"../../../data/rerank/{test_collection}")
rerank_outputs_path = os.path.join(os.path.dirname(__file__), f"../../data/rerank_outputs/{test_collection}")

bm25_dev_path = os.path.join(os.path.dirname(__file__), f"../../../data/test_collection_data/{test_collection}/ntcir15_BM25_top10_dev.txt")
bm25_path = os.path.join(os.path.dirname(__file__), f"../../../data/test_collection_data/{test_collection}/ntcir15_BM25_top10_test.txt")

qrel_path = os.path.join(os.path.dirname(__file__), f"../../../data/test_collection_data/{test_collection}/qrels.txt")

save_path = os.path.join(os.path.dirname(__file__), f"../../../data/results")

snippet_base = os.path.join(os.path.dirname(__file__), f"../../data/snippets/{test_collection}")
snippet_max_size = 20