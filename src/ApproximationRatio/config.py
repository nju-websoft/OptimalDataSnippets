import os

test_collection = "ntcir15" # or acordar1

label_index_path = os.path.join(os.path.dirname(__file__), f"../../data/index/{test_collection}/label_index")

snippet_base = os.path.join(os.path.dirname(__file__), f"../../data/snippets/{test_collection}")
snippet_max_size = 20