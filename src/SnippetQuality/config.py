import os

test_collection = "ntcir15" # or acordar1 / acordar2

label_collection_path = os.path.join(os.path.dirname(__file__), f"../../data/index/{test_collection}/label_collection")
label_index_path = os.path.join(os.path.dirname(__file__), f"../../data/index/{test_collection}/label_index")

component_collection_path = os.path.join(os.path.dirname(__file__), f"../../data/index/{test_collection}/component_collection")
component_index_path = os.path.join(os.path.dirname(__file__), f"../../data/index/{test_collection}/component_index")

stopword_file = os.path.join(os.path.dirname(__file__), f"../nltk_stopword.txt")

snippet_base = os.path.join(os.path.dirname(__file__), f"../../data/snippets/{test_collection}")
snippet_max_size = 20
alpha=0.6