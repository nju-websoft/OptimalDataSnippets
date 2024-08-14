# calculate connected component and index all text in component


import os
import json
import networkx as nx
from tqdm import tqdm
import sys
sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence

from .config import test_collection, component_collection_path

store_base = component_collection_path

data_util = DataUtil(test_collection)
def build_graph(triple_list):
    G = nx.MultiDiGraph()
    for s, p, o in triple_list:
        G.add_edge(s, o, key=p)
    return G

def get_edges_in_component(graph, component):
    edges_in_component = []
    for u, v, key in graph.edges(keys=True):
        if u in component and v in component:
            edges_in_component.append((u, key, v))
    return edges_in_component

def process_dataset(filename):
    term_list = data_util.get_terms_by_filename(filename)
    id2term = {term[0]: term[1] for term in term_list}
    triple_list = data_util.get_triples_by_filename(filename)
    graph = build_graph(triple_list)
    components = nx.weakly_connected_components(graph)

    collections = []
    for i, component in enumerate(components):
        edges_in_component = get_edges_in_component(graph, component)
        text_list = [' '.join([id2term[s], id2term[p], id2term[o]]) for s, p, o in edges_in_component]
        content = ' '.join(text_list)
        collections.append({"id": i + 1, "contents": ' '.join(deal_sentence(content).split())})
    output_path = os.path.join(store_base, str(filename))
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    with open(os.path.join(output_path, 'component.json'), 'w', encoding="utf-8", newline='\n') as writer:
        json.dump(collections, writer, ensure_ascii=False)

def main():
    qrels = data_util.get_qrels()
    datasets = set([qrel[1] for qrel in qrels])
    dataset2filename = data_util.get_dataset_to_file()
    filename_list = []
    for dataset_id in datasets:
        for file_name in dataset2filename[dataset_id]:
            if os.path.exists(f"{data_util.collection_base}/{file_name}/term.tsv"):
                filename_list.append(file_name)
    
    for filename in tqdm(filename_list):
        process_dataset(filename)
        


if __name__ == "__main__":
    main()