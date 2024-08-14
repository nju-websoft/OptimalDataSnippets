import itertools
import os
import time
import json
import math
import signal
import time
import networkx as nx
from typing import Dict, List
from pyserini.search.lucene import LuceneSearcher
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.data_structure import WeightedTriple
from .config import snippet_base, label_index_path, snippet_max_size, test_collection

data_util = DataUtil(test_collection)
index_path = label_index_path
max_size = snippet_max_size
id2ttlP = defaultdict(int)
id2pid2frq = defaultdict(dict)
id2triples = defaultdict(list)
id2ttInDegree = defaultdict(int)
id2ttOutDegree = defaultdict(int)
id2eid2inDegree = defaultdict(dict)
id2eid2outDegree = defaultdict(dict)
id2ttlE = defaultdict(int)
id2eid2edp = defaultdict(dict)
id2eid2edpCount = defaultdict(dict)
id2entity2edp = defaultdict(dict)

def get_snippet_triple(query_id: str, filename: str) -> Dict[int, List[List[int]]]:
    snippet_path = os.path.join(snippet_base, f'{query_id}_{filename}/ours.json')
    with open(snippet_path, 'r') as f:
        snippet_list = json.load(f)
    return [[int(term) for term in triple['triples'].strip().split(' ')] for triple in snippet_list[:snippet_max_size]]

def get_basic_info(filename):
    global id2pid2frq, id2ttlP
    global id2ttInDegree, id2ttOutDegree, id2eid2inDegree, id2eid2outDegree
    global id2eid2edp, id2eid2edpCount, id2ttlE, id2entity2edp

    edp_count_list = data_util.get_edp_count_by_filename(filename)
    edp_list = data_util.get_edp_by_filename(filename)
    entity_list= data_util.get_entity_by_filename(filename)
    property_count_list = data_util.get_property_by_filename(filename)
    triple_list = data_util.get_triples_by_filename(filename)

    id2triples[filename] = list(triple_list)

    id2ttlP[filename] = sum(count for _, count in property_count_list)
    for property_id, count in property_count_list:
        id2pid2frq[filename][property_id] = count

    id2ttInDegree[filename] = 0
    id2ttOutDegree[filename] = 0
    for entity_id, in_degree, out_degree, edp in entity_list:
        id2eid2inDegree[filename][entity_id] = math.log(in_degree + 1)
        id2eid2outDegree[filename][entity_id] = math.log(out_degree + 1)
        id2ttInDegree[filename] += id2eid2inDegree[filename][entity_id]
        id2ttOutDegree[filename] += id2eid2outDegree[filename][entity_id]
        id2entity2edp[filename][entity_id] = edp

    id2ttlE[filename] = sum(count for _, count in edp_count_list)
    for edp_id, count in edp_count_list:
        id2eid2edpCount[filename][edp_id] = count
    
    for edp_id, kind, term_id in edp_list:
        if edp_id not in id2eid2edp[filename]:
            id2eid2edp[filename][edp_id] = set()
        if kind == 0: # fp
            id2eid2edp[filename][edp_id].add(-term_id)
        else: # bp
            id2eid2edp[filename][edp_id].add(term_id)

def property_score(triples, filename):
    snipPrp = {p for _, p, _ in triples}
    frqPrp = sum(id2pid2frq[filename][p] for p in snipPrp)
    return frqPrp / id2ttlP[filename] if id2ttlP[filename] > 0 else 0

def entity_score(triples, filename):
    entity_set = set()
    in_total = 0
    out_total = 0
    for s, _, o in triples:
        if s in id2eid2inDegree[filename] and s not in entity_set:
            entity_set.add(s)
            in_total += id2eid2inDegree[filename][s]
            out_total += id2eid2outDegree[filename][s]
        if o in id2eid2inDegree[filename] and o not in entity_set:
            entity_set.add(o)
            in_total += id2eid2inDegree[filename][o]
            out_total += id2eid2outDegree[filename][o]
    if len(entity_set) == 0:
        return 0
    ret = 0
    if id2ttInDegree[filename] > 0:
        ret += in_total / id2ttInDegree[filename]
    if id2ttOutDegree[filename] > 0:
        ret += out_total / id2ttOutDegree[filename]
    return ret

def pattern_score(triples, filename):
    entity_set = set()
    for s, p, o in triples:
        if s in id2eid2inDegree[filename].keys() and s not in entity_set:
            entity_set.add(s)
        if o in id2eid2inDegree[filename].keys() and o not in entity_set:
            entity_set.add(o)
    snippetMap = defaultdict(set)
    for s, p, o in triples:
        if s in entity_set:
            snippetMap[s].add(p)
        if o in entity_set:
            snippetMap[o].add(-p)
    
    count = 0
    covered_edp_max_size = defaultdict(int)
    for entity, props in snippetMap.items():
        edp_id = id2entity2edp[filename][entity]
        covered_edp_max_size[edp_id] = max(covered_edp_max_size[edp_id], len(props))
    
    for edp_id, size in covered_edp_max_size.items():
        count += id2eid2edpCount[filename][edp_id] * size / len(id2eid2edp[filename][edp_id])

    return count / id2ttlE[filename] if id2ttlE[filename] > 0 else 0

def cohesion_score(triples):
    graph = nx.MultiDiGraph()
    for s, p, o in triples:
        graph.add_edge(s, o, key=p)
    
    components = nx.weakly_connected_components(graph)
    max_component_size = max(len(list(graph.subgraph(component).edges(keys=True))) for component in components)
    return max_component_size / len(triples)

def cal_score(triples, filename):
    if not triples:
        return 0
    score = (
        property_score(triples, filename)
        + entity_score(triples, filename)
        + pattern_score(triples, filename)
        + 2 * cohesion_score(triples)
    )
    return score

def timeout_handler(signum, frame):
    raise TimeoutError("Task timed out")

def run_with_timeout(task, timeout, *args):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = task(*args)
    except TimeoutError:
        result = None
    finally:
        signal.alarm(0)
    return result

def get_keyword_information(query_id, filename):
    searcher = LuceneSearcher(os.path.join(index_path, str(filename)))
    query_text = id2query[query_id]
    keywords = list(set(sum((keyword.split('/') for keyword in query_text.split()), [])))
    kws2term = {}
    for kw in keywords:
        hits = searcher.search(kw)
        terms = [int(hit.docid) for hit in hits]
        if terms:
            kws2term[kw] = terms
    term2kws = defaultdict(set)
    for kw, terms in kws2term.items():
        for term in terms:
            term2kws[term].add(kw)
    return kws2term, term2kws

def generate_best(args):
    query_id, filename = args
    if filename not in id2triples:
        get_basic_info(filename)

    best_selection = []
    best_selection_score = 0

    kws2term, term2kws = get_keyword_information(query_id, filename)
    target = min(max_size, len(kws2term))

    snippet_selection = get_snippet_triple(query_id, filename)
    snippet_selection_score = cal_score(snippet_selection, filename)

    if len(id2triples[filename]) <= max_size:
        best_selection = id2triples[filename]
        best_selection_score = cal_score(best_selection, filename)
    else:
        for selected_triples in itertools.combinations(id2triples[filename], max_size):
            covered_keywords = set()
            for triple in selected_triples:
                if len(covered_keywords) >= target:
                    break
                for term in triple:
                    if term in term2kws:
                        covered_keywords.update(term2kws[term])
            if len(covered_keywords) < target:
                continue
            selected_score = cal_score(selected_triples, filename)
            if selected_score > best_selection_score:
                best_selection = selected_triples
                best_selection_score = selected_score

    with open('valid_query_dataset_approximation_ratio.txt', 'a') as f:
        f.write(f"{query_id}\t{filename}\t{len(id2triples[filename])}\t{snippet_selection_score}\t{best_selection_score}\n")
    
    os.makedirs(f'{snippet_base}/{query_id}_{filename}', exist_ok=True)
    with open(f'{snippet_base}/{query_id}_{filename}/best_triples.txt', 'w+') as f:
        for subject, predicate, object in best_selection:
            f.write(f"{subject}\t{predicate}\t{object}\n")
    
    if best_selection_score == 0:
        return 0, 1
    return len(id2triples[filename]), best_selection_score / snippet_selection_score


def main():
    global id2query
    qrels = data_util.get_qrels()
    id2query = data_util.get_id_to_query()
    dataset2filename = data_util.get_dataset_to_file()

    query_filename_list = []
    for query_id, dataset_id, _, _ in qrels:
        for filename in dataset2filename[dataset_id]:
            triple_num = len(data_util.get_triples_by_filename(filename))
            if triple_num >= 25 and triple_num <= 35:
                query_filename_list.append((query_id, filename))
    
    timeout = 60 * 60 * 24
    success = 0
    approximation_list = []
    for query_id, filename in tqdm(query_filename_list):
        start_time = time.time()
        result = run_with_timeout(generate_best, timeout, (query_id, filename))
        end_time = time.time()
        if result:
            success += 1
            approximation_list.append(result[1])
            print(f"Instance {query_id}_{filename} success: took {end_time - start_time} seconds, success {success} instance, now average approximation ratio is {sum(approximation_list)/len(approximation_list)}")
        else:
            print(f"Instance {query_id}_{filename} timed out")


if __name__ == "__main__":
    main()