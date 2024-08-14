from collections import defaultdict
import math
import os
import json
from typing import Dict, List
import networkx as nx
import sys
sys.path.append("..")
from Utils.data_utils import DataUtil
from .config import snippet_base, snippet_max_size, component_index_path, label_index_path, test_collection
from pyserini.search.lucene import LuceneSearcher



max_size = snippet_max_size
index_path = label_index_path
component_index_path = component_index_path

id2ttlP = defaultdict(int)
id2pid2frq = defaultdict(dict)

id2MaxInDegree = defaultdict(int)
id2MaxOutDegree = defaultdict(int)
id2eid2inDegree = defaultdict(dict)
id2eid2outDegree = defaultdict(dict)

id2ttlE = defaultdict(int)
id2eid2edp = defaultdict(dict) # edp_id -> EDP set (fp + bp + class)
id2eid2edpCount = defaultdict(dict) # edp_id -> edp count
id2entity2edp = defaultdict(dict) # entity_id -> edp_id

data_util = DataUtil(test_collection)

def get_snippet_triple(query_id: str, filename: str) -> Dict[int, List[List[int]]]:
        snippet_path = os.path.join(snippet_base, f'{query_id}_{filename}/ours.json')
        if not os.path.exists(snippet_path):
            return []
        snippet_list = []
        with open(snippet_path, 'r') as f:
            snippet_list = json.load(f)
        return [[int(term) for term in triple['triples'].strip().split(' ')] for triple in snippet_list[:max_size]]

def skmRep(filename: str, triples: List[List[int]]) -> float:
    snipPrp = set()
    for s, p, o in triples:
        snipPrp.add(p)
    frqPrp = 0
    for p in snipPrp:
        frqPrp += id2pid2frq[filename][p]
    frqPrp = frqPrp / id2ttlP[filename] if id2ttlP[filename] > 0 else 0
    
    return frqPrp

def kwRel(filename: str, keywords: List[str], triples: List[List[int]]) -> float:
    count = 0
    terms = set()
    for triple in triples:
        for term in triple:
            terms.add(term)
    searcher = LuceneSearcher(os.path.join(index_path, str(filename)))
    keyword_count = 0
    for kw in keywords:
        hits = searcher.search(kw)
        hit_set = [int(hit.docid) for hit in hits]
        if len(hits) > 0:
            keyword_count += 1
            for term in terms:
                if term in hit_set:
                    count += 1
                    break
    if keyword_count == 0:
        return -1
    return count / keyword_count

def entRep(filename: str, triples: List[List[int]]) -> float:
    entity_set = set()
    in_total = 0
    out_total = 0
    for s, p, o in triples:
        if s in id2eid2inDegree[filename].keys() and s not in entity_set:
            entity_set.add(s)
            in_total += math.log(id2eid2inDegree[filename][s] + 1)
            out_total += math.log(id2eid2outDegree[filename][s] + 1)
        if o in id2eid2inDegree[filename].keys() and o not in entity_set:
            entity_set.add(o)
            in_total += math.log(id2eid2inDegree[filename][o] + 1)
            out_total += math.log(id2eid2outDegree[filename][o] + 1)
    if len(entity_set) == 0:
        return 0
    if id2MaxOutDegree[filename] == 0:
        out_total = 0
    else:
        out_total /= (len(entity_set) * math.log(id2MaxOutDegree[filename] + 1))
    if id2MaxInDegree[filename] == 0:
        in_total = 0
    else:
        in_total /= (len(entity_set) * math.log(id2MaxInDegree[filename] + 1))

    coDat = 0
    if id2MaxInDegree[filename] == 0:
        coDat = out_total
    elif in_total + out_total > 0:
        coDat = 2 * in_total * out_total / (in_total + out_total)
    return coDat

def descRep(filename: str, triples: List[List[int]]) -> float:
    entity_set = set()
    for s, p, o in triples:
        if s in id2eid2inDegree[filename].keys() and s not in entity_set:
            entity_set.add(s)
        if o in id2eid2inDegree[filename].keys() and o not in entity_set:
            entity_set.add(o)
    
    snippetMap = defaultdict(set) # entity_id -> covered property
    for s, p, o in triples:
        if s in entity_set:
            snippetMap[s].add(p)
        if o in entity_set:
            snippetMap[o].add(-p)
    
    count = 0
    covered_edp = set()
    for entity in snippetMap.keys():
        edp_id = id2entity2edp[filename][entity]
        if snippetMap[entity] == id2eid2edp[filename][edp_id] and edp_id not in covered_edp:
            covered_edp.add(edp_id)
            count += id2eid2edpCount[filename][edp_id]

    if id2ttlE[filename] > 0:
        return count / id2ttlE[filename]
    else:
        return 0

def qryRel(filename: str, keywords: List[str], triples: List[List[int]]) -> float:
    total = 0
    component_index = os.path.join(component_index_path, str(filename))
    if os.path.exists(component_index):
        searcher = LuceneSearcher(os.path.join(component_index_path, str(filename)))
        keyword_pairs = list(zip(keywords[:-1], keywords[1:]))
        for word1, word2 in keyword_pairs:
            hit_set1 = set(int(hit.docid) for hit in searcher.search(word1))
            hit_set2 = set(int(hit.docid) for hit in searcher.search(word2))
            if hit_set1.intersection(hit_set2):
                total += 1

    if total == 0:
        return kwRel(filename, keywords, triples)

    graph = nx.MultiDiGraph()
    for s, p, o in triples:
        graph.add_edge(s, o, key=p)
    components = nx.weakly_connected_components(graph)

    def get_edges_in_component(component):
        edges_in_component = []
        for u, v, key in graph.edges(keys=True):
            if u in component and v in component:
                edges_in_component.append((u, key, v))
        return edges_in_component

    term2component = dict()
    for i, component in enumerate(components):
        edges = get_edges_in_component(component)
        for triple in edges:
            for term in triple:
                term2component[term] = i
        
    searcher = LuceneSearcher(os.path.join(index_path, str(filename)))
    count = 0
    for i in range(len(keywords) - 1):
        word1 = keywords[i]
        word2 = keywords[i + 1]
        hit_set1 = set([int(hit.docid) for hit in searcher.search(word1)])
        hit_set2 = set([int(hit.docid) for hit in searcher.search(word2)])
        hit_term1 = hit_set1.intersection(term2component.keys())
        hit_term2 = hit_set2.intersection(term2component.keys())
        component1 = {term2component[term] for term in hit_term1}
        component2 = {term2component[term] for term in hit_term2}
        if component1.intersection(component2):
            count += 1
    return count / total

def preprocess_dataset(filename: str):
    global id2pid2frq, id2ttlP
    global id2MaxInDegree, id2MaxOutDegree, id2eid2inDegree, id2eid2outDegree
    global id2eid2edp, id2eid2edpCount, id2ttlE, id2entity2edp

    edp_count_list = data_util.get_edp_count_by_filename(filename)
    edp_list = data_util.get_edp_by_filename(filename)
    entity_list= data_util.get_entity_by_filename(filename)
    property_count_list = data_util.get_property_by_filename(filename)

    id2ttlP[filename] = 0
    for property_id, count in property_count_list:
        id2ttlP[filename] += count
        id2pid2frq[filename][property_id] = count

    id2MaxInDegree[filename] = 0
    id2MaxOutDegree[filename] = 0
    for entity_id, in_degree, out_degree, edp in entity_list:
        id2MaxOutDegree[filename] = max(id2MaxOutDegree[filename], out_degree)
        id2MaxInDegree[filename] = max(id2MaxInDegree[filename], in_degree)
        id2eid2inDegree[filename][entity_id] = in_degree
        id2eid2outDegree[filename][entity_id] = out_degree
        id2entity2edp[filename][entity_id] = edp

    id2ttlE[filename] = 0
    for edp_id, count in edp_count_list:
        id2ttlE[filename] += count
        id2eid2edpCount[filename][edp_id] = count
    
    for edp_id, kind, term_id in edp_list:
        if edp_id not in id2eid2edp[filename]:
            id2eid2edp[filename][edp_id] = set()
        if kind == 0: # fp
            id2eid2edp[filename][edp_id].add(-term_id)
        else: # bp
            id2eid2edp[filename][edp_id].add(term_id)
    

def get_score(query_id: int, query_text: str, filename: str) -> str:
    if filename not in id2ttlP.keys():
        preprocess_dataset(filename)
    
    triples = get_snippet_triple(query_id, filename)
    if len(triples) == 0:
        return -1
    keywords = sum((keyword.split('/') for keyword in query_text.split()), [])
    score = {
        "SkmRep": skmRep(filename, triples),
        "KwRel": kwRel(filename, keywords, triples),
        "EntRep": entRep(filename, triples),
        "DescRep": descRep(filename, triples),
        "QryRel": qryRel(filename, keywords, triples)
    }

    return score


def main():
    qrels = data_util.get_qrels()
    id2query = data_util.get_id_to_query()
    dataset2filename = data_util.get_dataset_to_file()

    query_filename_list = []
    for qrel in qrels:
        for filename in dataset2filename[qrel[1]]:
            if os.path.exists(f"{data_util.collection_base}/{filename}/term.tsv"):
                query_filename_list.append((qrel[0], filename))
    
    skm_list = []
    kw_list = []
    ent_list = []
    des_list = []
    qry_list = []

    for query_id, filename in query_filename_list:
        score = get_score(query_id, id2query[query_id], filename)
        if score == -1:
            continue
        skm_list.append(score['SkmRep'])
        if score['KwRel'] != -1:
            kw_list.append(score['KwRel'])
        ent_list.append(score['EntRep'])
        des_list.append(score['DescRep'])
        if score['QryRel'] != -1:
            qry_list.append(score['QryRel'])
        print(f'========={len(skm_list)}_{max_size}============')
        print(f'Avg SkmRep: {sum(skm_list) / len(skm_list)}')
        print(f'Avg EntRep: {sum(ent_list) / len(ent_list)}')
        print(f'Avg DescRep: {sum(des_list) / len(des_list)}')
        print(f'Avg KwRel: {sum(kw_list) / len(kw_list)}')
        print(f'Avg QryRel: {sum(qry_list) / len(qry_list)}')

    print(f'=========All_{max_size}============')
    print(f'Avg SkmRep: {sum(skm_list) / len(skm_list)}')
    print(f'Avg EntRep: {sum(ent_list) / len(ent_list)}')
    print(f'Avg DescRep: {sum(des_list) / len(des_list)}')
    print(f'Avg KwRel: {sum(kw_list) / len(kw_list)}')
    print(f'Avg QryRel: {sum(qry_list) / len(qry_list)}')

if __name__ == "__main__":
    main()