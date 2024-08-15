import os
import json
import math
import sys
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from collections import defaultdict

sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence
from Utils.data_structure import WeightedTriple, ConnectedComponent
from .config import test_collection, snippet_base, snippet_max_size, label_index_path


store_base = snippet_base
max_size = snippet_max_size
data_util = DataUtil(test_collection)
index_path = label_index_path


def get_basic_info(filename):
    global id2term
    global triple_list, entity_list
    global edp2fp, edp2bp, edp2size
    global property_count_list, edp_count_list
    global term2triples
    global edp_weight, property_weight, entity_weight
    global entity2edp, edp2entities, best_data_rep_triple

    origin_triple_list = data_util.get_triples_by_filename(filename)
    edp_count_list = data_util.get_edp_count_by_filename(filename)
    edp_list = data_util.get_edp_by_filename(filename)
    entity_list= data_util.get_entity_by_filename(filename)
    property_count_list = data_util.get_property_by_filename(filename)
    
    id2term = data_util.get_id2term_by_filename(filename)
    
    edp2fp = defaultdict(list)
    edp2bp = defaultdict(list)
    edp2size = defaultdict(int)
    for edp, kind, term_id in edp_list:
        if kind == 1:
            edp2bp[edp].append(term_id)
        else:
            edp2fp[edp].append(term_id)
        edp2size[edp] += 1

    edp_total = sum(count for _, count in edp_count_list)
    edp_weight = {edp_id: count / edp_total for edp_id, count in edp_count_list}

    property_total = sum(count for _, count in property_count_list)
    property_weight = {property_id: count / property_total for property_id, count in property_count_list}

    log_sum_in = 0
    log_sum_out = 0
    entity2edp = dict()
    edp2entities = defaultdict(list)
    for entity_id, in_degree, out_degree, edp in entity_list:
        log_sum_in += math.log(in_degree + 1)
        log_sum_out += math.log(out_degree + 1)
        entity2edp[entity_id] = edp
        edp2entities[edp].append(entity_id)
        
    log_sum_in = 1 if log_sum_in == 0 else log_sum_in
    log_sum_out = 1 if log_sum_out == 0 else log_sum_out
    entity_weight = {entity_id: math.log(in_degree + 1) / log_sum_in + math.log(out_degree + 1) / log_sum_out for entity_id, in_degree, out_degree, _ in entity_list}
    
    term2triples = defaultdict(list)
    triple_list = []
    best_data_rep_triple = WeightedTriple(0, 0, 0)
    for subject, predicate, object in origin_triple_list:
        prpW = property_weight[predicate]
        sedpW = 0
        sentW = 0
        oedpW = 0
        oentW = 0
        if subject in entity2edp:
            sedpW = edp_weight[entity2edp[subject]] / edp2size[entity2edp[subject]]
            sentW = entity_weight[subject]

        if object in entity2edp:
            oedpW = edp_weight[entity2edp[object]] / edp2size[entity2edp[object]]
            oentW = entity_weight[object]
        
        triple = WeightedTriple(subject, predicate, object, prpW=prpW, kwsW=0, sentW=sentW, sedpW=sedpW, oentW=oentW, oedpW=oedpW)
        triple_list.append(triple)
        term2triples[subject].append(triple)
        term2triples[predicate].append(triple)
        term2triples[object].append(triple) 
        if triple.data_weight > best_data_rep_triple.data_weight:
            best_data_rep_triple = triple



def generate_ours(query_id, filename):
    global store_base, max_size
    global id2query, term2triples
    
    get_basic_info(filename)
    covered_property = set()
    covered_entity = set()
    covered_edp_to_entity = {} 
    cc = ConnectedComponent()

    def update_edp_info(entity, predicate, edp_key, covered_fp=False, covered_bp=False):
        covered_entity.add(entity)
        if edp_key not in covered_edp_to_entity:
            covered_edp_to_entity[edp_key] = {
                'entity': entity, 
                'covered_fp': set(), 
                'covered_bp': set()
            }
        if entity == covered_edp_to_entity[edp_key]['entity']:
            if covered_fp:
                covered_edp_to_entity[edp_key]['covered_fp'].add(predicate)
            elif covered_bp:
                covered_edp_to_entity[edp_key]['covered_bp'].add(predicate)

    def update_cover_list(selected_triple: WeightedTriple):
        covered_property.add(selected_triple.predicate)
        
        if selected_triple.subject in entity_weight:
            s_edp = entity2edp[selected_triple.subject]
            update_edp_info(selected_triple.subject, selected_triple.predicate, s_edp, covered_fp=True)
        
        if selected_triple.object in entity_weight:
            o_edp = entity2edp[selected_triple.object]
            update_edp_info(selected_triple.object, selected_triple.predicate, o_edp, covered_bp=True)

    def compute_score(triple: WeightedTriple):
        cover = 0
        if triple.predicate not in covered_property:
            cover += triple.prpW
        if triple.subject in entity_weight and triple.subject not in covered_entity:
            cover += triple.sentW
            s_edp = entity2edp[triple.subject]
            if s_edp not in covered_edp_to_entity or (triple.subject == covered_edp_to_entity[s_edp]['entity'] and triple.predicate not in covered_edp_to_entity[s_edp]['covered_fp']):
                cover += triple.sedpW
        if triple.object in entity_weight and triple.object not in covered_entity:
            cover += triple.oentW
            o_edp = entity2edp[triple.object]
            if o_edp not in covered_edp_to_entity or (triple.object == covered_edp_to_entity[o_edp]['entity'] and triple.predicate not in covered_edp_to_entity[o_edp]['covered_bp']):
                cover += triple.oedpW
        # cohesion
        # give 2 times weight
        cover += 2 * cc.cover_gain(triple) / max_size
        return cover

    result_candidate_list = []
    searcher = LuceneSearcher(os.path.join(index_path, str(filename)))
    # first stage: relevance-constraint triple selection
    query_text = id2query[query_id]
    keywords = list(set(sum((keyword.split('/') for keyword in query_text.split()), [])))
    kws2term = dict()
    for kw in keywords:
        hits = searcher.search(kw)
        terms = [int(hit.docid) for hit in hits]
        if len(terms) > 0:
            kws2term[kw] = terms
    if len(kws2term) != 0:
        related_terms = set()
        term_to_keywords = defaultdict(set)
        
        for keyword in kws2term:
            for term in kws2term[keyword]:
                term_to_keywords[term].add(keyword)
            related_terms.update(kws2term[keyword])
        
        triple_to_terms = defaultdict(set)
        for term in related_terms:
            if term in term2triples:
                for triple in term2triples[term]:
                    triple_to_terms[triple].add(term)

        covered_keywords = set()
        target_size = min(max_size, len(kws2term))
        triple_to_keywords = defaultdict(set)
        
        pre_insert_flag = len(kws2term) < snippet_max_size
        for triple, terms in triple_to_terms.items():
            for term in terms:
                triple_to_keywords[triple].update(term_to_keywords[term])
            if not pre_insert_flag and len(triple_to_keywords[triple]) > 1:
                pre_insert_flag = True
        
        if pre_insert_flag:
            result_candidate_list.append(best_data_rep_triple)
            cc.add(best_data_rep_triple)
            update_cover_list(best_data_rep_triple)
        while len(result_candidate_list) < max_size and len(covered_keywords) < target_size:
            best_triple = None
            best_cover = 0
            best_score = 0
            
            for triple, keywords in triple_to_keywords.items():
                cover = len(keywords - covered_keywords)
                if cover > best_cover:
                    best_cover = cover
                    score = compute_score(triple)
                    if score > best_score:
                        best_score = score
                        best_triple = triple
            
            if best_triple is None:
                break

            result_candidate_list.append(best_triple)
            cc.add(best_triple)
            covered_keywords.update(triple_to_keywords[best_triple])
            update_cover_list(best_triple)
            del triple_to_keywords[best_triple]
    # second stage: greedy triple selection
    while len(result_candidate_list) < max_size:
        best_triple = None
        best_cover = 0
        for triple in triple_list:
            cover = compute_score(triple)
            if cover > best_cover:
                best_cover = cover
                best_triple = triple

        if best_triple is None:
            break
        
        result_candidate_list.append(best_triple)
        cc.add(best_triple)
        update_cover_list(best_triple)
        
    store_path = os.path.join(store_base, f"{query_id}_{filename}")
    if not os.path.exists(store_path):
        os.mkdir(store_path)

    candidate_triple_list = []
    for rank_num, triple in enumerate(result_candidate_list):
        content = ' '.join([id2term.get(triple.subject, ''), id2term.get(triple.predicate, ''), id2term.get(triple.object, '')])
        candidate_triple_list.append({'rank_num': rank_num + 1,
                                      'triples': str(triple),
                                      'text': ' '.join(deal_sentence(content).split())})
    with open(os.path.join(store_path, 'ours.json'), 'w') as f:
        json.dump(candidate_triple_list, f, ensure_ascii=False, indent=2)


def main():
    global id2query
    qrels = data_util.get_qrels()
    id2query = data_util.get_id_to_query()
    dataset2filename = data_util.get_dataset_to_file()

    query_filename_list = []
    for query_id, dataset_id, _, _ in qrels:
        for filename in dataset2filename[dataset_id]:
            snippet_path = os.path.join(store_base, f"{query_id}_{filename}", 'ours.json')
            if os.path.exists(snippet_path):
                continue
            if os.path.exists(f"{data_util.collection_base}/{filename}/term.tsv"):
                query_filename_list.append((query_id, filename))
    for query_id, filename in tqdm(query_filename_list):
        generate_ours(query_id, filename)


if __name__ == "__main__":
    main()