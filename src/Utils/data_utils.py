import pandas as pd
import csv
import os
from string_utils import deal_iri
import pymysql

class DataUtil:

    def __init__(self, test_collection_name: str) -> None:
        self.collection_name = test_collection_name
        self.collection_base = os.path.join(os.path.dirname(__file__), f"../../data/test_collection_data/{test_collection_name}")

    def get_id_to_query(self):
        id2query = {}
        query_path = os.path.join(self.collection_base, 'query.txt')
        with open(query_path, 'r') as f:
            for line in f.readlines():
                query_id, query_text = line.strip().split('\t')
                id2query[query_id] = query_text
        return id2query

    def get_qrels(self):
        qrels_list = []
        qrels_path = os.path.join(self.collection_base, 'qrels.txt')
        with open(qrels_path, 'r') as f:
            for line in f.readlines():
                query_id, dataset_id, rel_score, split = line.strip().split('\t')
                qrels_list.append((query_id, dataset_id, rel_score, split))
        return qrels_list

    def get_dataset_to_file(self):
        dataset_filename_list = []
        filename_path = os.path.join(self.collection_base, 'dataset_filename.txt')
        with open(filename_path, 'r') as f:
            for line in f.readlines():
                dataset_id, filename = line.strip().split('\t')
                dataset_filename_list.append((dataset_id, filename))
        return dataset_filename_list

    def get_triples_by_filename(self, filename: str):
        triple_path = os.path.join(self.collection_base, filename, 'triple.tsv')

        chunksize = 10**6
        triple_list = []
        for chunk in pd.read_csv(triple_path, sep='\t', header=None, names=['subject', 'predicate', 'object'], dtype=int, chunksize=chunksize,na_values=[], keep_default_na=False, quoting=3, low_memory=False):
            triple_list.extend(chunk.values.tolist())
    
        return triple_list
    
    def get_terms_by_filename(self, filename: str):
        term_path = os.path.join(self.collection_base, filename, 'term.tsv')
        term_list = []
        
        chunksize = 10**6  
        term_list = []
        for chunk in pd.read_csv(term_path, sep='\t', header=None, names=['term_id', 'label', 'kind'],
                                 dtype=str, chunksize=chunksize,
                                 na_values=[], keep_default_na=False, quoting=3, index_col='term_id', low_memory=False):
            chunk['term_id'] = chunk['term_id'].astype(int)
            chunk['label'] = chunk['label'].apply(deal_iri)
            term_list.extend(chunk.values.tolist())

        return term_list
    
    def get_id2term_by_filename(self, filename: str):
        term_path = os.path.join(self.collection_base, filename, 'term.tsv')
        id2term = {}
        
        chunksize = 10**6 
        for chunk in pd.read_csv(term_path, sep='\t', header=None, names=['term_id', 'label', 'kind'],
                                 dtype=str, chunksize=chunksize,
                                 na_values=[], keep_default_na=False, quoting=3, index_col='term_id', low_memory=False):
            chunk['term_id'] = chunk['term_id'].astype(int)
            chunk['label'] = chunk['label'].apply(deal_iri)
            id2term.update(chunk.set_index('term_id')['label'].to_dict())
        
        return id2term

    def insert_data(self, filename: str, data_type: str, data_list):
        assert data_type in ['edp', 'edp_count', 'entity', 'property']
        file_path = os.path.join(self.collection_base, filename, f'{data_type}.tsv')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            for data in data_list:
                writer.writerow(data)
    
    def get_edp_count_by_filename(self, filename: str):
        data_path = os.path.join(self.collection_base, filename, 'edp_count.tsv')
        assert os.path.exists(data_path)
        
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if len(row) == 3:
                    data_list.append((int(row[1]), int(row[2])))
        
        return data_list

    def get_property_by_filename(self, filename: str):
        data_path = os.path.join(self.collection_base, filename, 'property.tsv')
        assert os.path.exists(data_path)
        
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if len(row) == 3:
                    data_list.append((int(row[1]), int(row[2])))
        
        return data_list
    
    def get_entity_by_filename(self, filename: str):
        data_path = os.path.join(self.collection_base, filename, 'entity.tsv')
        assert os.path.exists(data_path)
        
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if len(row) == 5:
                    data_list.append((int(row[1]), int(row[2]), int(row[3]), int(row[4])))
        
        return data_list
    
    def get_edp_by_filename(self, filename: str):
        data_path = os.path.join(self.collection_base, filename, 'edp.tsv')
        assert os.path.exists(data_path)
        
        data_list = []
        with open(data_path, 'r', encoding='utf-8') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for row in reader:
                if len(row) == 4:
                    data_list.append((int(row[1]), int(row[2]), int(row[3])))
        
        return data_list
