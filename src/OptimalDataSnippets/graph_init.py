# pre-compute entity and edp information for each data graph
# entity.tsv
# {filename}\t{entity_id}\t{in_degree}\t{out_degree}\t{edp_id}
# edp.tsv
# {filename}\t{edp_id}\t{kind}\t{term_id}
# edp_count.tsv
# {filename}\t{edp_id}\t{count}
# property.tsv
# {filename}\t{property_id}\t{count}


import os
from tqdm import tqdm
import sys

from .config import test_collection
sys.path.append("..")
from Utils.data_utils import DataUtil

class EEHandler:
    FORWARDPROPERTY = 0
    BACKWARDPROPERTY = 1

    def __init__(self, data_util: DataUtil, dataset_id: int, filename: str):
        self.term_list = None
        self.entity_count_edp_dict = None
        self.entity_list = None
        self.triple_list = None
        self.edp_dict = {}
        self.edp_count_dict = {}
        self.property_count_dict = {}
        self.property_list = None
        self.data_util = data_util
        self.filename = filename
        self.dataset_id = dataset_id

    def read_from_source(self):
        # triple_list:[[s, p, o], ...]
        self.triple_list = self.data_util.get_triples_by_filename(self.filename)
        # term_list:[[term_id, label, kind]] kind 0: entity, 1: literal, 2:property
        self.term_list = self.data_util.get_terms_by_filename(self.filename)
        
    def get_edp_entity_stored(self):
        self.read_from_source()

        self.entity_list = list(filter(lambda x: int(x[2]) == 0, self.term_list))
        # {entity_id: [in_degree, out_degree, edp_id]}
        self.entity_count_edp_dict = {entity[0]: [0, 0, -1] for entity in self.entity_list}
        self.property_list = list(filter(lambda x: int(x[2]) == 2, self.term_list))
        self.property_count_dict = {prop[0]: 0 for prop in self.property_list}
        entity_edp_dict = {entity[0]: [set(), set()] for entity in self.entity_list}  # {entity_id: [fp, bp]}
        for sub, pre, obj in self.triple_list:
            self.property_count_dict[pre] += 1
            if sub in self.entity_count_edp_dict:
                self.entity_count_edp_dict[sub][1] += 1  # out_degree
                entity_edp_dict[sub][1].add(pre)  # bp
            if obj in self.entity_count_edp_dict:
                self.entity_count_edp_dict[obj][0] += 1  # in_degree
                entity_edp_dict[obj][0].add(pre)  # fp

        edp_id = 0
        self.edp_dict = {}
        for entity_id, edp in entity_edp_dict.items():
            if len(edp[0]) == 0 and len(edp[1]) == 0:
                continue
            edp_tuple = (tuple(edp[0]), tuple(edp[1]))
            if edp_tuple not in self.edp_dict:
                edp_id += 1
                self.edp_dict[edp_tuple] = edp_id
            self.entity_count_edp_dict[entity_id][2] = self.edp_dict[edp_tuple]
            if self.edp_dict[edp_tuple] not in self.edp_count_dict:
                self.edp_count_dict[self.edp_dict[edp_tuple]] = 0
            self.edp_count_dict[self.edp_dict[edp_tuple]] += 1

        self.entity_count_edp_dict = {k: v for k, v in self.entity_count_edp_dict.items() if v[2] != -1}
        
        self.insert_to_dest()

    def insert_to_dest(self):
        edp_table_list = []
        edp_count_table_list = []
        entity_table_list = []
        property_table_list = []
        for edp_tuple, edp_id in self.edp_dict.items():
            fp_list = edp_tuple[0]
            bp_list = edp_tuple[1]
            edp_count = self.edp_count_dict[edp_id]
            for fp in fp_list:
                edp_table_list.append((self.dataset_id, edp_id, self.FORWARDPROPERTY, fp))
            for bp in bp_list:
                edp_table_list.append((self.dataset_id, edp_id, self.BACKWARDPROPERTY, bp))
            edp_count_table_list.append((self.dataset_id, edp_id, edp_count))
        for entity_id, [in_degree, out_degree, edp_id] in self.entity_count_edp_dict.items():
            entity_table_list.append((self.dataset_id, entity_id, in_degree, out_degree, edp_id))
        for property_id, count in self.property_count_dict.items():
            property_table_list.append((self.dataset_id, property_id, count))

        self.data_util.insert_data(self.filename, 'edp', edp_table_list)
        self.data_util.insert_data(self.filename, 'edp_count', edp_count_table_list)
        self.data_util.insert_data(self.filename, 'entity', entity_table_list)
        self.data_util.insert_data(self.filename, 'property', property_table_list)


def main():
    data_util = DataUtil(test_collection)
    dataset2files = data_util.get_dataset_to_file()
    for dataset_id in tqdm(dataset2files.keys()):
        for filename in dataset2files[dataset_id]:
            if not os.path.exists(f"{data_util.collection_base}/{filename}") or not os.path.exists(f"{data_util.collection_base}/{filename}/term.tsv"):
                continue
            handler = EEHandler(data_util, dataset_id, filename)
            handler.get_edp_entity_stored()

if __name__ == "__main__":
    main()
