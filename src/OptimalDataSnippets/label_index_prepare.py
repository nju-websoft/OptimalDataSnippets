
import os
import json
import traceback
import sys

from concurrent import futures

sys.path.append("..")
from Utils.data_utils import DataUtil
from Utils.string_utils import deal_sentence
from .config import test_collection, label_collection_path


store_base = label_collection_path


def process_dataset(data_utils: DataUtil, filename: str):
    term_list = data_utils.get_terms_by_filename(filename)
    collections = []
    for term in term_list:
        term_id = str(term[0])
        label = deal_sentence(term[1])
        if label != "":
            label = ' '.join(label.split())
        collections.append({"id": term_id, "contents": label})
    
    output_path = os.path.join(store_base, str(filename))
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(output_path, 'labelID.json'), 'w', encoding="utf-8", newline='\n') as writer:
        json.dump(collections, writer, ensure_ascii=False, indent=2)

def main():
    global dataset2filename
    data_util = DataUtil(test_collection)
    dataset2filename = data_util.get_dataset_to_file()
    filename_list = []
    for dataset_id, filenames in dataset2filename.items():
        for file_name in filenames:
            if os.path.exists(f"{data_util.collection_base}/{file_name}/term.tsv"):
                filename_list.append(file_name)
    worker_num = 10
    completed_tasks = 0
    total_tasks = len(filename_list)
    with futures.ProcessPoolExecutor(max_workers=worker_num) as executor:
        work_set = [executor.submit(process_dataset, data_util, filename) for filename in filename_list]
        for future in futures.as_completed(work_set):
            completed_tasks += 1
            if completed_tasks % 1000 == 0:
                print(f"Progress: {completed_tasks}/{total_tasks} tasks completed")
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()
        


if __name__ == "__main__":
    main()