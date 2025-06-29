"""
这个代码实现了从MMDetection到COCO格式预测框结果json
"""
from utils.utils import *
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SPOD',  help='choose dataset, Avon SPOD Sandiego MUUFLGulfport')
    dataset = parser.parse_args().dataset   # Avon MUUFLGulfport SanDiego SPOD
    dataset_result_path = os.path.join('./eval_result/', dataset + 'Result/HOD')
    mmdet_result_path = os.path.join(dataset_result_path, 'result')
    json_output_path = os.path.join(dataset_result_path, 'cocojson')
    os.makedirs(json_output_path, exist_ok=True)
    method_list = list_and_sort_subdirectories(mmdet_result_path)
    for method in method_list:
        print(method)
        result_pkl_path = os.path.join(mmdet_result_path, method, 'result.pkl')
        if not os.path.exists(result_pkl_path):
            pkl_files = [f for f in os.listdir(os.path.join(mmdet_result_path, method)) if f.endswith('.pkl')]
            if pkl_files:
                result_pkl_path = os.path.join(mmdet_result_path, method, pkl_files[0])
        pred_json_path = os.path.join(json_output_path, method+'.json')
        pkl_to_cocojson(result_pkl_path, pred_json_path, if_bbox_round=False, sorce_th=0.01)






