"""
这个代码用于评估高光谱目标检测结果（矩形框预测）
包含 coco AP AR
"""
from utils.utils import *
import json
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SPOD',  help='choose dataset, Avon SPOD Sandiego MUUFLGulfport')
    parser.add_argument('--eval_methods', default='HTD', help='choose methods , HOD HTD')
    parser.add_argument('--eval_f1', default=True, help='')

    dataset = parser.parse_args().dataset   # Avon MUUFLGulfport SanDiego SPOD
    eval_methods = parser.parse_args().eval_methods  # 'HOD'  'HTD'
    eval_f1 = parser.parse_args().eval_f1

    if dataset == 'SPOD':
        data_path = os.path.join('./datasets/', 'SPOD_30b_8c')
    else:
        data_path = os.path.join('./datasets/', dataset)
    result_json_path = os.path.join('./eval_result/', dataset + 'Result/', eval_methods, 'cocojson')
    output_path = os.path.join('./eval_result/', dataset + 'Result/eval')
    os.makedirs(output_path, exist_ok=True)
    json_files = [f for f in os.listdir(result_json_path) if f.endswith('.json')]
    gt_json_path = os.path.join(data_path, 'annotations', 'test.json')

    if dataset in ['Avon', 'SPOD']:
        """
        精细边界框评估  
        """
        txt_list = []
        for json_file in json_files:
            method = json_file.replace('.json', '')
            eval_txt = os.path.join(output_path, method + '.txt')
            try:
                eval_ap_ar(os.path.join(result_json_path, json_file), gt_json_path, eval_txt)
                txt_list.append(method + '.txt')
            except:
                pass
        with open(gt_json_path, 'r', encoding='utf-8') as file:
            gt_data = json.load(file)
        cat_list = [cat_i['name'] for cat_i in gt_data['categories']]
        all_method_result_txt = os.path.join(output_path, '0'+eval_methods+'apar.txt')
        write_comparedmethod_apar(all_method_result_txt, cat_list, output_path, txt_list)
        if eval_f1:
            score_th = 0.2
            iou_th = 0.25
            all_method_f1result_txt = os.path.join(output_path, '0' + eval_methods + 'reprf1.txt')
            log = open(all_method_f1result_txt, 'w')
            print_log("method, GTNum, PredictionNum, TruePositiveNum, FalseAlarmNum, Recall, Precision, F1, FalseAlarmRate", log)
            for json_file in json_files:
                method = json_file.replace('.json', '')
                try:
                    gt_num, det_num, get_num, Recall, Precision, F1, FalseAlarmRate =\
                        re_pr_f1(os.path.join(result_json_path, json_file), gt_json_path, score_th=score_th, iou_th=iou_th)
                    print_log('{:}, {:d}, {:d}, {:d}, {:d}, {:6f}, {:6f}, {:6f}, {:5e}'.format(method, gt_num, det_num,
                        get_num, det_num-get_num, Recall, Precision, F1, FalseAlarmRate), log)
                except:
                    pass

    if dataset in ['MUUFLGulfport', 'SanDiego']:
        """
        不精细边界框评估  
        """
        output_path = os.path.join('./eval_result/', dataset + 'Result/eval')
        os.makedirs(output_path, exist_ok=True)
        txt_list = []
        for json_file in json_files:
            method = json_file.replace('.json', '')
            eval_txt = os.path.join(output_path, method + '.txt')
            log = open(eval_txt, 'w')
            if dataset == 'SanDiego':
                eval_ap_ar_mini(os.path.join(result_json_path, json_file), gt_json_path, log=log,
                                iouThr=0.25, write_type='w', iouType='bbox')
            else:
                eval_ap_ar_mini(os.path.join(result_json_path, json_file), gt_json_path, log=log,
                                iouThr=0.1, write_type='w', iouType='dualwindow')
            txt_list.append(method + '.txt')
        with open(gt_json_path, 'r', encoding='utf-8') as file:
            gt_data = json.load(file)
        cat_list = [cat_i['name'] for cat_i in gt_data['categories']]
        all_method_result_txt = os.path.join(output_path, '0' + eval_methods + 'apar.txt')
        write_comparedmethod_apar_mini(all_method_result_txt, cat_list, output_path, txt_list)
        if eval_f1:
            score_th = 0.2
            if dataset == 'SanDiego':
                iou_th = 0.25
            else:
                iou_th = 0.1
            all_method_f1result_txt = os.path.join(output_path, '0' + eval_methods + 'reprf1.txt')
            log = open(all_method_f1result_txt, 'w')
            print_log("method, GTNum, PredictionNum, TruePositiveNum, FalseAlarmNum, Recall, Precision, F1, FalseAlarmRate", log)
            for json_file in json_files:
                method = json_file.replace('.json', '')
                gt_num, det_num, get_num, Recall, Precision, F1, FalseAlarmRate =\
                    re_pr_f1(os.path.join(result_json_path, json_file), gt_json_path, score_th=score_th, iou_th=iou_th)
                print_log('{:}, {:d}, {:d}, {:d}, {:d}, {:6f}, {:6f}, {:6f}, {:5e}'.format(method, gt_num, det_num,
                    get_num, det_num-get_num, Recall, Precision, F1, FalseAlarmRate), log)
