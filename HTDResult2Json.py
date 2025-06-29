"""
这个代码提供了从HTD检测预测图结果到COCO格式预测框结果的转换示例
以Avon数据集为例展示。
HTD的检测分数图保存成mat，形状为Nc*H*W，其中Nc为目标类别数目，意味着经过Nc次光谱匹配算法得到。
"""

import os.path
from utils.utils import *
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import cv2
import numpy as np
import json


def hotmap_seg(result_path, data_path, dataset='SPOD'):
    # 查找HTD检测分数图结果文件
    data_names = find_and_sort_files(result_path, ['mat'])

    # 确定测试图像H，W和目标类别数cat_num，这里默认所有测试图像有相同的H和W
    results = sio.loadmat(os.path.join(result_path, data_names[0]))['result']
    cat_num, m, n = results.shape
    i_num = len(data_names)
    results_dict = {}
    gt_dict = {}
    for cat_i in range(cat_num):
        results_dict[cat_i] = np.zeros((i_num, m, n))
        gt_dict[cat_i] = np.zeros((i_num, m, n))

    for d_i, data_name in enumerate(data_names):
        # 读取HTD检测分数图
        results = sio.loadmat(os.path.join(result_path, data_name))['result']
        # 读取mask真值 形状为同样为Nc*H*W，
        if dataset is not 'SPOD':
            gts = sio.loadmat(os.path.join(data_path, 'mask_gt', 'test' + data_name))['gt']
        else:
            gts = sio.loadmat(os.path.join(data_path, 'mask_gt', data_name))['gt']

        for cat_i in range(cat_num):
            results_dict[cat_i][d_i] = results[cat_i]
            gt_dict[cat_i][d_i] = gts[cat_i]
    results_norm_dict = {}
    seg_th_dict = {}

    # 每一类别确定按最佳分割IoU方式分别确定分割阈值
    for cat_i in range(cat_num):
        gt = gt_dict[cat_i]
        res = results_dict[cat_i]
        res_min = np.nanmin(res)
        res = np.nan_to_num(res, nan=res_min)
        res_max = np.max(res)
        seg_th = 0
        iou_max = 0
        if res_min != res_max:
            res = (res - res_min) / (res_max - res_min)
            for th_i in range(100, 0, -1):
                seg_map = res * 100 >= th_i
                sum_map = seg_map + gt
                i_map = sum_map == 2
                u_map = sum_map > 0
                iou = np.sum(i_map) / np.sum(u_map)
                if iou > iou_max:
                    iou_max = iou
                    seg_th = th_i
        else:
            res = np.zeros_like(res)
        results_norm_dict[cat_i] = res
        seg_th_dict[cat_i] = seg_th
    return results_norm_dict, gt_dict, seg_th_dict


def hotmap_to_cocojson(results_norm_dict, seg_th_dict, result_path, pred_json_path):
    det_json = []
    data_names = find_and_sort_files(result_path, ['mat'])
    results = sio.loadmat(os.path.join(result_path, data_names[0]))['result']
    cat_num, m, n = results.shape
    data_num = len(data_names)
    for cat_i in range(cat_num):
        res = results_norm_dict[cat_i]
        res_min = np.min(res)
        res_max = np.max(res)
        if res_min != res_max:
            for d_i in range(data_num):
                res_current = res[d_i]
                seg_map = res_current * 100 >= seg_th_dict[cat_i]
                seg_map = seg_map.astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_map, connectivity=8)
                for i in range(1, num_labels):
                    label_mask = np.uint8(labels == i)
                    score = np.max(res_current[label_mask])
                    contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        bbox = [x, y, w, h]
                        annotation = {
                            'id': int(len(det_json) + 1),
                            'image_id': int(data_names[d_i].replace('.mat', '')),
                            'category_id': int(cat_i + 1),
                            'bbox': bbox,
                            'score': score,
                            'iscrowd': 0,
                        }
                        det_json.append(annotation)
    with open(pred_json_path, 'w') as f:
        json.dump(det_json, f)
    f.close()


def eval_score_map(results_norm_dict, gt_dict, seg_th_dict, txt):
    if txt is not '':
        log = open(txt, 'w')
    else:
        log = txt
    cat_num = len(results_norm_dict.keys())
    iou_list = []
    auc_list = []
    for cat_i in range(cat_num):
        res = results_norm_dict[cat_i]
        res_min = np.min(res)
        res_max = np.max(res)
        if res_min != res_max:
            gt = gt_dict[cat_i]
            seg_th = seg_th_dict[cat_i]
            seg_map = res * 100 >= seg_th
            sum_map = seg_map + gt
            i_map = sum_map == 2
            u_map = sum_map > 0
            iou = np.sum(i_map) / np.sum(u_map)
            auc = roc_auc_score(gt.reshape(-1), res.reshape(-1))
        else:
            iou = 0
            auc = 0.5
        print_log('%s iou: %.6f, auc: %.6f' % ('Cat' + str(cat_i + 1).ljust(10), iou, auc,), log)
        iou_list.append(iou)
        auc_list.append(auc)
    miou = np.mean(np.array(iou_list))
    mauc = np.mean(np.array(auc_list))
    print_log('%s iou: %.6f, auc: %.6f' % ('overall'.ljust(10), miou, mauc,), log)


if __name__ == '__main__':
    dataset = 'Avon'
    data_path = './datasets/'
    htd_result_path = './eval_result/AvonResult/HTD/result/'
    json_output_path = './eval_result/AvonResult/HTD/cocojson'
    os.makedirs(json_output_path, exist_ok=True)
    method_list = list_and_sort_subdirectories(htd_result_path)
    for method in method_list:
        print(method)
        result_path = os.path.join(htd_result_path, method)
        pred_json_path = os.path.join(json_output_path, method + '.json')
        results_norm_dict, gt_dict, seg_th_dict = hotmap_seg(result_path, data_path, dataset)
        hotmap_to_cocojson(results_norm_dict, seg_th_dict, result_path, pred_json_path)
        eval_score_map(results_norm_dict, gt_dict, seg_th_dict, '')

