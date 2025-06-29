import random
import os
import numpy as np
from pycocotools.coco import COCO, maskUtils
from .cocoeval import COCOeval
import json
import copy
import base64
import torch
import math
import pickle


def aviris_in_order(data, wave_center_list):
    indices = np.argsort(np.array(wave_center_list))
    data_new = data[:,:,indices]
    return data_new


def color_enhance(color_img, up_limit =3500, low_limit = 600,factor=(1,1,1)):
    for i in range(3):
        color_img[:,:,i] = color_img[:,:,i] *factor[i]
    new_img = (color_img - low_limit) / up_limit
    new_img[new_img > 1] = 1
    new_img[new_img < 0] = 0
    return new_img


def seed_set(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def cutimage_and_decide(data, img_size, row_start, col_start, up_limit =None):
    IsSave = False
    if isinstance(data, np.ndarray):
        img = data[row_start:(row_start + img_size), :, :][:, col_start:(col_start + img_size), :]
    else:
        img = aviris_in_order(data[row_start:(row_start + img_size), :, :][:, col_start:(col_start + img_size), :],
                              data.bands.centers)
    # if np.sum(img==-50) ==0:
    if np.sum(img==-50) ==0:
        if np.sum(img == 55537) == 0:
             if up_limit != None:
                if np.sum(img>up_limit) ==0:
                    IsSave = True
             else:
                 IsSave = True
    return img, IsSave


def xywhtoxxyy(bbox1):
    x, y, w, h =bbox1
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def bbox_round(x1, y1, x2, y2):
    x1_new =torch.round(x1)
    x2_new = torch.round(x2)
    y1_new = torch.round(y1)
    y2_new = torch.round(y2)
    if x1_new==x2_new:
        x1_new = torch.floor((x1+x2)/2)
        x2_new = torch.ceil((x1+x2)/2)
    if y1_new==y2_new:
        y1_new = torch.floor((y1+y2)/2)
        y2_new = torch.ceil((y1+y2)/2)
    return x1_new, y1_new, x2_new, y2_new


def float_bbox_round(x1, y1, x2, y2):
    x1_new = round(x1)
    x2_new = round(x2)
    y1_new = round(y1)
    y2_new = round(y2)
    if x1_new==x2_new:
        x1_new = math.floor((x1+x2)/2)
        x2_new = math.ceil((x1+x2)/2)
    if y1_new==y2_new:
        y1_new = math.floor((y1+y2)/2)
        y2_new = math.ceil((y1+y2)/2)
    return x1_new, y1_new, x2_new, y2_new


def calculate_iouv2(det_bboxes, gt_bboxes):
    """
    计算两组边界框之间的 IoU（交并比）

    参数:
    det_bboxes: numpy数组，形状为 (n, 4)，表示检测到的边界框，每行表示一个边界框，格式为 [x_min, y_min, width, height]
    gt_bboxes: numpy数组，形状为 (m, 4)，表示真实标签的边界框，每行表示一个边界框，格式为 [x_min, y_min, width, height]

    返回:
    iou: numpy数组，形状为 ( n, m)，表示每个真实标签边界框与每个检测到的边界框之间的IoU值
    """
    det_x1, det_y1 = det_bboxes[:, 0], det_bboxes[:, 1]
    det_x2, det_y2 = det_x1 + det_bboxes[:, 2], det_y1 + det_bboxes[:, 3]
    gt_x1, gt_y1 = gt_bboxes[:, 0], gt_bboxes[:, 1]
    gt_x2, gt_y2 = gt_x1 + gt_bboxes[:, 2], gt_y1 + gt_bboxes[:, 3]
    inter_x1 = np.maximum(det_x1.reshape(-1, 1), gt_x1)
    inter_y1 = np.maximum(det_y1.reshape(-1, 1), gt_y1)
    inter_x2 = np.minimum(det_x2.reshape(-1, 1), gt_x2)
    inter_y2 = np.minimum(det_y2.reshape(-1, 1), gt_y2)
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
    det_area = det_bboxes[:, 2] * det_bboxes[:, 3]
    gt_area = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    union_area = det_area.reshape(-1, 1) + gt_area - inter_area
    iou = inter_area / union_area

    return iou


def calculate_iou(bbox_gt, bboxes):
    """
    计算两组边界框之间的 IoU（交并比）

    参数:
    bbox_gt: numpy数组，形状为 (n, 4)，表示真实标签的边界框，每行表示一个边界框，格式为 [x_min, y_min, width, height]
    bboxes: numpy数组，形状为 (m, 4)，表示检测到的边界框，每行表示一个边界框，格式为 [x_min, y_min, width, height]
    返回:
    iou: numpy数组，形状为 ( n, m)，表示每个真实标签边界框与每个检测到的边界框之间的IoU值
    """
    bbox_gt_x2 = bbox_gt[0] + bbox_gt[2]
    bbox_gt_y2 = bbox_gt[1] + bbox_gt[3]
    bboxes_x2 = bboxes[:, 0] + bboxes[:, 2]
    bboxes_y2 = bboxes[:, 1] + bboxes[:, 3]
    intersection_x1 = np.maximum(bbox_gt[0], bboxes[:, 0])
    intersection_y1 = np.maximum(bbox_gt[1], bboxes[:, 1])
    intersection_x2 = np.minimum(bbox_gt_x2, bboxes_x2)
    intersection_y2 = np.minimum(bbox_gt_y2, bboxes_y2)
    intersection_area = np.maximum(intersection_x2 - intersection_x1, 0) * np.maximum(intersection_y2 - intersection_y1,
                                                                                      0)
    bbox_gt_area = bbox_gt[2] * bbox_gt[3]
    bboxes_area = bboxes[:, 2] * bboxes[:, 3]
    union_area = bbox_gt_area + bboxes_area - intersection_area
    iou = intersection_area / union_area
    return iou


def RLE2base64(mask_dict):
    new_mask_dict = copy.deepcopy(mask_dict)
    new_mask_dict['counts'] =base64.b64encode(mask_dict['counts']).decode('utf-8')
    return new_mask_dict


def base642RLE(mask_dict):
    new_mask_dict = copy.deepcopy(mask_dict)
    new_mask_dict['counts'] =base64.b64decode(mask_dict['counts'])
    return new_mask_dict


def pkl_to_cocojson(pkl_file, json_file, if_bbox_round=True, sorce_th=0):
    with open(pkl_file, 'rb') as f:
        result_list = pickle.load(f)
    det_json = []
    for result_img in result_list:
        image_id = result_img['img_id']
        preb_num = len(result_img['pred_instances']['labels'])
        for i in range(preb_num):
            x1, y1, x2, y2 = result_img['pred_instances']['bboxes'][i]
            if if_bbox_round:
                x1, y1, x2, y2 = bbox_round(x1, y1, x2, y2)
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            width = x2 - x1
            height = y2 - y1
            annotation = {
                    'id': int(len(det_json) + 1),
                    'image_id': int(image_id),
                    'category_id': int(result_img['pred_instances']['labels'][i]+1),
                    'bbox': [x1, y1, width, height],
                    'score': float(result_img['pred_instances']['scores'][i]),
                }
            if 'mask' in result_img['pred_instances'].keys():
                annotation['mask'] = RLE2base64(result_img['pred_instances']['masks'][i])
            if annotation['score'] >sorce_th:
                det_json.append(annotation)
    with open(json_file, 'w') as f:
        json.dump(det_json, f)


def find_and_sort_files(folder_path, file_extensions):
    if not os.path.exists(folder_path):
        print(f" Folder path'{folder_path}' does not exist.")
        return []
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    filtered_files = [f for f in all_files if any(f.endswith(ext) for ext in [f'.{ext}' for ext in file_extensions])]
    sorted_files = sorted(filtered_files)
    return sorted_files


def list_and_sort_subdirectories(path):
    items = os.listdir(path)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(path, item))]
    sorted_subdirectories = sorted(subdirectories)
    return sorted_subdirectories


def print_log(print_string, log):
    print("{:}".format(print_string))
    if log is not '':
        log.write('{:}\n'.format(print_string))
        log.flush()


def eval_ap_ar(pred_json_path,gt_json_path, txt='', write_type='w', iouType='bbox', eval_cat=True):
    if txt is not '':
        log = open(txt, write_type)
    else:
        log = txt
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    print_log('box_num'+str(len(coco_pred.anns)), log)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType)
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_labels = [category['name'] for category in categories]
    catId_list = [category['id'] for category in categories]
    coco_eval.params.maxDets = [300, 300, 300]
    coco_eval.params.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap5095 = coco_eval._summarize(1, maxDets=300)
    ap50 = coco_eval._summarize(1, iouThr=0.5,maxDets=300)
    ap75 = coco_eval._summarize(1, iouThr=0.75, maxDets=300)
    ap95 = coco_eval._summarize(1, iouThr=0.95, maxDets=300)
    ar5095 = coco_eval._summarize(2, maxDets=300)
    ar50 = coco_eval._summarize(2, iouThr=0.5, maxDets=300)
    ar75 = coco_eval._summarize(2, iouThr=0.75, maxDets=300)
    ar95 = coco_eval._summarize(2, iouThr=0.95, maxDets=300)
    coco_eval.params.iouThrs = np.linspace(0.1, 0.25, 2, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap10 = coco_eval._summarize(1, iouThr=0.1,maxDets=300)
    ap25 = coco_eval._summarize(1, iouThr=0.25, maxDets=300)
    ar10 = coco_eval._summarize(2, iouThr=0.1, maxDets=300)
    ar25 = coco_eval._summarize(2, iouThr=0.25, maxDets=300)
    print_log('%s ap5095: %.6f, ap10: %.6f, ap25: %.6f, ap50: %.6f, ap75: %.6f, ap95: %.6f, ar5095: %.6f, ar10: %.6f, ar25: %.6f, ar50: %.6f, ar75: %.6f, ar95: %.6f,'%\
          ('overall'.ljust(10), ap5095,ap10,ap25,ap50,ap75,ap95,ar5095,ar10,ar25,ar50,ar75,ar95),log)
    if eval_cat:
        for catId in catId_list:
            category_name = coco_gt.loadCats(catId)[0]['name']
            coco_eval.params.catIds = [catId]
            coco_eval.params.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
            coco_eval.evaluate()
            coco_eval.accumulate()
            ap5095 = coco_eval._summarize(1, maxDets=300)
            ap50 = coco_eval._summarize(1, iouThr=0.5, maxDets=300)
            ap75 = coco_eval._summarize(1, iouThr=0.75, maxDets=300)
            ap95 = coco_eval._summarize(1, iouThr=0.95, maxDets=300)
            ar5095 = coco_eval._summarize(2, maxDets=300)
            ar50 = coco_eval._summarize(2, iouThr=0.5, maxDets=300)
            ar75 = coco_eval._summarize(2, iouThr=0.75, maxDets=300)
            ar95 = coco_eval._summarize(2, iouThr=0.95, maxDets=300)
            coco_eval.params.iouThrs = np.linspace(0.1, 0.25, 2, endpoint=True)
            coco_eval.evaluate()
            coco_eval.accumulate()
            ap10 = coco_eval._summarize(1, iouThr=0.1, maxDets=300)
            ap25 = coco_eval._summarize(1, iouThr=0.25, maxDets=300)
            ar10 = coco_eval._summarize(2, iouThr=0.1, maxDets=300)
            ar25 = coco_eval._summarize(2, iouThr=0.25, maxDets=300)
            print_log(
                '%s ap5095: %.6f, ap10: %.6f, ap25: %.6f, ap50: %.6f, ap75: %.6f, ap95: %.6f, ar5095: %.6f, ar10: %.6f, ar25: %.6f, ar50: %.6f, ar75: %.6f, ar95: %.6f,' % \
                (category_name.ljust(10), ap5095, ap10, ap25, ap50, ap75, ap95, ar5095, ar10, ar25, ar50, ar75, ar95),log)


def eval_ap_ar_mini(pred_json_path,gt_json_path, txt='',log = None, iouThr = 0.2, write_type='w', iouType='dualwindow'):
    if log is  None:
        if txt is not '':
            log = open(txt, write_type)
        else:
            log = txt
    coco_gt = COCO(gt_json_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    print_log('box_num'+str(len(coco_pred.anns)), log)
    coco_eval = COCOeval(coco_gt, coco_pred, iouType)
    categories = coco_gt.loadCats(coco_gt.getCatIds())
    category_labels = [category['name'] for category in categories]
    catId_list = [category['id'] for category in categories]
    coco_eval.params.maxDets = [300, 300, 300]
    coco_eval.params.iouThrs = np.linspace(iouThr, iouThr, 1, endpoint=True)
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap = coco_eval._summarize(1, iouThr=iouThr, maxDets=300)
    ar = coco_eval._summarize(2, iouThr=iouThr, maxDets=300)
    print_log('%s ap: %.6f, re: %.6f,'%\
          ('overall'.ljust(10), ap, ar), log)
    for catId in catId_list:
        category_name = coco_gt.loadCats(catId)[0]['name']
        coco_eval.params.catIds = [catId]
        coco_eval.params.iouThrs = np.linspace(iouThr, iouThr,1, endpoint=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        ap = coco_eval._summarize(1, iouThr=iouThr, maxDets=300)
        ar = coco_eval._summarize(2, iouThr=iouThr, maxDets=300)
        print_log('%s ap: %.6f, re: %.6f,' % \
                  (category_name.ljust(10), ap, ar), log)


def check_bboxes_within_limit(bbox_max_limit, bboxes):
    """
    判断边界框是否超过最大边界范围

    参数:
    bbox_max_limit: list，形状为 (4)，表示最大边界范围，格式为 [x_min, y_min, width, height]
    bboxes: numpy数组，形状为 (m, 4)，表示检测到的边界框，格式为 [x_min, y_min, width, height]
    返回:
    : numpy数组，形状为(m,)，每个元素为True表示对应的bbox未超过最大边界范围，为False表示bbox超过最大边界范围
    """
    x_min_limit, y_min_limit, width_limit, height_limit = bbox_max_limit
    xmax_limit = x_min_limit + width_limit
    ymax_limit = y_min_limit + height_limit
    x_maxes = bboxes[:, 0] + bboxes[:, 2]
    y_maxes = bboxes[:, 1] + bboxes[:, 3]
    within_limit = (bboxes[:, 0] >= x_min_limit) & \
                   (bboxes[:, 1] >= y_min_limit) & \
                   (x_maxes <= xmax_limit) & \
                   (y_maxes <= ymax_limit)

    return within_limit


def judging_prediction(gts, preds_in, score_th=0.2, iou_th=0.25, false_preds_cls=False, return_matched_pairs=False,
                       match_relu = 'IoUFirst'):
    """
    Parameters
    ----------
    gts
    preds_in
    score_th
    iou_th

    Returns
    -------
    true_preds 正确预测框
    false_preds 错误预测框 false_preds_redun+false_preds_cat+false_preds_pos
    matched_gt 被正确预测的真值
    unmatched_gt 未被预测的真值
    false_preds_redun 冗余的正确预测框
    false_preds_cat  位置匹配但类别未匹配的错误预测框
    false_preds_pos 位置未匹配的错误预测框
    match_relu  当有一个真值框多个匹配预测框损失的选取规则 'IoUFirst' IoU最大作为匹配结果 'ScoreFirst' 置信度分数最大作为匹配结果
    """
    preds = []
    for det in preds_in:
        if det['score'] >= score_th:
            preds.append(det)
    true_preds = []
    matched_gt = []
    unmatched_gt = []
    matched_pairs = []
    for gt in gts:
        bbox_gt = np.array(gt['bbox'])
        preds_cat = [item for item in preds if item['category_id']== gt['category_id']]
        bboxes_pred = np.array([item['bbox'] for item in preds_cat])
        scores_pred = np.array([item['score'] for item in preds_cat])
        pred = None
        if len(preds_cat) > 0:
            iou = calculate_iou(bbox_gt, bboxes_pred)
            if 'out_bbox' in gt:
                within_limit = check_bboxes_within_limit(gt['out_bbox'], bboxes_pred)
                iou[within_limit!=True] = 0
            if np.max(iou) >= iou_th:
                matched_gt.append(gt)
                if match_relu == 'IoUFirst':
                    match_index = np.argmax(iou)
                else:
                    scores_pred[iou<iou_th] = 0
                    match_index = np.argmax(scores_pred)
                pred = preds_cat[match_index]
                true_preds.append(pred)
                preds.remove(pred)
            else:
                unmatched_gt.append(gt)
        else:
            unmatched_gt.append(gt)
        if (pred is not None) and return_matched_pairs:
            matched_pairs.append({'gt':gt,'pred':pred})
    false_preds = preds
    if not false_preds_cls:
        if return_matched_pairs:
            return true_preds, false_preds, matched_gt, unmatched_gt, matched_pairs
        else:
            return true_preds, false_preds, matched_gt, unmatched_gt
    false_preds_redun = []
    false_preds_cat = []
    false_preds_pos = copy.deepcopy(false_preds)
    if len(false_preds) > 0:
        for gt in gts:
            bbox_gt = np.array(gt['bbox'])
            preds_cat = [item for item in false_preds_pos if item['category_id'] == gt['category_id']]
            bboxes_pred_cat = np.array([item['bbox'] for item in preds_cat])
            preds_other_cat = [item for item in false_preds_pos if item['category_id'] != gt['category_id']]
            bboxes_pred_other_cat = np.array([item['bbox'] for item in preds_other_cat])
            if len(preds_cat) > 0:
                iou = calculate_iou(bbox_gt, bboxes_pred_cat)
                if 'out_bbox' in gt:
                    within_limit = check_bboxes_within_limit(gt['out_bbox'], bboxes_pred_cat)
                    iou[within_limit != True] = 0
                idxes = np.where(iou >= iou_th)
                for idx in list(idxes[0]):
                    pred = preds_cat[idx]
                    false_preds_redun.append(pred)
                    false_preds_pos.remove(pred)
            if len(preds_other_cat) > 0:
                iou = calculate_iou(bbox_gt, bboxes_pred_other_cat)
                if 'out_bbox' in gt:
                    within_limit = check_bboxes_within_limit(gt['out_bbox'], bboxes_pred_cat)
                    iou[within_limit != True] = 0
                idxes = np.where(iou >= iou_th)
                for idx in list(idxes[0]):
                    pred = preds_other_cat[idx]
                    false_preds_cat.append(pred)
                    false_preds_pos.remove(pred)
    if return_matched_pairs:
        return true_preds, false_preds, matched_gt, unmatched_gt, matched_pairs, false_preds_redun, false_preds_cat, false_preds_pos
    else:
        return true_preds, false_preds, matched_gt, unmatched_gt, false_preds_redun, false_preds_cat, false_preds_pos


def re_pr_f1(pred_json_path, ann_path, score_th=0.2, iou_th=0.25):
    coco_gt = COCO(ann_path)
    coco_pred = coco_gt.loadRes(pred_json_path)
    image_ids = coco_gt.getImgIds()
    image_data = coco_gt.loadImgs(image_ids)
    gt_num = 0
    det_num = 0
    get_num = 0
    img_pixels = 0
    for image in image_data:
        img_pixels += image['height'] * image['width']
        gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image['id']))
        image_detections = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=image['id']))
        preds = []
        true_preds = []
        matched_gt = []
        unmatched_gt = []
        for det in image_detections:
            if det['score'] >= score_th:
                preds.append(det)
        det_num += len(preds)
        for gt in gts:
            bbox_gt = np.array(gt['bbox'])
            preds_cat = [item for item in preds if item['category_id'] == gt['category_id']]
            bboxes_pred = np.array([item['bbox'] for item in preds_cat])
            pred = None
            if len(preds_cat) > 0:
                iou = calculate_iou(bbox_gt, bboxes_pred)
                if np.max(iou) >= iou_th:
                    matched_gt.append(gt)
                    max_iou_index = np.argmax(iou)
                    pred = preds_cat[max_iou_index]
                    if 'out_bbox' in gt:
                        out_bbox = gt['out_bbox']
                        pred_bbox = pred['bbox']
                        if out_bbox[0] <= pred_bbox[0] and out_bbox[1] <= pred_bbox[1] and \
                                out_bbox[0] + out_bbox[2] >= pred_bbox[0] + pred_bbox[2] and out_bbox[1] + out_bbox[
                            3] >= pred_bbox[1] + pred_bbox[3]:
                            true_preds.append(pred)
                            preds.remove(pred)
                        else:
                            unmatched_gt.append(gt)
                            pred = None
                    else:
                        true_preds.append(pred)
                        preds.remove(pred)
                else:
                    unmatched_gt.append(gt)
            else:
                unmatched_gt.append(gt)
        false_preds = preds
        gt_num += len(gts)
        get_num += len(matched_gt)
    Recall = get_num / gt_num
    if det_num>0:
        Precision = get_num / det_num
    else:
        Precision = 0
    if Recall + Precision>0:
        F1 = 2 * Recall * Precision / (Recall + Precision)
    else:
        F1 = 0
    FalseAlarmRate = (det_num-get_num)/img_pixels
    return gt_num, det_num, get_num, Recall, Precision, F1, FalseAlarmRate


def write_comparedmethod_apar(file_path, cat_list, txt_path, txt_list):
    with open(file_path, "w") as newfile:
        info = [0 for i in range(11+len(cat_list)*2)]
        info[0] = 'method'
        info[1] = 'ap'
        info[2] = "ap10"
        info[3] = "ap25"
        info[4] = "ap50"
        info[5] = "ap75"
        info[6] = "ar"
        info[7] = "ar10"
        info[8] = "re25"
        info[9] = "re50"
        info[10] = "re75"
        for i in range(len(cat_list)):
            info[11 + i*2] = "ap"+cat_list[i]
            info[12 + i*2] = "ar"+cat_list[i]
        for i, item in enumerate(info):
            if i < len(info) - 1:
                newfile.write(str(item) + ", ")
            else:
                newfile.write(str(item) + "\n")
        for txt in txt_list:
            info = [0 for i in range(11+len(cat_list)*2)]
            with open(os.path.join(txt_path, txt), 'r') as file:
                info[0] = txt.replace('.txt','')
                for line in file:
                    if line.startswith('overall'):
                        if 'ap5095' in line:
                            ap5095 = float(line.split('ap5095: ')[-1].split(',')[0])
                            ap10 = float(line.split('ap10: ')[-1].split(',')[0])
                            ap25 = float(line.split('ap25: ')[-1].split(',')[0])
                            ap50 = float(line.split('ap50: ')[-1].split(',')[0])
                            ap75 = float(line.split('ap75: ')[-1].split(',')[0])
                            ar5095 = float(line.split('ar5095: ')[-1].split(',')[0])
                            ar10 = float(line.split('ar10: ')[-1].split(',')[0])
                            ar25 = float(line.split('ar25: ')[-1].split(',')[0])
                            ar50 = float(line.split('ar50: ')[-1].split(',')[0])
                            ar75 = float(line.split('ar75: ')[-1].split(',')[0])
                            info[1] = ap5095
                            info[2] = ap10
                            info[3] = ap25
                            info[4] = ap50
                            info[5] = ap75
                            info[6] = ar5095
                            info[7] = ar10
                            info[8] = ar25
                            info[9] = ar50
                            info[10] = ar75
                    for i, cat in enumerate(cat_list):
                        if cat in line:
                            ap5095 = float(line.split('ap5095: ')[-1].split(',')[0])
                            ar5095 = float(line.split('ar5095: ')[-1].split(',')[0])
                            info[11+ i*2] = ap5095
                            info[12+ i*2] = ar5095
            for i, item in enumerate(info):
                # Add a comma after each item except the last one
                if i == 0:
                    newfile.write(str(item) + ", ")
                elif i < len(info) - 1:
                    newfile.write('%.6f'%(item) + ", ")
                else:
                    newfile.write('%.6f'%(item)  + "\n")

def write_comparedmethod_apar_mini(file_path, cat_list, txt_path, txt_list):
    with open(file_path, "w") as newfile:
        info = [0 for i in range(3+len(cat_list)*2)]
        info[0] = 'method'
        info[1] = 'ap'
        info[2] = "re"
        for i in range(len(cat_list)):
            info[3 + i*2] = "ap"+cat_list[i]
            info[4 + i*2] = "re"+cat_list[i]
        for i, item in enumerate(info):
            if i < len(info) - 1:
                newfile.write(str(item) + ", ")
            else:
                newfile.write(str(item) + "\n")
        for txt in txt_list:
            info = [0 for i in range(3+len(cat_list)*2)]
            with open(os.path.join(txt_path, txt), 'r') as file:
                info[0] = txt.replace('.txt','')
                for line in file:
                    if line.startswith('overall'):
                            ap = float(line.split('ap: ')[-1].split(',')[0])
                            re = float(line.split('re: ')[-1].split(',')[0])
                            info[1] = ap
                            info[2] = re
                    for i, cat in enumerate(cat_list):
                        if cat in line:
                            ap = float(line.split('ap: ')[-1].split(',')[0])
                            re = float(line.split('re: ')[-1].split(',')[0])
                            info[3 + i*2] = ap
                            info[4 + i*2] = re
            for i, item in enumerate(info):
                # Add a comma after each item except the last one
                if i == 0:
                    newfile.write(str(item) + ", ")
                elif i < len(info) - 1:
                    newfile.write('%.6f'%(item) + ", ")
                else:
                    newfile.write('%.6f'%(item)  + "\n")
