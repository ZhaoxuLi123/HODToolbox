import os

from utils.utils import *
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def show_prediction(coco_gt, coco_pred, image_ids, color_path, score_th=0.2, iou_th=0.25, method_name='',show_socres=False,
                    FP_cls=True, save_path=None, save_format='pdf'):
    image_data = coco_gt.loadImgs(image_ids)
    for image in image_data:
        image_path = os.path.join(color_path, image['file_name']).replace('npy','png')
        image_annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=image['id']))
        image_detections = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=image['id']))
        true_preds, false_preds, matched_gt, unmatched_gt, false_preds_redun, false_preds_cat, false_preds_pos = judging_prediction(
            image_annotations, image_detections, score_th=score_th, iou_th=iou_th, false_preds_cls=True, return_matched_pairs=False, match_relu='ScoreFirst')
        plt.figure()
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        if FP_cls:
            for annotation in false_preds_redun:
                x, y, w, h = annotation['bbox']
                edgecolor = [color_i / 255 for color_i in [0, 255, 0]]
                ellipse = Ellipse((x - 0.5 + w / 2, y - 0.5 + h / 2), width=max(w, 4), height=max(h, 4), angle=0,
                                  edgecolor=edgecolor,
                                  linewidth=1.5, fc='None')
                plt.gca().add_patch(ellipse)
            for annotation in false_preds_cat:
                x, y, w, h = annotation['bbox']
                edgecolor = [color_i / 255 for color_i in [255, 0, 0]]
                ellipse = Ellipse((x - 0.5 + w / 2, y - 0.5 + h / 2), width=max(w, 4), height=max(h, 4), angle=0,
                                  edgecolor=edgecolor,
                                  linewidth=1.5, fc='None')
                plt.gca().add_patch(ellipse)
            for annotation in false_preds_pos:
                x, y, w, h = annotation['bbox']
                edgecolor = [color_i / 255 for color_i in [255, 0, 0]]
                rect = plt.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor=edgecolor, linewidth=1.5)
                plt.gca().add_patch(rect)
        else:
            for annotation in false_preds:
                x, y, w, h = annotation['bbox']
                edgecolor = [color_i / 255 for color_i in [255, 0, 0]] # 红色
                # rect = plt.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor=edgecolor, linewidth=1.5)
                # plt.gca().add_patch(rect)
                ellipse = Ellipse((x - 0.5 + w / 2, y - 0.5 + h / 2), width=max(w,4), height=max(h,4), angle=0,
                                  edgecolor=edgecolor,
                                  linewidth=1.5, fc='None')
                plt.gca().add_patch(ellipse)
        for annotation in true_preds:
            x, y, w, h  = annotation['bbox']
            category_id = annotation['category_id']
            edgecolor = [color_i / 255 for color_i in [0, 255, 0]]
            rect = plt.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor=edgecolor, linewidth=1.5)
            plt.gca().add_patch(rect)
            if show_socres:
                plt.text(x +w/2-7, y + h + 5,  "{:.3f}".format(annotation['score']), fontsize=12, color=edgecolor, fontweight='bold')
        for annotation in unmatched_gt:
            x, y, w, h = annotation['bbox']
            category_id = annotation['category_id']
            # ellipse = Ellipse((x - 0.5 + w / 2, y - 0.5 + h / 2), width=w, height=h, angle=0,
            #                   edgecolor='orange',
            #                   linewidth=1.5, fc='None')
            # plt.gca().add_patch(ellipse)
            rect = plt.Rectangle((x - 0.5, y - 0.5), w, h, fill=False, edgecolor='orange', linewidth=1.5)
            plt.gca().add_patch(rect)
        if save_path is not None:
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            if save_format == 'png':
                plt.savefig(os.path.join(save_path,image['file_name'].replace('.npy','')+'_'+method_name+'.png'),
                            bbox_inches='tight', pad_inches=0.0, dpi=500)
            if save_format == 'eps':
                plt.savefig(os.path.join(save_path,image['file_name'].replace('.npy','')+'_'+method_name+'.eps'),
                            bbox_inches='tight', pad_inches=0.0)
            if save_format == 'pdf':
                plt.savefig(os.path.join(save_path, image['file_name'].replace('.npy','')+'_'+method_name+'.pdf'),
                            bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    dataset = 'Avon'   # Avon MUUFLGulfport SanDiego SPOD
    eval_methods = 'HOD'   # HOD HTD
    methods = []  # 'specdetr'
    image_ids = []  # [1]
    score_th = 0.1
    iou_th = 0.25
    if dataset == 'MUUFLGulfport':
        iou_th = 0.01
    if dataset == 'SPOD':
        data_path = os.path.join('./datasets/', 'SPOD_30b_8c')
    else:
        data_path = os.path.join('./datasets/', dataset)
    gt_json_path = os.path.join(data_path,'annotations','test.json')
    result_json_path = os.path.join('./eval_result/', dataset + 'Result/', eval_methods, 'cocojson')
    img_save_path = os.path.join('./eval_result/', dataset + 'Result/', 'img')
    os.makedirs(img_save_path, exist_ok=True)
    if len(methods) == 0:
        methods = [f.replace('.json', '') for f in os.listdir(result_json_path) if f.endswith('.json')]
    for method in methods:
        pred_json_path = os.path.join(result_json_path, method+'.json')
        coco_gt = COCO(gt_json_path)
        try:
            coco_pred = coco_gt.loadRes(pred_json_path)
            categories = coco_gt.loadCats(coco_gt.getCatIds())
            category_names = [category['name'] for category in categories]
            if len(image_ids) == 0:
                image_ids = coco_gt.getImgIds()
            color_path = os.path.join(data_path, 'color')
            show_prediction(coco_gt, coco_pred, image_ids, color_path, score_th=score_th, iou_th=iou_th, show_socres=False,
                            FP_cls=True, save_path=img_save_path, method_name=method)
        except:
            pass