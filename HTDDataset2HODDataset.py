"""
这个代码以Avon数据集为例，实现了将传统HTD数据集转换成SpecDETR所使用的到COCO格式的object detection 数据集
"""

import argparse
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import spectral
from utils.maskV2 import MaskGenerator
from utils.simulate import add_targets
from utils.utils import aviris_in_order, color_enhance, seed_set, cutimage_and_decide


def create_train_set(target_list, output_path, seed,good_bands, color_bands):
    seed_set(seed=seed)
    max_dict = {'BlueTrap': -1, 'BrownTrap': -1}
    aviris_img_list = ['0920-1631_rad_gc', ]
    img_path = 'utils/avon'
    hdr_path = 'utils/avon.hdr'
    img = spectral.envi.open(hdr_path, img_path)
    band_center = img.bands.centers
    data = aviris_in_order(img[:, :, :], band_center)
    brown_pos = (49, 73)
    blue_pos = (54, 66)
    spec_dict = {}
    spec_dict['BlueTrap'] = data[blue_pos]
    spec_dict['BrownTrap'] = data[brown_pos]
    img_size = 128
    target_num_max = 20
    train_img_num_max = 150
    root_path = output_path
    ann_path = os.path.join(root_path, 'annotations')
    save_data_path = os.path.join(root_path, 'train')
    color_path = os.path.join(root_path, 'color')
    mask_path = os.path.join(root_path, 'mask')
    if not os.path.exists(ann_path):
        os.makedirs(ann_path)
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    if not os.path.exists(color_path):
        os.makedirs(color_path)
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    MGor = MaskGenerator(img_size, img_size, target_num_max, target_list,
        size_dict={1: (11, 16)}, edge_abu_interval=(0.1, 1), center_edge_abu_interval=(0.95, 1), object_cat='in_turn')
    images, categories, annotations = [], [], []
    images_color = []
    for target in target_list:
        categories.append({"supercategory": "", "id": int(target['cat_id']), "name": target['name']})
    for data_id in range(1, len(aviris_img_list)+1):
        ann_id_count = 1
        img_id_count = 1
        if os.path.exists(aviris_img_list[data_id-1] + '.hdr'):
            """
            从以下地址下载0920-1631_rad_gc数据
            http://dirsapps.cis.rit.edu/share2012/SPECTIR_HSI/SPECTIR_HSI_AVON_AM/DATA/RADIANCE/0920-1631.zip
            """
            aviris_img_path = aviris_img_list[data_id-1]
            aviris_hdr_path = aviris_img_list[data_id-1] + '.hdr'
            img_full_now = spectral.envi.open(aviris_hdr_path, aviris_img_path)
            nr, nc, nb = img_full_now.nrows, img_full_now.ncols, img_full_now.nbands
        else:
            # 随机数据演示
            nr, nc, nb = 1000, 1000, 360
            img_full_now = np.random.rand(nr, nc, nb)*8000
        img_dis = 64
        r_group_num = int(nr/img_dis)
        for r_group_id in range(r_group_num):
            if img_id_count > train_img_num_max:
                break
            r_id = r_group_id*img_dis
            c_id = 0
            start_flat = 0
            while (c_id+img_size)<nc:
                img, IsSave = cutimage_and_decide(img_full_now, img_size, r_id, c_id)
                if not IsSave:
                    c_id += img_size
                    continue
                else:
                    if start_flat == 0:
                        start_flat = 1
                    c_id += img_dis
                    img, bbox_list, segment_list, area_list, cat_list, cat_mask, abundance_masks, target_mask = add_targets(
                         img,  good_bands, target_list, max_dict, spec_dict, MGor)
                    img_name = str(img_id_count)
                    img_name = img_name.zfill(6) + '.npy'
                    color_img = color_enhance(img[:, :, color_bands], up_limit = 9000, low_limit = 0)
                    plt.imsave(os.path.join(color_path, img_name.replace('.npy', '.png')), color_img)
                    plt.imsave(os.path.join(mask_path, img_name.replace('.npy', 'abund.png')), abundance_masks[:,:,0])
                    sio.savemat(os.path.join(mask_path, img_name.replace('.npy', 'mask.mat')),
                                {'abundance_masks': abundance_masks, 'cat_mask': cat_mask, 'target_mask': target_mask})
                    np.save(os.path.join(save_data_path, img_name), img)
                    images.append({"file_name": img_name, "height": img_size, "width": img_size, "id": img_id_count})
                    images_color.append(
                        {"file_name": img_name.replace('.npy', '.png'), "height": img_size, "width": img_size,
                         "id": img_id_count})
                    """
                    annotation info:
                    id : anno_id_count
                    category_id : category_id
                    bbox : bbox
                    segmentation : [segment]
                    area : area
                    iscrowd : 0
                    image_id : image_id
                    """
                    for j in range(len(bbox_list)):
                        anno_info = {'id': ann_id_count,
                                     'category_id': cat_list[j],
                                     'bbox': bbox_list[j],
                                     'segmentation': [segment_list[j]],
                                     'area': area_list[j], 'image_id': img_id_count}
                        annotations.append(anno_info)
                        ann_id_count += 1
                    img_id_count += 1
                    print(img_name, 'has created')
                    if img_id_count > train_img_num_max:
                        break
    train_json = {"images": images, "annotations": annotations, "categories": categories}
    train_ann_file = os.path.join(ann_path, 'train.json')
    with open(train_ann_file, "w") as f:
        json.dump(train_json, f)
    f.close()
    # train_json = {"images": images_color, "annotations": annotations, "categories": categories}
    # train_ann_file = os.path.join(ann_path, 'train_color.json')
    # with open(train_ann_file, "w") as f:
    #     json.dump(train_json, f)
    # f.close()


def create_spectra_dict(target_list, output_path, seed, good_bands):
    seed_set(seed=seed)
    img_path = 'utils/avon'
    hdr_path = 'utils/avon.hdr'
    img = spectral.envi.open(hdr_path, img_path)
    band_center = img.bands.centers
    data = aviris_in_order(img[:, :, :], band_center)
    brown_pos = (49, 73)
    blue_pos = (54, 66)
    root_path = output_path
    spec_num = 1
    target_spectra = np.zeros([len(target_list), spec_num, good_bands.size])  # 形状 目标类别数目*目标先验光谱数目*波段数目
    target_spectra[0] = data[blue_pos][good_bands]
    target_spectra[1] = data[brown_pos][good_bands]
    sio.savemat(os.path.join(root_path, 'target.mat'), {'data': target_spectra})


def create_test_set(target_list, output_path, good_bands, color_bands):
    train_json_path = os.path.join(output_path, 'annotations','train.json')
    test_json_path = os.path.join(output_path, 'annotations', 'test.json')
    test_color_json_path = os.path.join(output_path, 'annotations', 'test_color.json')
    color_path = os.path.join(output_path, 'color')
    with open(train_json_path) as f:
        train_json = json.load(f)
    img_size =128
    test_json = train_json.copy()
    test_json['annotations'] = []
    test_json['images'] = []
    test_color_json = train_json.copy()
    test_color_json['annotations'] = []
    test_color_json['images'] = []
    test_list = [str(i) for i in range(1,2)]
    test_data_path = os.path.join(output_path,'test')
    os.makedirs(test_data_path,exist_ok=True)
    for i in range(len(test_list)):
        img_path = 'utils/avon'
        hdr_path = 'utils/avon.hdr'
        img = spectral.envi.open(hdr_path, img_path)
        img = img[:, :, :][:, :, good_bands]
        img_name = str(i+1)  # 数字转化为字符串
        img_name = 'test'+img_name.zfill(6) + '.npy'
        np.save(os.path.join(test_data_path, img_name), img)
        test_json['images'].append({"file_name": img_name, "height": img_size, "width": img_size, "id": int(i+1)})
        test_color_json['images'].append(
            {"file_name": img_name.replace('.npy', '.png'), "height": img_size, "width": img_size,
             "id":int(i+1)})
        color_img = color_enhance(img[:, :, color_bands], up_limit=6000, low_limit=0)
        plt.imsave(os.path.join(color_path, img_name.replace('.npy', '.png')), color_img)
    gt_masks = sio.loadmat('utils/avon_label.mat')
    gt_list = [gt_masks[target['name']] for target in target_list]
    for t_i, gt in enumerate(gt_list):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gt, connectivity=4)
        for i in range(1, num_labels):
            label_mask = np.uint8(labels == i)
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bbox = [x, y, w, h]
                area = w*h
                annotation = {
                    'id': int(len(test_json['annotations']) + 1),
                    'image_id': int(1),
                    'category_id': int(t_i+1),
                    'segmentation': [],
                    'area': area,
                    'bbox': bbox,
                    'iscrowd': 0,
                }
                test_json['annotations'].append(annotation)
                test_color_json['annotations'].append(annotation)
    with open(test_json_path, "w") as f:
        json.dump(test_json, f)
    f.close()
    # with open(test_color_json_path, "w") as f:
    #     json.dump(test_color_json, f)
    # f.close()
    os.makedirs(os.path.join(output_path, 'mask_gt'), exist_ok=True)
    htd_gt = np.array(gt_list)
    htd_gt = htd_gt.astype(np.int8)
    sio.savemat(os.path.join(output_path, 'mask_gt', 'test'+str(1).zfill(6) + '.mat'), {'gt': htd_gt})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create a hsi point point dataset')
    parser.add_argument('--output_path', default='./dataset/example/', help='')
    parser.add_argument('--seed', default=42, help='42')
    args = parser.parse_args()
    target1 = {'sp_num': 1, 'name': 'BlueTrap', 'sp': ['BlueTrap'], 'mix': 'no', 'size': 1, 'cat_id': 1}
    target2 = {'sp_num': 1, 'name': 'BrownTrap', 'sp': ['BrownTrap'], 'mix': 'no', 'size': 1, 'cat_id': 2}
    target_list = [target1, target2, ]
    good_bands = np.r_[0:360]
    color_bands = [52, 33, 15]
    create_train_set(target_list, args.output_path, args.seed, good_bands, color_bands)
    create_spectra_dict(target_list, args.output_path, args.seed, good_bands)
    create_test_set(target_list, args.output_path, good_bands, color_bands)