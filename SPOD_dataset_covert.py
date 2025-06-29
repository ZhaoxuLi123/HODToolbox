
from utils.utils import seed_set,find_and_sort_files
import shutil
import random
import numpy as np
import os
import json
import scipy.io as sio



def copy_npy(in_file, out_file,mode, band = None):
    if band == None:
        shutil.copyfile(in_file,out_file)
    else:
        loadData = np.load(in_file)
        m, n, b = loadData.shape
        if b <= band:
            shutil.copyfile(in_file, out_file)
        else:
            id_list = np.linspace(0, b, band + 1)
            data = np.zeros([m, n, band])
            for i in range(band):
                if mode == 'select':
                    data[:, :, i] = loadData[:, :, round(id_list[i])]
                else:
                    data[:, :, i] = np.mean(loadData[:, :, round(id_list[i]):round(id_list[i+1])], axis=2)
            np.save(out_file, data)


def create_subspectra_dict(output_path, final_output_path, seed, mode, band=None):
    seed_set(seed=seed)
    target_spectra = sio.loadmat(os.path.join(output_path, 'target.mat'))['data']
    m, n, b = target_spectra.shape
    if b <= band:
        new_target_spectra =target_spectra
    else:
        id_list = np.linspace(0, b, band + 1)
        new_target_spectra = np.zeros([m, n, band])
        for i in range(band):
            if mode == 'select':
                new_target_spectra[:, :, i] = target_spectra[:, :, round(id_list[i])]
            else:
                new_target_spectra[:, :, i] = np.mean(target_spectra[:, :, round(id_list[i]):round(id_list[i + 1])], axis=2)
    sio.savemat(os.path.join(final_output_path, 'target.mat'), {'data': new_target_spectra})


def split_train_test(output_path, final_output_path, seed, mode, band=None):
    seed_set(seed=seed)
    json_files = ['all.json',]
    image_folder = os.path.join(output_path, 'data')
    train_folder = os.path.join(final_output_path, 'train')
    test_folder = os.path.join(final_output_path, 'test')
    ann_folder = os.path.join(final_output_path, 'annotations')
    train_num = 100
    test_num = 500
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(ann_folder, exist_ok=True)
    merged_annotations = []
    merged_images = []
    for json_file in json_files:
        with open(os.path.join(output_path,'annotations',json_file), 'r') as f:
            data = json.load(f)
            annotations = data['annotations']
            images = data['images']
            merged_annotations.extend(annotations)
            merged_images.extend(images)
    train_json = data.copy()
    test_json = data.copy()
    train_split_json = os.path.join(output_path, 'annotations', 'train_split.json')
    test_split_json  = os.path.join(output_path, 'annotations', 'test_split.json')
    if not os.path.exists(train_split_json):
        train_images = random.sample(merged_images[:859], train_num)
        test_images = random.sample(merged_images[859:], test_num)
        train_images = sorted(train_images, key=lambda x: x['file_name'])
        test_images = sorted(test_images, key=lambda x: x['file_name'])
        with open(train_split_json, "w") as f:
            json.dump(train_images, f)
        f.close()
        with open(test_split_json, "w") as f:
            json.dump(test_images, f)
        f.close()
    else:
        with open(train_split_json, 'r') as f:
            train_images = json.load(f)
        with open(test_split_json, 'r') as f:
            test_images = json.load(f)
    train_img_id_list = [img_dict['id'] for img_dict in train_images]
    train_annotations = []
    for ann_i in merged_annotations:
        if ann_i['image_id'] in train_img_id_list:
            train_annotations.append(ann_i)
    train_json['annotations'] = train_annotations
    train_json['images'] = train_images
    test_img_id_list = [img_dict['id'] for img_dict in test_images]
    test_annotations = []
    for ann_i in annotations:
        if ann_i['image_id'] in test_img_id_list:
            test_annotations.append(ann_i)
    test_json['annotations'] = test_annotations
    test_json['images'] = test_images
    train_ann_file = os.path.join(ann_folder, 'train.json')
    with open(train_ann_file, "w") as f:
        json.dump(train_json, f)
    f.close()
    test_ann_file = os.path.join(ann_folder, 'test.json')
    with open(test_ann_file, "w") as f:
        json.dump(test_json, f)
    f.close()
    for i in range(len(train_json['images'])):
        copy_npy(os.path.join(image_folder, train_json['images'][i]['file_name']),
                 os.path.join(train_folder, train_json['images'][i]['file_name']),mode, band)
    for i in range(len(test_json['images'])):
        copy_npy(os.path.join(image_folder, test_json['images'][i]['file_name']),
                 os.path.join(test_folder, test_json['images'][i]['file_name']), mode, band)
    print("Dataset split successfully!")


def create_testset_mask_gt(output_path, final_output_path):
    class_num = 8
    test_images = find_and_sort_files(os.path.join(final_output_path,'test'),['npy'])
    os.makedirs(os.path.join(final_output_path,'mask_gt'),exist_ok=True)
    for test_img in test_images:
        test_mask = test_img.replace('.npy', 'mask.mat')
        cat_mask = sio.loadmat(os.path.join(output_path, 'mask', test_mask))['cat_mask']
        m,n = cat_mask.shape
        htd_gt = np.zeros([class_num,m,n])
        for cat_i in range(class_num):
            htd_gt[cat_i,:, :][cat_mask == (cat_i + 1)] = 1
        htd_gt = htd_gt.astype(bool)
        sio.savemat(os.path.join(final_output_path, 'mask_gt', test_img.replace('.npy','.mat')), {'gt': htd_gt})


if __name__ == '__main__':

    band_num = 30
    seed = 42
    covert_mode = 'merge'  # 'select' or 'merge'
    dataset_path = './datasets/SPOD_150b_8c/'
    output_path = os.path.join('./datasets', 'SPOD_150b_8c'.replace('150',str(band_num)))
    split_train_test(dataset_path, output_path, seed, covert_mode, band=band_num)
    create_subspectra_dict(dataset_path, output_path, seed, covert_mode, band=band_num)
    create_testset_mask_gt(dataset_path, output_path)









