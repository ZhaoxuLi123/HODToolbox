import numpy as np
import random
import itertools
import os
import spectral
import scipy.io as sio
from .utils import aviris_in_order


def spectra_simulation(target_spec, max_value, gain, good_bands,):
    if max_value == -1:
        target_spec_1 = target_spec
        return target_spec_1
    else:
        """
        以下是SPOD数据集目标光谱仿真流程，考虑局部光谱波动和广域光谱波动
        """
        info_name = 'aviris_good_bands_info.mat'
        # good_bands = np.r_[7:103, 122:145, 146:149, 172:220]
        if not os.path.exists(os.path.join('./utils', info_name)):
            """
            从以下地址下载f180601t01p00r09数据
            https://popo.jpl.nasa.gov/avcl/y18_data/f180601t01p00r09_refl.tar.gz
            """
            img_path = 'f180601t01p00r09rdn_e_sc01_ort_img'
            hdr_path = 'f180601t01p00r09rdn_e_sc01_ort_img.hdr'
            img_full = spectral.envi.open(hdr_path, img_path)
            band_center = img_full.bands.centers
            area_range = [40900, 41780, 180, 640]
            water_area = aviris_in_order(img_full[area_range[0]:area_range[1], :, :][:, area_range[2]:area_range[3], :], band_center)
            water_area = water_area[:, :, good_bands]
            mean_water_spec = np.mean(water_area, axis=(0, 1))
            std_water_spec = np.std(water_area, axis=(0, 1))
            CoeVar = std_water_spec / mean_water_spec
            background_fluctuation = (water_area - mean_water_spec) / mean_water_spec
            normalized_background_fluctuation = background_fluctuation / CoeVar
            nbf_noise = (normalized_background_fluctuation - np.expand_dims(np.mean(normalized_background_fluctuation, axis=(2)), axis=2))
            nbf_noise_std_wise_band = np.std(nbf_noise,axis=(0,1))
            nbf_std = np.std(np.mean(normalized_background_fluctuation, axis=(2)))
            sio.savemat(os.path.join('./utils', info_name), {'mean_water_spec':mean_water_spec,
                'nbf_noise_std_wise_band':nbf_noise_std_wise_band, 'nbf_std':nbf_std,'CoeVar':CoeVar})
        else:
            data = sio.loadmat(os.path.join('./utils', info_name))
            mean_water_spec = data['mean_water_spec']
            nbf_noise_std_wise_band = data['nbf_noise_std_wise_band']
            nbf_std = data['nbf_std']
            CoeVar = data['CoeVar']
        target_spec_1 = max_value*target_spec/np.max(target_spec)
        nbf_mean = 1*nbf_std*np.random.randn()*np.ones(nbf_noise_std_wise_band.size)
        nbf_noise = 1*nbf_noise_std_wise_band * np.random.randn(nbf_noise_std_wise_band.size)
        ratio =(nbf_mean+nbf_noise)*CoeVar
        target_spec_1 = (1+gain)*target_spec_1*(1+ratio)
        return target_spec_1


def add_spectra(img, good_bands, target_list, max_dict, spec_dict, cat_mask, abundance_masks, target_mask):
    h, w, b = img.shape
    target_ids=list(set(list(target_mask.reshape(-1))))
    target_ids.remove(0)
    gain_dict = {}
    for t_id in target_ids:
        gain_dict[t_id] = random.uniform(-0.3,0.3)
    for i, j in itertools.product(range(h), range(w)):
        if cat_mask[i, j]>0:
            add_sps = np.zeros([b, 3])
            sp_name_list = target_list[int(cat_mask[i, j]-1)]['sp']
            for sp_i in range(len(sp_name_list)):
                sp_name = sp_name_list[sp_i]
                sp_ori = spec_dict[sp_name]
                sp_max = max_dict[sp_name]
                gain = gain_dict[target_mask[i, j]]
                sp_add = spectra_simulation(sp_ori[good_bands], sp_max, gain, good_bands)
                add_sps[:, sp_i] = sp_add
            sp_add_all = abundance_masks[i,j,1]*add_sps[:,0] + abundance_masks[i,j,2]*add_sps[:,1] +\
                     abundance_masks[i,j,3]*add_sps[:,2]
            img[i, j, :] = sp_add_all * abundance_masks[i, j, 0] + img[i, j, :] * (1 - abundance_masks[i, j, 0])
    return img


def add_targets(img, good_bands, target_list, max_dict, spec_dict, MGor):
    img = img[:, :, good_bands]
    cat_mask, abundance_masks, target_mask, bbox_list, segment_list, area_list, cat_list = MGor()
    img_new = add_spectra(img, good_bands, target_list, max_dict, spec_dict, cat_mask, abundance_masks, target_mask)
    return img_new, bbox_list, segment_list, area_list, cat_list, cat_mask, abundance_masks, target_mask










