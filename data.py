import os
import torch
import numpy as np
from PIL import Image
from os.path import join
import cv2
import glob
import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class LLIEDataset(torch.utils.data.Dataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = False, isdemo = False, dataset_type = 'LOL-v1'):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list(dataset_type)
        self.transforms = transforms
        self.isdemo = isdemo
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name).convert('RGB')
            )
        LL_image_PIL = Image.open(ll_image_name).convert('RGB')

        LL_image = self.transforms(
            LL_image_PIL
            )
        
        if self.isdemo:
            return ori_image, LL_image, LL_image, ori_image_name.split('/')[-1].split("\\")[-1]
        
        return ori_image, LL_image, LL_image

    def __len__(self):
        return len(self.file_list)
    
    def get_image_pair_list(self, dataset_type):

        if dataset_type == 'LOL-v1':
            image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key), 
                                    os.path.join(self.lowlight_root, key)])
        elif dataset_type == 'LOL-v2' or dataset_type == 'LOL-v2-real' or dataset_type == 'LOL-v2-Syn':
            if self.istrain:
                Real_Low_root = join(self.lowlight_root,'Real_captured', 'Train', "Low")
                Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Train', "Low")
                Real_High_root = join(self.ori_root,'Real_captured', 'Train', "Normal")
                Synthetic_High_root = join(self.ori_root,'Synthetic', 'Train', "Normal")
            else:
                Real_Low_root = join(self.lowlight_root,'Real_captured', 'Test', "Low")
                Synthetic_Low_root = join(self.lowlight_root,'Synthetic', 'Test', "Low")
                Real_High_root = join(self.ori_root,'Real_captured', 'Test', "Normal")
                Synthetic_High_root = join(self.ori_root,'Synthetic', 'Test', "Normal")
            
            # For Real
            if dataset_type == 'LOL-v2-Syn':
                Real_name_list =[]
            else:
                Real_name_list = [join(Real_Low_root, x) for x in os.listdir(Real_Low_root) if is_image_file(x)]
            
            for key in Real_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Real_High_root, 'normal'+key[3:]), 
                                    os.path.join(Real_Low_root, key)])
            
            # For Synthetic

            if dataset_type == 'LOL-v2-real':
                Synthetic_name_list =[]
            else:
                Synthetic_name_list = [join(Synthetic_Low_root, x) for x in os.listdir(Synthetic_Low_root) if is_image_file(x)]
            
            for key in Synthetic_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(Synthetic_High_root, key), 
                                    os.path.join(Synthetic_Low_root, key)])
        
        elif dataset_type == 'RESIDE':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            # if self.istrain:
            if os.path.isfile( os.path.join(self.ori_root, image_name_list[0].split('_')[0]+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key.split('_')[0]+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        elif dataset_type == 'expe':
            image_name_list = [x for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            if os.path.isfile( os.path.join(self.ori_root, '_'.join(image_name_list[0].split('_')[:-1])+'.jpg')):
                FileE = '.jpg'
            else:
                FileE = '.png'
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, '_'.join(key.split('_')[:-1])+FileE), 
                                    os.path.join(self.lowlight_root,key)])   
        elif dataset_type == 'VE-LOL':
            image_name_list = [join(self.lowlight_root, x) for x in os.listdir(self.lowlight_root) if is_image_file(x)]
            for key in image_name_list:
                key = key.split("/")[-1]
                if os.name == 'nt':
                    key = key.split("\\")[-1]
                self.file_list.append([os.path.join(self.ori_root, key.replace('low', 'normal',)), 
                                    os.path.join(self.lowlight_root, key)])
        else:
            raise ValueError(str(dataset_type) + "does not support! Please change your dataset type")
                
        if self.istrain or (dataset_type[:6] == 'LOL-v2'):
            random.shuffle(self.file_list)

    def add_dataset(self, ori_root, lowlight_root, dataset_type = 'LOL-v1',):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.get_image_pair_list(dataset_type)

class LLIE_Dataset(LLIEDataset):
    def __init__(self, ori_root, lowlight_root, transforms, istrain = True):
        self.lowlight_root = lowlight_root
        self.ori_root = ori_root
        self.image_name_list = glob.glob(os.path.join(self.lowlight_root, '*.png'))
        self.matching_dict = {}
        self.file_list = []
        self.istrain = istrain
        self.get_image_pair_list()
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        """
        :param item:
        :return: haze_img, ori_img
        """
        ori_image_name, ll_image_name = self.file_list[item]
        ori_image = self.transforms(
            Image.open(ori_image_name)
            )

        LL_image_PIL = Image.open(ll_image_name)
        LL_image = self.transforms(
            LL_image_PIL
            )
        
        return ori_image, LL_image

    def __len__(self):
        return len(self.file_list)