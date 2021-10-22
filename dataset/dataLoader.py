"""

 Author : T.Dhar

 Date : 22.10.2021

 """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import skimage.io as io


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import albumentations as alb

data_dir = r'Data directory path'
metadata_df = pd.read_csv(os.path.join(data_dir,'metadata.csv'))
metadata_df = metadata_df[metadata_df['split']=='train']
metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
metadata_df['mask_path'] = metadata_df['mask_path'].apply(lambda img_pth : os.path.join(data_dir,img_pth))
metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(lambda img_pth : os.path.join(data_dir,img_pth))


metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)


#train and validation split.....9:1

valid_df = metadata_df.sample(frac=0.1, random_state=42)
train_df = metadata_df.drop(valid_df.index)
len(train_df), len(valid_df)



class_dict = pd.read_csv(os.path.join(data_dir,'class_dict.csv'))
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()

print('All dataset classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)



select_classes = ['urban_land', 'agriculture_land', 'rangeland', 'forest_land', 'water', 'barren_land', 'unknown']

# Get RGB values of required classes
select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

print('Selected classes and their corresponding RGB values in labels:')
print('Class Names: ', class_names)
print('Class RGB values: ', class_rgb_values)




def plot_image(**images):
    n_images = len(images)
    plt.figure(figsize=(17,8))
    for idx,(name, image) in enumerate(images.items()):
        plt.subplot(1,n_images,idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
    
def one_hot_encoder(label,label_values):
    semantic_map = []
    for color in label_values:
        equality = np.equal(label,colour)
        class_map = np.all(equality,axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map,axis = -1)
    return semantic_map



def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x
        



class DeepGlobeSegDat(Dataset):
    def __init__(self,df,class_rgb_values,augmentation = None,transforms = None):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.transforms = transforms
        
        
        
    def __len__(self):
        return len(self.image_paths)
    
    
        
    def __getitem__(self,idx):
        image = io.imread(self.image_paths[idx])
        mask = io.imread(self.mask_paths[idx])
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    



train_data = DeepGlobeSegDat(train_df, class_rgb_values = select_class_rgb_values)
random_idx = random.randint(0, len(train_data)-1)
image, mask = train_data[73]

#plot_image(original_image = image,ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),one_hot_encoded_mask = reverse_one_hot(mask))


validation_data = DeepGlobeSegDat(valid_df,class_rgb_values = select_class_rgb_values)








