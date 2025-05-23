import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os
import pandas as pd
from PIL import Image
import random


class ImageDataset_Train(Dataset):
    '''
    Data format in .csv file each line:
    Image Path,Predicted Gender,Predicted Age,Predicted Race,Reliability Score Gender,Reliability Score Age,Reliability Score Race,Ground Truth Gender,Ground Truth Age,Ground Truth Race,Intersection,Target,Specific
    '''

    def __init__(self, csv_file, owntransforms):
        super(ImageDataset_Train, self).__init__()
        self.img_path_label = pd.read_csv(csv_file)
        self.transform = owntransforms 
        
    def __len__(self):
        return len(self.img_path_label)
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img_path = self.img_path_label.iloc[idx, 0]
                # Debug print to verify the content of img_path
        # print(f"Image path at index {idx}: {img_path}")
        # Check if the img_path is indeed a string and exists
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            raise ValueError(f"Expected img_path to be a valid file path, got {type(img_path)}: {img_path}")



        if img_path != 'image_path':
            img = Image.open(img_path)
            img = self.transform(img)
            # label = np.array(self.img_path_label.iloc[idx, 1])
            label = np.array(self.img_path_label.loc[idx, 'label'])

            # intersec_label = np.array(self.img_path_label.iloc[idx, 6])
            intersec_label = np.array(self.img_path_label.loc[idx, 'ismale'])

        return {'image': img, 'label': label, 'intersec_label': intersec_label}
    


class ImageDataset_Test_Dro(Dataset):
    def __init__(self, csv_file, attribute, label_attribute, owntransforms):

        # Get real and fake image lists
       
        # self.image_list = pd.read_csv(csv_file)
        self.transform = owntransforms
        self.img = []
        self.label = []
        self.intersec_labels = []

        attribute_to_labels = {
            'male': 1, 'female': -1 #ff++_ori: -1, ff++_clean: 0
        }
        
        label_attribute_to_labels = {
            'pos': 1, 'neg': 0 
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        if label_attribute not in label_attribute_to_labels:
            raise ValueError(f"Attribute {label_attribute} is not recognized.")
        
        intersec_label= attribute_to_labels[attribute]
        label = label_attribute_to_labels[label_attribute]

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            img_path = row['image_path']
            # mylabel = int(row['label'])
            
            # Depending on the attribute, check the corresponding label
            if intersec_label is not None and int(row['ismale']) == intersec_label: #PROXY_0.10_ismale
                if label is not None and int(row['label']) == label:
                    self.img.append(img_path)
                    self.label.append(label)
                    self.intersec_labels.append(intersec_label)

    def __getitem__(self, index):
        path = self.img[index]
        # img = Image.open(path)  
        img = np.array(Image.open(path))
        label = self.label[index]
        intersec_labels = self.intersec_labels[index]
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label,
            'intersec_label': intersec_labels
        }

        return data_dict

    def __len__(self):
        return len(self.img)

   

class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, attribute, owntransforms):
        self.transform = owntransforms
        self.img = []
        self.label = []
        
        # Mapping from attribute strings to (intersec_label, age_label) tuples
        # Note: if an attribute doesn't correspond to an age label, we use None
        attribute_to_labels = {
            'male,asian': (0, None), 'male,white': (1, None), 'male,black': (2, None),
            'male,others': (3, None), 'nonmale,asian': (4, None), 'nonmale,white': (5, None),
            'nonmale,black': (6, None), 'nonmale,others': (7, None), 'young': (None, 0),
            'middle': (None, 1), 'senior': (None, 2), 'ageothers': (None, 3)
        }

        # Check if the attribute is valid
        if attribute not in attribute_to_labels:
            raise ValueError(f"Attribute {attribute} is not recognized.")
        
        intersec_label, age_label = attribute_to_labels[attribute]

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            next(rows)  # Skip the header row
            for row in rows:
                img_path = row[0]
                mylabel = int(row[11])
                
                # Depending on the attribute, check the corresponding label
                if intersec_label is not None and int(row[10]) == intersec_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                elif age_label is not None and int(row[8]) == age_label:
                    self.img.append(img_path)
                    self.label.append(mylabel)

    def __getitem__(self, index):
        path = self.img[index] 
        img = np.array(Image.open(path))
        label = self.label[index]
        augmented = self.transform(image=img)
        img = augmented['image']  # This is now a PyTorch tensor


        data_dict = {
            'image': img,
            'label': label
        }

        return data_dict


    def __len__(self):
        return len(self.img)



