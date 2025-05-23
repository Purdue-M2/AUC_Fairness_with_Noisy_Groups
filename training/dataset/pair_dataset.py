'''
The code is designed for scenarios such as disentanglement-based methods where it is necessary to ensure an equal number of positive and negative samples.
'''


import random
import numpy as np

import torch
from torchvision import transforms
from typing import List
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import pandas as pd





class pairDataset(Dataset):
    def __init__(self, csv_fake_file, csv_real_file, owntransforms):

        # Get real and fake image lists
       
        self.fake_image_list = pd.read_csv(csv_fake_file)
        self.real_image_list = pd.read_csv(csv_real_file)
        self.transform = owntransforms


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print(idx, '00000000')
        fake_img_path = self.fake_image_list.loc[idx, 'image_path']
        real_idx = random.randint(0, len(self.real_image_list) - 1)
        real_img_path = self.real_image_list.loc[real_idx, 'image_path']

        if fake_img_path != 'image_path':
            fake_img = Image.open(fake_img_path)
            fake_trans = self.transform(fake_img)
            fake_label = np.array(self.fake_image_list.loc[idx, 'label'])
            fake_spe_label = np.array(self.fake_image_list.loc[idx, 'spe_label'])
            fake_intersec_label = np.array(self.fake_image_list.loc[idx, 'intersec_label'])
          
        if real_img_path != 'image_path':
            real_img = Image.open(real_img_path)
            real_trans = self.transform(real_img)
            real_label = np.array(self.real_image_list.loc[real_idx, 'label'])
            real_spe_label = np.array(self.real_image_list.loc[real_idx, 'label'])
            real_intersec_label = np.array(
                self.real_image_list.loc[real_idx, 'intersec_label'])
           

        return {"fake": (fake_trans, fake_label, fake_spe_label, fake_intersec_label),
                "real": (real_trans, real_label, real_spe_label, real_intersec_label)}

    def __len__(self):
        return len(self.fake_image_list)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor
                    

        Returns:
            A tuple containing the image tensor, the label tensor
        """
        # Separate the image, label,  tensors for fake and real data
        fake_images, fake_labels, fake_spe_labels, fake_intersec_labels = zip(
            *[data["fake"] for data in batch])
  
        fake_labels = tuple(x.item() for x in fake_labels)
        fake_spe_labels = tuple(x.item() for x in fake_spe_labels)
        fake_intersec_labels = tuple(x.item() for x in fake_intersec_labels)
   
        real_images, real_labels, real_spe_labels, real_intersec_labels = zip(
            *[data["real"] for data in batch])
        real_labels = tuple(x.item() for x in real_labels)
        real_spe_labels = tuple(x.item() for x in real_spe_labels)
        real_intersec_labels = tuple(x.item() for x in real_intersec_labels)


        # Stack the image, label, tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        fake_spe_labels = torch.LongTensor(fake_spe_labels)
        fake_intersec_labels = torch.LongTensor(fake_intersec_labels)


        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_spe_labels = torch.LongTensor(real_spe_labels)
        real_intersec_labels = torch.LongTensor(real_intersec_labels)


        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        spe_labels = torch.cat([real_spe_labels, fake_spe_labels], dim=0)
        intersec_labels = torch.cat(
            [real_intersec_labels, fake_intersec_labels], dim=0)
    

        data_dict = {
            'image': images,
            'label': labels,
            'label_spe': spe_labels,
            'intersec_label': intersec_labels,
        }
        return data_dict
    
    
class pairDatasetDro(Dataset):
    def __init__(self, csv_fake_male_file, csv_fake_female_file, csv_real_male_file, csv_real_female_file, owntransforms):

        # Get real and fake image lists
       
        self.fake_male_image_list = pd.read_csv(csv_fake_male_file)
        self.fake_female_image_list = pd.read_csv(csv_fake_female_file)
        self.real_male_image_list = pd.read_csv(csv_real_male_file)
        self.real_female_image_list = pd.read_csv(csv_real_female_file)
        self.transform = owntransforms

        # Define the attribute to labels mapping
        self.attribute_to_labels = {
            'male': 1, 'female' : -1
        }

    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        fake_female_path = self.fake_female_image_list.loc[idx, 'image_path']
        if idx <= len(self.fake_male_image_list) - 1:
            fake_male_idx = idx
        else:
            fake_male_idx = random.randint(0, len(self.fake_male_image_list) - 1)
        fake_male_path = self.fake_male_image_list.loc[fake_male_idx, 'image_path']
            
        if idx <= len(self.real_male_image_list) - 1:
            real_male_idx = idx
        else:
            real_male_idx = random.randint(0, len(self.real_male_image_list) - 1)
        real_male_path = self.real_male_image_list.loc[real_male_idx, 'image_path']
            
        if idx <= len(self.real_female_image_list) - 1:
            real_female_idx = idx
        else:
            real_female_idx = random.randint(0, len(self.real_female_image_list) - 1)
        real_female_path = self.real_female_image_list.loc[real_female_idx, 'image_path']

        if fake_female_path != 'image_path':
            fake_female_img = Image.open(fake_female_path)
            fake_female_trans = self.transform(fake_female_img)
            fake_female_label = np.array(self.fake_female_image_list.loc[idx, 'label'])
            fake_female_intersec_labels = np.array(
                self.fake_female_image_list.loc[idx, 'ismale'])
            
        if fake_male_path != 'image_path':
            fake_male_img = Image.open(fake_male_path)
            fake_male_trans = self.transform(fake_male_img)
            fake_male_label = np.array(self.fake_male_image_list.loc[fake_male_idx, 'label'])          
            fake_male_intersec_labels = np.array(self.fake_male_image_list.loc[fake_male_idx, 'ismale'])
          
        if real_male_path != 'image_path':
            real_male_img = Image.open(real_male_path)
            real_male_trans = self.transform(real_male_img)
            real_male_label = np.array(self.real_male_image_list.loc[real_male_idx, 'label'])
            real_male_intersec_labels = np.array(
                self.real_male_image_list.loc[real_male_idx, 'ismale'])
            
            
        if real_female_path != 'image_path':
            real_female_img = Image.open(real_female_path)
            real_female_trans = self.transform(real_female_img)
            real_female_label = np.array(self.real_female_image_list.loc[real_female_idx, 'label'])
            real_female_intersec_labels = np.array(
                self.real_female_image_list.loc[real_female_idx, 'ismale'])   
           
        return {
            "fake_male": (fake_male_trans, fake_male_label, fake_male_intersec_labels),
            "fake_female": (fake_female_trans, fake_female_label, fake_female_intersec_labels),
            "real_female": (real_female_trans, real_female_label, real_female_intersec_labels),
            "real_male": (real_male_trans, real_male_label, real_male_intersec_labels)
        }

    def __len__(self):
        return max(len(self.fake_male_image_list), len(self.fake_female_image_list), len(self.real_female_image_list), len(self.real_male_image_list))


    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of dictionaries containing the image tensor, and the label tensor.

        Returns:
            A dictionary containing the collated tensors.
        """
        # Separate the image, label, specific label, intersection label, and phat tensors for fake and real data
        fake_male_trans, fake_male_label, fake_male_intersec_label = zip(
            *[data["fake_male"] for data in batch])
        fake_female_trans, fake_female_label, fake_female_intersec_label = zip(
            *[data["fake_female"] for data in batch])
        real_female_trans, real_female_label, real_female_intersec_label= zip(
            *[data["real_female"] for data in batch])
        real_male_trans, real_male_label, real_male_intersec_label = zip(
            *[data["real_male"] for data in batch])

        # Convert labels and phats to tensors using tuples
        fake_male_label = torch.LongTensor(tuple(x.item() for x in fake_male_label))
        fake_male_intersec_label = torch.LongTensor(tuple(x.item() for x in fake_male_intersec_label))

        fake_female_label = torch.LongTensor(tuple(x.item() for x in fake_female_label))
        fake_female_intersec_label = torch.LongTensor(tuple(x.item() for x in fake_female_intersec_label))
        
        real_female_label = torch.LongTensor(tuple(x.item() for x in real_female_label))
        real_female_intersec_label = torch.LongTensor(tuple(x.item() for x in real_female_intersec_label))
        
        real_male_label = torch.LongTensor(tuple(x.item() for x in real_male_label))
        real_male_intersec_label = torch.LongTensor(tuple(x.item() for x in real_male_intersec_label))

        # Stack the image tensors for fake and real data
        fake_male_images = torch.stack(fake_male_trans, dim=0)
        fake_female_images = torch.stack(fake_female_trans, dim=0)
        real_female_images = torch.stack(real_female_trans, dim=0)
        real_male_images = torch.stack(real_male_trans, dim=0)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([fake_male_images, fake_female_images, real_female_images, real_male_images], dim=0)
        labels = torch.cat([fake_male_label, fake_female_label, real_female_label, real_male_label], dim=0)
        intersec_labels = torch.cat([fake_male_intersec_label, fake_female_intersec_label, real_female_intersec_label, real_male_intersec_label], dim=0)

        data_dict = {
            'image': images,
            'label': labels,
            'intersec_label': intersec_labels
        }
        return data_dict


