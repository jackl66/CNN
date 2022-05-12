import torch as T
import numpy as np
import os
from torchvision import transforms, datasets
from torch import nn
import torch.nn.functional as F
import timm  # PyTorch Image Models
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import time 


class EfficientNet_Model:
    def __init__(self,  model_idx=0):
        self.token = str(round(time.time()))
        if model_idx == 0:
            # resnet18 tf_efficientnet_b0
            self.model_name = 'tf_efficientnet_b0'  # Model name (we are going to import model from timm)
            self.img_size = 224  # Resize all the images to be 224 by 224 fit the model

            # going to be used for loading dataset
            # according to the model we use
            self.train_path = './refine_train'
            self.validate_path = './refine_validation'
            self.test_path = './refine_test'
            self.final=1280

        elif model_idx==1:
            self.model_name = 'tf_efficientnet_b2'  
            self.img_size = 260  
            self.final = 1408
            # going to be used for loading dataset
            self.train_path = './fused_train'
            self.validate_path = './fused_validation'
            self.test_path = './fused_test'

        elif model_idx==2:
            self.model_name = 'tf_efficientnet_b4'   
            self.img_size = 380  # Resize all the images to be 380 by 380
            self.final = 1792
            # going to be used for loading dataset
            self.train_path = './fused_train'
            self.validate_path = './fused_validation'
            self.test_path = './fused_test'

        else:
            #resnet18   densenet201
            self.model_name = 'resnet18'   
            self.img_size = 224   
            # going to be used for loading dataset
            self.train_path = './refine_train'
            self.validate_path = './refine_validation'
            self.test_path = './refine_test'

        # data augumentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            # Randomly Rotate Images by +/- 20 degrees, without changing the img size
            transforms.RandomRotation(degrees=(-10, +10), expand=False),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), shear=0.2),
            transforms.ToTensor(),
            # converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Normalize by 3 means 3 StD's of the image net, 3 channels

        ])

        self.validate_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            # T.RandomRotation(degrees=(-20,+20)), #NO need for validation
            transforms.ToTensor(),
            # converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Normalize by 3 means 3 StD's of the image net, 3 channels

        ])

    # replace the classifier to accommodate the project
    def create_new_top_layer(self):
        model = timm.create_model(self.model_name, pretrained=True)  # load pretrained model

        # freeze other para so that won't be changed by balls images
        for param in model.parameters():
            param.requires_grad = False
        
        # change top layer
        if self.model_name == 'tf_efficientnet_b0':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1280, out_features=64),  # 1792 is the orginal in_features
                nn.ReLU(),  # ReLu to be the activation function
                # nn.Dropout(p=0.2),
                # nn.Linear(in_features=512, out_features=256),
                # nn.ReLU(),
                nn.Linear(in_features=64, out_features=6),
            )
        elif self.model_name == 'tf_efficientnet_b2':
            model.classifier = nn.Sequential(
                nn.Linear(in_features=1408, out_features=512),  # 1792 is the orginal in_features
                nn.ReLU(),  # ReLu to be the activation function
                # nn.Dropout(p=0.2),
                # nn.Linear(in_features=512, out_features=128),
                # nn.ReLU(),
                # nn.Linear(in_features=128, out_features=6),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=6),
            )
        elif self.model_name == 'tf_efficientnet_b4':
            model.classifier = nn.Sequential(
                nn.Linear(in_features=1792, out_features=512),  # 1792 is the orginal in_features
                nn.ReLU(),  # ReLu to be the activation function
                # nn.Dropout(p=0.2),
                # nn.Linear(in_features=512, out_features=128),
                # nn.ReLU(),
                # nn.Linear(in_features=128, out_features=6),
                nn.Linear(in_features=512, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=6),
            )
        else:
            model.fc = nn.Sequential(
                nn.Linear(in_features=512, out_features=256),  # fc for resnet is 512, densenet is 1920
                nn.ReLU(),  # ReLu to be the activation function
               
                nn.Linear(in_features=256, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=6),
            )
        return model

    def save_checkpoint(self, state, file_name='best_model.pt'):
        self.file_path = './models/'+ self.model_name + '/'+ self.token 
        T.save(state, self.file_path)
        print("saving succeed\n")