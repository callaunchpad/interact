# Imports
from __future__ import print_function

import os
import time

import dgl
import networkx as nx
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split

import ipdb
import h5py
import pickle
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import datetime

import utils.io as io
from model.cnn_model import HOCNN
from datasets import metadata
from utils.vis_tool import vis_img
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn

from optimizers import adabound

import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%matplotlib inline

# Set random seed for reproducibility
torch.manual_seed(21)
np.random.seed(21)

# Define data paths
TRAIN_IMG_PATH = "datasets/hico/images/train2015/"

# Setup training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on {}'.format(device))

# Define arguments
batch_size = 32
epochs = 300
initial_lr = 0.0005
final_lr = 0.1 # Only for adabound optimizer
l2_weight_decay = 0.00005
weighted_loss_alpha = 1 # Boost the weight of positives to help negative dominance
weighted_loss_beta = 0.9999 # From the class based balance formula

feat_type = 'fc7'
data_aug = False
exp_ver = 'v2_nl2_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
# w = weighted loss, n = normalized (/255), l2 = l2 norm
# 0 = adagrad, 1 = adabound
save_dir = './checkpoints/hico'
log_dir = './log/hico'
save_every = 5
print_batch_every = 100
print_epoch_every = 1

# set the cache size [0 means infinite]
max_img_cache_size = 1 #40000

print('Running experiment ' + exp_ver)

# Define dataloaders
data_const = HicoConstants(feat_type=feat_type)

train_dataset = HicoDataset(data_const=data_const, subset='train', data_aug=data_aug)
val_dataset = HicoDataset(data_const=data_const, subset='val', data_aug=False, test=True)
dataset = {'train': train_dataset, 'val': val_dataset}
print('Set up dataset variable successfully')

train_dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_dataloader = DataLoader(dataset=dataset['val'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
dataloader = {'train': train_dataloader, 'val': val_dataloader}
print('Set up dataloader successfully')

# Define model
model = HOCNN().to(device)

# Display parameter information
parameter_num = 0
for param in model.parameters():
    parameter_num += param.numel()
print(f'The number of parameters in this model is {parameter_num / 1e6} million')

# Define weighted loss due to unbalanced dataset
# Implemented class balanced loss from [https://github.com/richardaecn/class-balanced-loss]
# The average of these weights is 1.0
'''
with open('datasets/processed/hico/hoi_cls_count.json') as f:
    hoi_class_count = json.load(f)
    hoi_classes = hoi_class_count.keys()
    samples_per_class = np.zeros((len(hoi_classes), ))
    for hoi_class in hoi_classes:
        samples_per_class[int(hoi_class) - 1] = hoi_class_count[hoi_class]
    effective_num = 1.0 - np.power(weighted_loss_beta, samples_per_class)
    loss_weights = (1.0 - weighted_loss_beta) / np.array(effective_num)
    loss_weights = weighted_loss_alpha * loss_weights / np.sum(loss_weights) * len(hoi_classes)
    loss_weights = torch.tensor(loss_weights).float().to(device)
'''

# Todo: look at distribution balanced loss from [https://github.com/wutong16/DistributionBalancedLoss]
  
# Weighted loss using difference between class sizes
# Each class pos_weight is num_greatest_class / num_pos
# It seems like the graients explode even using only the num_greatest
'''
with open('datasets/processed/hico/hoi_cls_count.json') as f:
    hoi_class_count = json.load(f)
    hoi_classes = hoi_class_count.keys()
    samples_per_class = np.zeros((len(hoi_classes), ))
    greatest_sample = 0
    for hoi_class in hoi_classes:
        greatest_sample = max(greatest_sample, hoi_class_count[hoi_class])
        samples_per_class[int(hoi_class) - 1] = hoi_class_count[hoi_class]
    loss_weights = greatest_sample / samples_per_class
    loss_weights = torch.tensor(loss_weights).float().to(device)
    
print('Calculated loss weights')
'''

# Define loss function
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCEWithLogitsLoss(reduction='none')
#criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
#criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=l2_weight_decay)
#optimizer = adabound.AdaBound(model.parameters(), lr=initial_lr, final_lr=final_lr, weight_decay=l2_weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=0)

# Setup visualization
writer = SummaryWriter(log_dir=log_dir + '/' + exp_ver + '/' + 'epoch_train')
io.mkdir_if_not_exists(os.path.join(save_dir, exp_ver, 'epoch_train'), recursive=True)

# Training loop
with open('datasets/processed/hico/anno_list.json') as f:
    anno_list = json.load(f)
    
img_cache = {} # format {key: [human, object, pairwise]}
img_cache_counter = 0
    
print('Training has started!')
    
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for phase in ['train', 'val']:
        start_time = time.time()
        running_loss = 0.0
        running_correct = 0
        idx = 0
        
        for data in dataloader[phase]:
            train_data = data
            img_name = train_data['img_name']
            
            labels = np.zeros((batch_size, 600))
            batch_correct = 0
            for i in range(batch_size):
                # Get image data
                parsed_img_name = img_name[i].split(".")[0]
                img = [x for x in anno_list if x['global_id'] == parsed_img_name][0]
                img_data = img['hois'][0]
                pos_hois = list(map(int, img['pos_hoi_ids']))
                for pos_hoi in pos_hois:
                    labels[i][pos_hoi - 1] = 1
                human_bboxes = img_data['human_bboxes']
                object_bboxes = img_data['object_bboxes']

                # Apply masks to images [with caching]
                src_img_path = TRAIN_IMG_PATH + parsed_img_name + '.jpg'
                if src_img_path in img_cache: # Use cache if available
                    human_bbox_img, obj_bbox_img, pairwise_bbox_img = img_cache[src_img_path]
                else:
                    src = cv2.imread(src_img_path)
                    human_mask = np.zeros_like(src)
                    for bbox in human_bboxes:
                        cv2.rectangle(human_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
                    human_bbox_img = cv2.bitwise_and(src, human_mask, mask=None)

                    obj_mask = np.zeros_like(src)
                    pairwise_mask = human_mask
                    for bbox in object_bboxes:
                        cv2.rectangle(obj_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
                        cv2.rectangle(pairwise_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
                    obj_bbox_img = cv2.bitwise_and(src, obj_mask, mask=None)
                    pairwise_bbox_img = cv2.bitwise_and(src, pairwise_mask, mask=None)
                    
                    # Add to cache if within limits
                    if not max_img_cache_size or img_cache_counter < max_img_cache_size:
                        img_cache[src_img_path] = [human_bbox_img, obj_bbox_img, pairwise_bbox_img]
                        img_cache_counter += 1

                '''
                # Visualization of masks NOTE: Not guarenteed to work outside of jupyter notebook
                f, axarr = plt.subplots(1,3)

                human_bbox_rgb = cv2.cvtColor(human_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[0].imshow(human_bbox_rgb)

                object_mask_rgb = cv2.cvtColor(obj_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[1].imshow(object_mask_rgb)

                pairwise_rgb = cv2.cvtColor(pairwise_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[2].imshow(pairwise_rgb)

                plt.show()
                f.clf()
                '''

                human_bbox_img = cv2.resize(human_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                obj_bbox_img = cv2.resize(obj_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                pairwise_bbox_img = cv2.resize(pairwise_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                
                # Normalize images by dividing by 255
                human_bbox_img = human_bbox_img/255
                obj_bbox_img = obj_bbox_img/255
                pairwise_bbox_img = pairwise_bbox_img/255

                human_bbox_img = torch.from_numpy(human_bbox_img).to(device)
                obj_bbox_img = torch.from_numpy(obj_bbox_img).to(device)
                pairwise_bbox_img = torch.from_numpy(pairwise_bbox_img).to(device)

                if i == 0:
                    res_human_input = human_bbox_img.unsqueeze(0)
                    res_obj_input = obj_bbox_img.unsqueeze(0)
                    res_pairwise_input = pairwise_bbox_img.unsqueeze(0)
                else:
                    res_human_input = torch.cat((res_human_input, human_bbox_img.unsqueeze(0)), dim=0)
                    res_obj_input = torch.cat((res_obj_input, obj_bbox_img.unsqueeze(0)), dim=0)
                    res_pairwise_input = torch.cat((res_pairwise_input, pairwise_bbox_img.unsqueeze(0)), dim=0)

            res_human_input = res_human_input.permute([0,3,1,2]).float().to(device)
            res_obj_input = res_obj_input.permute([0,3,1,2]).float().to(device)
            res_pairwise_input = res_pairwise_input.permute([0,3,1,2]).float().to(device)
            labels = torch.from_numpy(labels).float().to(device)
            
            if phase == 'train':
                # Initial train loop
                model.train()
                model.zero_grad()
                
                # Forward pass: human, objects, pairwise streams
                outputs = model.forward(res_human_input, res_obj_input, res_pairwise_input)
                loss = criterion(outputs, labels)
                #loss = (loss * loss_weights).mean()
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(outputs, dim=1)
                for accuracy_iterator in range(len(preds)):
                    ground_labels = torch.nonzero(labels[accuracy_iterator]).squeeze()
                    if preds[accuracy_iterator] in ground_labels:
                        batch_correct += 1
                
            else:
                # Evaluation after train loop
                model.eval()
                with torch.no_grad(): # Disable gradients for validation
                    outputs = model.forward(res_human_input, res_obj_input, res_pairwise_input)
                    loss = criterion(outputs, labels)
                    #loss = (loss * loss_weights).mean()
                    
                    preds = torch.argmax(outputs, dim=1)
                    for accuracy_iterator in range(len(preds)):
                        ground_labels = torch.nonzero(labels[accuracy_iterator]).squeeze()
                        if preds[accuracy_iterator] in ground_labels:
                            batch_correct += 1
                    
            # Accumulate loss of each batch (average * batch size)
            running_loss += loss.item() * batch_size
            running_correct += batch_correct
            
            # Print out status per print_batch_every
            idx += 1
            if (idx % print_batch_every) == 0:
                print("[{}] Epoch: {}/{} Batch: {}/{} Loss: {} Accuracy: {}".format(\
                        phase, epoch+1, epochs, idx, len(dataloader[phase]), \
                        loss.item(), 100 * batch_correct / batch_size))
            
        # Epoch loss and accuracy
        epoch_loss = running_loss / len(dataset[phase])
        epoch_accuracy = 100 * running_correct / len(dataset[phase])
        
        # Log trainval data for visualization
        if phase == 'train':
            train_loss = epoch_loss 
            train_accuracy = epoch_accuracy
        else:
            writer.add_scalars('trainval_loss_epoch', {'train': train_loss, 'val': epoch_loss}, epoch)
            writer.add_scalars('trainval_accuracy_epoch', {'train': train_accuracy, 'val': epoch_accuracy}, epoch)
            
        # Output data per print_epoch_every
        if (epoch % print_epoch_every) == 0:
            end_time = time.time()
            print("[{}] Epoch: {}/{} Loss: {} Accuracy: {} Execution time: {}".format(\
                    phase, epoch+1, epochs, epoch_loss, epoch_accuracy, (end_time-start_time)))
    
    # Save the model per save_every
    if epoch_loss<0.000405 or epoch % save_every == (save_every - 1) and epoch >= (10-1):
        checkpoint = { 
                        'lr': initial_lr,
                       'b_s': batch_size,
                 'feat_type': feat_type,
                'state_dict': model.state_dict()
        }
        save_name = "checkpoint_" + str(epoch+1) + '_epoch.pth'
        torch.save(checkpoint, os.path.join(save_dir, exp_ver, 'epoch_train', save_name))
        
print('Finishing training!')

# Close visualization
writer.close()
