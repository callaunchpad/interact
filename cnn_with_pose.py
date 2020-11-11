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
#from model.cnn_model import HOCNN
from model.cnn_with_pose import HOPOSECNN
from datasets import metadata
from utils.vis_tool import vis_img
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn

import json
import cv2
import numpy as np
from matplotlib import pyplot as plt

#%matplotlib inline

print(os.environ)

# Set random seed for reproducibility
torch.manual_seed(21)
np.random.seed(21)

# Define data paths
TRAIN_IMG_PATH = "datasets/hico/images/train2015/"
TRAIN_POSE_PATH = "datasets/human_pose/train2015/"

# Setup training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('training on {}'.format(device))

# Define arguments
batch_size = 32
epochs = 80
initial_lr = 0.00001

feat_type = 'fc7'
data_aug = False
exp_ver = 'v3_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
save_dir = './checkpoints/hicopose'
log_dir = './log/hicopose'
save_every = 5
print_batch_every = 100
print_epoch_every = 1

# set the cache size [0 means infinite]
max_img_cache_size = 0#40000
max_pose_cache_size = 0#40000

print('Running experiment ' + exp_ver)

# Define dataloaders
data_const = HicoConstants(feat_type=feat_type)

train_dataset = HicoDataset(data_const=data_const, subset='train', data_aug=data_aug)
val_dataset = HicoDataset(data_const=data_const, subset='val', data_aug=False, test=True)
dataset = {'train': train_dataset, 'val': val_dataset}
print('set up dataset variable successfully')

train_dataloader = DataLoader(dataset=dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
val_dataloader = DataLoader(dataset=dataset['val'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
dataloader = {'train': train_dataloader, 'val': val_dataloader}
print('set up dataloader successfully')

# Define model
model = HOPOSECNN().to(device)

# Display parameter information
parameter_num = 0
for param in model.parameters():
    parameter_num += param.numel()
print(f'The number of parameters in this model is {parameter_num / 1e6} million')

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)

# Setup visualization
writer = SummaryWriter(log_dir=log_dir + '/' + exp_ver + '/' + 'epoch_train')
io.mkdir_if_not_exists(os.path.join(save_dir, exp_ver, 'epoch_train'), recursive=True)

# Training loop
with open('datasets/processed/hico/anno_list.json') as f:
    anno_list = json.load(f)
    
img_cache = {} # format {key: [human, object, pairwise]}
img_cache_counter = 0
pose_img_cache = {}
pose_img_cache_counter = 0
    
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
                img = img['hois'][0]
                img_id = int(img['id']) - 1
                labels[i][img_id] = 1
                human_bboxes = img['human_bboxes']
                object_bboxes = img['object_bboxes']

                # Apply masks to images [with caching]
                src_img_path = TRAIN_IMG_PATH + 'HICO_train2015_' + img['id'].rjust(8, '0') + '.jpg'
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

                # Get pose data [with caching]
                pose_img_path = TRAIN_POSE_PATH + 'HICO_train2015_' + img['id'].rjust(8, '0') + '_keypoints.json'
                if pose_img_path in pose_img_cache: # Use cache if available
                    pose_img = pose_img_cache[pose_img_path]
                else:
                    with open(pose_img_path) as f:
                        pose_data = json.load(f)

                    pose_img = np.zeros((src.shape[0], src.shape[1], 1))

                    for person in pose_data['people']:
                        pose_data_2d = person['pose_keypoints_2d']
                        prev_point = None
                        for inner_point in range(0, len(pose_data_2d), 3):
                            x, y, c = pose_data_2d[inner_point], pose_data_2d[inner_point + 1], pose_data_2d[inner_point + 2]
                            if c > 0:
                                pose_img = cv2.circle(pose_img, (int(x), int(y)), 10, (255,0,0), \
                                                      thickness=-1, lineType=cv2.FILLED)
                                if prev_point:
                                    pose_img = cv2.line(pose_img, (int(prev_point[0]), int(prev_point[1])), \
                                                        (int(x), int(y)), (255, 0, 0), 3)
                                prev_point = (x, y)

                    # Add to cache if within limits
                    if not max_pose_cache_size or pose_img_cache_counter < max_pose_cache_size:
                        pose_img_cache[pose_img_path] = pose_img
                        pose_img_cache_counter += 1

                '''
                # Visualization of masks and pose NOTE: Not guarenteed to work outside of jupyter notebook
                f, axarr = plt.subplots(1,4)

                human_bbox_rgb = cv2.cvtColor(human_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[0].imshow(human_bbox_rgb)

                object_mask_rgb = cv2.cvtColor(obj_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[1].imshow(object_mask_rgb)

                pairwise_rgb = cv2.cvtColor(pairwise_bbox_img, cv2.COLOR_BGR2RGB)
                axarr[2].imshow(pairwise_rgb)

                zeroed_channel = np.zeros((src.shape[0], src.shape[1], 1))
                stacked_pose_img = np.dstack((pose_img, zeroed_channel, zeroed_channel))
                axarr[3].imshow(stacked_pose_img)

                plt.show()
                f.clf()
                '''

                human_bbox_img = cv2.resize(human_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                obj_bbox_img = cv2.resize(obj_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                pairwise_bbox_img = cv2.resize(pairwise_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
                pose_img = cv2.resize(pose_img, (64, 64), interpolation=cv2.INTER_AREA)

                human_bbox_img = torch.from_numpy(human_bbox_img).to(device)
                obj_bbox_img = torch.from_numpy(obj_bbox_img).to(device)
                pairwise_bbox_img = torch.from_numpy(pairwise_bbox_img).to(device)
                pose_img = torch.from_numpy(pose_img).to(device)

                if i == 0:
                    res_human_input = human_bbox_img.unsqueeze(0)
                    res_obj_input = obj_bbox_img.unsqueeze(0)
                    res_pairwise_input = pairwise_bbox_img.unsqueeze(0)
                    res_pose_input = pose_img.unsqueeze(0)
                else:
                    res_human_input = torch.cat((res_human_input, human_bbox_img.unsqueeze(0)), dim=0)
                    res_obj_input = torch.cat((res_obj_input, obj_bbox_img.unsqueeze(0)), dim=0)
                    res_pairwise_input = torch.cat((res_pairwise_input, pairwise_bbox_img.unsqueeze(0)), dim=0)
                    res_pose_input = torch.cat((res_pose_input, pose_img.unsqueeze(0)), dim=0)

            res_human_input = res_human_input.permute([0,3,1,2]).float().to(device)
            res_obj_input = res_obj_input.permute([0,3,1,2]).float().to(device)
            res_pairwise_input = res_pairwise_input.permute([0,3,1,2]).float().to(device)
            res_pose_input = res_pose_input.unsqueeze(3).permute([0,3,1,2]).float().to(device)
            labels = torch.from_numpy(labels).long().to(device)
            
            if phase == 'train':
                # Initial train loop
                model.train()
                model.zero_grad()
                
                # Forward pass: human, objects, pairwise streams
                outputs = model.forward(res_human_input, res_obj_input, res_pairwise_input, res_pose_input)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(outputs, dim=1)
                ground_labels = torch.max(labels, 1)[1]
                for accuracy_iterator in range(len(ground_labels)):
                    if preds[accuracy_iterator] == ground_labels[accuracy_iterator]:
                        batch_correct += 1
                
            else:
                # Evaluation after train loop
                model.eval()
                with torch.no_grad(): # Disable gradients for validation
                    outputs = model.forward(res_human_input, res_obj_input, res_pairwise_input, res_pose_input)
                    loss = criterion(outputs, torch.max(labels, 1)[1])
                    
                    preds = torch.argmax(outputs, dim=1)
                    ground_labels = torch.max(labels, 1)[1]
                    for accuracy_iterator in range(len(ground_labels)):
                        if preds[accuracy_iterator] == ground_labels[accuracy_iterator]:
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
    if epoch_loss<0.0405 or epoch % save_every == (save_every - 1) and epoch >= (10-1):
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