from __future__ import print_function
import sys
import os
import ipdb
import json
import cv2
import pickle
import h5py
import argparse
import numpy as np
from tqdm import tqdm

import dgl
import networkx as nx
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from model.bgnet_model import AGRNN
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from datasets import metadata
import utils.io as io

TRAIN_PATH = "datasets/hico/images/test2015/"

def main(args):
    # use GPU if available else revert to CPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print("Testing on", device)

    # Load checkpoint and set up model
    try:
        # load checkpoint
        checkpoint = torch.load(args.pretrained, map_location=device)
        print('Checkpoint loaded!')

        # set up model and initialize it with uploaded checkpoint
        # ipdb.set_trace()
        if not args.exp_ver:
            args.exp_ver = args.pretrained.split("/")[-3]+"_"+args.pretrained.split("/")[-1].split("_")[-2]
        data_const = HicoConstants(feat_type=checkpoint['feat_type'], exp_ver=args.exp_ver)
        model = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)
    
    print('Creating hdf5 file for predicting hoi dets ...')
    if not os.path.exists(data_const.result_dir):
        # print(data_const.result_dir)
        os.makedirs(data_const.result_dir)
    pred_hoi_dets_hdf5 = os.path.join(data_const.result_dir, 'pred_hoi_dets.hdf5')
    pred_hois = h5py.File(pred_hoi_dets_hdf5,'w')

    test_dataset = HicoDataset(data_const=data_const, subset='test', test=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    with open('datasets/processed/hico/anno_list.json') as f:
        anno_list = json.load(f)

    # for global_id in tqdm(test_list): 
    for data in tqdm(test_dataloader):
        train_data = data
        global_id = train_data['global_id'][0]
        img_name = train_data['img_name'][0]
        det_boxes = train_data['det_boxes'][0]
        roi_scores = train_data['roi_scores'][0]
        roi_labels = train_data['roi_labels'][0]
        node_num = train_data['node_num']
        features = train_data['features'] 
        spatial_feat = train_data['spatial_feat']
        word2vec = train_data['word2vec']

        parsed_img_name = img_name.split(".")[0]
        img = [x for x in anno_list if x['global_id'] == parsed_img_name][0]
        img = img['hois'][0]
        img_id = int(img['id']) - 1
        human_bboxes = img['human_bboxes']
        object_bboxes = img['object_bboxes']

        # apply masks to images
        src = cv2.imread(TRAIN_PATH + img_name)
        print(TRAIN_PATH + img_name)
        print(src)
        mask = np.ones_like(src) * 255

        for bbox in human_bboxes:
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), thickness=-1)

        for bbox in object_bboxes:
            cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), thickness=-1)

        background_img = cv2.bitwise_and(src, mask, mask=None)
        background_img = cv2.resize(background_img, (64, 64), interpolation=cv2.INTER_AREA)
        background_img = torch.from_numpy(background_img)

        res_background_input = background_img.unsqueeze(0)
        res_background_input = res_background_input.permute([0,3,1,2]).float().to(device)  

        # referencing
        features, spatial_feat, word2vec = features.to(device), spatial_feat.to(device), word2vec.to(device)
        outputs, attn, attn_lang = model(node_num, features, spatial_feat, word2vec, [roi_labels], bg=res_background_input)    # !NOTE: it is important to set [roi_labels] 
        
        action_score = nn.Sigmoid()(outputs)
        action_score = action_score.cpu().detach().numpy()
        attn = attn.cpu().detach().numpy()
        attn_lang = attn_lang.cpu().detach().numpy()
        # save detection result
        pred_hois.create_group(global_id)
        det_data_dict = {}
        h_idxs = np.where(roi_labels == 1)[0]
        labeled_edge_list = np.cumsum(node_num - np.arange(len(h_idxs)) - 1)
        labeled_edge_list[-1] = 0
        for h_idx in h_idxs:
            for i_idx in range(len(roi_labels)):
                if i_idx <= h_idx:
                    continue
                # import ipdb; ipdb.set_trace()
                edge_idx = labeled_edge_list[h_idx-1] + (i_idx-h_idx-1)
                # score = roi_scores[h_idx] * roi_scores[i_idx] * action_score[edge_idx] * (attn[h_idx][i_idx-1]+attn_lang[h_idx][i_idx-1])
                score = roi_scores[h_idx] * roi_scores[i_idx] * action_score[edge_idx]
                try:
                    hoi_ids = metadata.obj_hoi_index[roi_labels[i_idx]]
                except Exception as e:
                    ipdb.set_trace()
                for hoi_idx in range(hoi_ids[0]-1, hoi_ids[1]):
                    hoi_pair_score = np.concatenate((det_boxes[h_idx], det_boxes[i_idx], np.expand_dims(score[metadata.hoi_to_action[hoi_idx]], 0)), axis=0)
                    if str(hoi_idx+1).zfill(3) not in det_data_dict.keys():
                        det_data_dict[str(hoi_idx+1).zfill(3)] = hoi_pair_score[None,:]
                    else:
                        det_data_dict[str(hoi_idx+1).zfill(3)] = np.vstack((det_data_dict[str(hoi_idx+1).zfill(3)], hoi_pair_score[None,:]))
        for k, v in det_data_dict.items():
            pred_hois[global_id].create_dataset(k, data=v)

    pred_hois.close()

def str2bool(arg):
    arg = arg.lower()
    if arg in ['yes', 'true', '1']:
        return True
    elif arg in ['no', 'false', '0']:
        return False
    else:
        # raise argparse.ArgumentTypeError('Boolean value expected!')
        pass

if __name__ == "__main__":
    # set some arguments
    parser = argparse.ArgumentParser(description='Evaluate the model')

    parser.add_argument('--pretrained', '-p', type=str, default='checkpoints/v3_2048/epoch_train/checkpoint_300_epoch.pth', #default='checkpoints/v3_2048/epoch_train/checkpoint_300_epoch.pth',
                        help='Location of the checkpoint file: ./checkpoints/checkpoint_150_epoch.pth')

    parser.add_argument('--gpu', type=str2bool, default='true',
                        help='use GPU or not: true')

    # parser.add_argument('--feat_type', '--f_t', type=str, default='fc7', required=True, choices=['fc7', 'pool'],
    #                     help='if using graph head, here should be pool: default(fc7) ')

    parser.add_argument('--exp_ver', '--e_v', type=str, default=None,
                        help='the version of code, will create subdir in log/ && checkpoints/ ')

    args = parser.parse_args()
    # data_const = HicoConstants(feat_type=args.feat_type, exp_ver=args.exp_ver)
    # inferencing
    main(args)