import os
import numpy as np
import h5py
import cv2
import json
from PIL import Image

import random
import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import spatial
from model.bgnet_model import AGRNN

img = cv2.imread("HICO_train2015_00000019.jpg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, \
                                                                 box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3)
faster_rcnn.cuda()
faster_rcnn.eval()

node_num = []
features = None
spatial_feat = None
word2vec_emb = None
roi_labels = None
bg = None

checkpoint = torch.load("checkpoints/run_bg_final_final/v8/epoch_train/checkpoint_21_epoch.pth", map_location=device)
model = AGRNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

word2vec = h5py.File("datasets/processed/hico/hico_word2vec.hdf5", 'r')
# coco_dict = ['__background__',
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic_light', 'fire_hydrant', 'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse',
#     'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat', 'baseball_glove',
#     'skateboard', 'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon',
#     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut',
#     'cake', 'chair', 'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop', 'mouse',
#     'remote', 'keyboard', 'cell_phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
#     'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush']

coco_dict = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant', 'N/A', 'stop_sign',
    'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball_bat', 'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot_dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted_plant', 'bed', 'N/A', 'dining_table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
]

verbs = ["adjust", "assemble", "block", "blow", "board", "break", "brush_with", "buy", "carry", "catch", "chase", 
         "check", "clean", "control", "cook", "cut", "cut_with", "direct", "drag", "dribble", "drink_with", "drive",
         "dry", "eat", "eat_at", "exit", "feed", "fill", "flip", "flush", "fly", "greet", "grind", "groom", "herd",
         "hit", "hold", "hop_on", "hose", "hug", "hunt", "inspect", "install", "jump", "kick", "kiss", "lasso", 
         "launch", "lick", "lie_on", "lift", "light", "load", "lose", "make", "milk", "move", "no_interaction",
         "open", "operate", "pack", "paint", "park", "pay", "peel", "pet", "pick", "pick_up", "point", "pour",
         "pull", "push", "race", "read", "release", "repair", "ride", "row", "run", "sail", "scratch", "serve",
         "set", "shear", "sign", "sip", "sit_at", "sit_on", "slide", "smell", "spin", "squeeze", "stab", "stand_on",
         "stand_under", "stick", "stir", "stop_at", "straddle", "swing", "tag", "talk_on", "teach", "text_on", "throw",
         "tie", "toast", "train", "turn", "type_on", "walk", "wash", "watch", "wave", "wear", "wield", "zip"]
         
outputs = []

def hook(module, input, output):
    outputs.clear()
    outputs.append(output)

def hook2(module, input, output):
    outputs.append(output)
 
def hook3(module, input, output):
    outputs.append(output)
 
faster_rcnn.roi_heads.box_head.fc7.register_forward_hook(hook)
faster_rcnn.roi_heads.box_predictor.cls_score.register_forward_hook(hook2)
faster_rcnn.roi_heads.box_predictor.bbox_pred.register_forward_hook(hook3)

img_norm = img / 255
frcnn_img_tensor = torch.from_numpy(img_norm)
frcnn_img_tensor = frcnn_img_tensor.permute([2,0,1]).float().to(device) # chw format
rcnn_input = [frcnn_img_tensor]

out = faster_rcnn(rcnn_input)[0]
feat_embeddings = outputs[0]
cls_prob = F.softmax(outputs[1])
cls_embeddings = {}
 
for i in range(len(cls_prob)):
    cls = int(torch.argmax(cls_prob[i]))
    if cls != 0:
        if cls in cls_embeddings:
            cls_embeddings[cls].append(feat_embeddings[i])
        else:
            cls_embeddings[cls] = [feat_embeddings[i]]

embeddings = []
bboxes = None

for i in range(len(out['boxes'])):
    if out['scores'][i] < .7:
        bboxes = out['boxes'][:i].detach().cpu()
        roi_labels = [out['labels'][:i].detach().cpu().numpy()]
        break
    emb = random.choice(cls_embeddings[out['labels'][i].item()])
    embeddings.append(list(emb.cpu().detach().numpy()))
    
features = torch.Tensor(embeddings).to(device)

img_wh = [img.shape[1], img.shape[0]]
spatial_feat = spatial.calculate_spatial_feats(bboxes, img_wh)
spatial_feat = torch.Tensor(spatial_feat).to(device)

word2vec_emb = np.empty((0,300))
for id in roi_labels[0]:
    vec = word2vec[coco_dict[id]]
    word2vec_emb = np.vstack((word2vec_emb, vec))
word2vec_emb = torch.Tensor(word2vec_emb).to(device)

mask = np.ones_like(img) * 255
for bbox in bboxes:
    bbox = bbox.detach().cpu().numpy()
    cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), thickness=-1)
background_img = cv2.bitwise_and(img, mask, mask=None)
background_img_resize = cv2.resize(background_img, (64, 64), interpolation=cv2.INTER_AREA)
background_img_tensor = torch.from_numpy(background_img)
res_background_input = background_img_tensor.unsqueeze(0)
res_background_input = res_background_input.permute([0,3,1,2]).float().to(device)

with torch.no_grad():
    node_num = [len(roi_labels[0])]
    model_preds, edges = model(node_num, features, spatial_feat, word2vec_emb, roi_labels, bg=res_background_input, validation=True)
    
model_preds = model_preds.detach().cpu()
model_preds = torch.sigmoid(model_preds)
preds = torch.Tensor()
confs = torch.Tensor()

for edge in model_preds:
    conf, pred = torch.topk(edge, 5)
    preds = torch.cat((preds, pred.float()))
    confs = torch.cat((confs, conf.float()))

return_predictions = {}
_, top5_ids = torch.topk(confs, 5)

for id in top5_ids:
    print(verbs[int(preds[id.item()].item())])

for id in out['labels']:
    if id.item() != 1:
        print(coco_dict[id.item()])