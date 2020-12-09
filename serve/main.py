import sys
sys.path.append("..")

import os
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import cv2
import json
import h5py
import random

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from openpose.body import Body
from model.cnn_model import HOCNN
from model.cnn_with_pose import HOPOSECNN
from model.bgnet_model import AGRNN

from util import spatial

print('[*] Beginning server startup!')

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1' # Change this ID to an unused GPU
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on', device)

'''
Load models and required data
'''
# HOCNN
hocnn_weights_name = 'v2_nl2_2020-11-20_09-57_ep20.pth'
hocnn_weights_path = 'weights/hico/' + hocnn_weights_name

hocnn_weights = torch.load(hocnn_weights_path, map_location=device)['state_dict']

hocnn_model = HOCNN().to(device)
hocnn_model.load_state_dict(hocnn_weights)
hocnn_model.eval()

print('HOCNN model loaded')

# HOPOSECNN
hoposecnn_weights_name = 'v4_nl2_2020-11-24_21-32_ep35.pth'
hoposecnn_weights_path = 'weights/hicopose/' + hoposecnn_weights_name

hoposecnn_weights = torch.load(hoposecnn_weights_path, map_location=device)['state_dict']

hoposecnn_model = HOPOSECNN().to(device)
hoposecnn_model.load_state_dict(hoposecnn_weights)
hoposecnn_model.eval()

print('HOPOSECNN model loaded')

# Cool Background Net
cbgn_weights_name = 'bgnet_weights.pth'
cbgn_weights_path = 'weights/cbgn/' + cbgn_weights_name

cgbn_checkpoint = torch.load(cbgn_weights_path, map_location=device)
cbgn_model = AGRNN(feat_type=cgbn_checkpoint['feat_type'], bias=cgbn_checkpoint['bias'], bn=cgbn_checkpoint['bn'], dropout=cgbn_checkpoint['dropout'], multi_attn=cgbn_checkpoint['multi_head'], layer=cgbn_checkpoint['layers'], diff_edge=cgbn_checkpoint['diff_edge']).to(device)
cbgn_model.load_state_dict(cgbn_checkpoint['state_dict'])
cbgn_model.eval()

print('Cool Background Net model loaded')

# Faster RCNN, OpenPose, and data lists

faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
faster_rcnn.eval()
print('Faster rcnn for cnn models loaded')

faster_rcnn_cbgn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=200, \
    box_batch_size_per_image=128, box_score_thresh=0.1, box_nms_thresh=0.3).to(device)
faster_rcnn_cbgn.eval()

outputs = []

def hook(module, input, output):
    outputs.clear()
    outputs.append(output)

def hook2(module, input, output):
    outputs.append(output)
 
def hook3(module, input, output):
    outputs.append(output)

faster_rcnn_cbgn.roi_heads.box_head.fc7.register_forward_hook(hook)
faster_rcnn_cbgn.roi_heads.box_predictor.cls_score.register_forward_hook(hook2)
faster_rcnn_cbgn.roi_heads.box_predictor.bbox_pred.register_forward_hook(hook3)
print('Faster rcnn and hooks for cbgn loaded')

body_estimation = Body('openpose/body_pose_model.pth')
print('OpenPose loaded')

coco_dict = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
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

with open('../datasets/processed/hico/hoi_list.json') as f:
    hoi_list = json.load(f)

word2vec = h5py.File('word2vec/hico_word2vec.hdf5', 'r')
    
print('COCO dictionary, verbs list, HOI list, and word2vec loaded')

'''
Initialize server and setup CORS
'''

app = FastAPI()

origins = [
    '*'
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
HOCNN prediction route
'''
@app.post("/api/hocnn/predict")
async def HOCNN_predict(image: UploadFile = File(...)):
    try:
        img_contents = await image.read()
        # Convert string data to numpy array
        np_img = np.fromstring(img_contents, np.uint8)
        # Convert numpy array to image
        parsed_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'error': str(e)}

    # Prepare for and pass through Faster RCNN
    frcnn_inp_img = parsed_img / 255 # Faster RCNN needs image normalized on all channels
    frcnn_img_tensor = torch.from_numpy(frcnn_inp_img)
    frcnn_img_tensor = frcnn_img_tensor.permute([2,0,1]).float().to(device) # chw format
    rcnn_input = [frcnn_img_tensor]
    
    frcnn_output = faster_rcnn(rcnn_input)[0]
    
    human_bboxes, object_bboxes = [], []
    frcnn_objs = []
    object_dic = {}
    for i in range(len(frcnn_output['boxes'])):
        if frcnn_output['scores'][i] < .7:
            break
        bbox = list(map(int, frcnn_output['boxes'][i]))
        if frcnn_output['labels'][i] == 1: # human
            human_bboxes.append(bbox)
        else: 
            object_bboxes.append(bbox)
            object_dic[coco_dict[frcnn_output['labels'][i]]] = bbox
            frcnn_objs.append(coco_dict[frcnn_output['labels'][i]])
            
    # Apply masks using Faster RCNN output
    human_mask = np.zeros_like(parsed_img)
    for bbox in human_bboxes:
        cv2.rectangle(human_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
    human_bbox_img = cv2.bitwise_and(parsed_img, human_mask, mask=None)

    obj_mask = np.zeros_like(parsed_img)
    pairwise_mask = human_mask
    for bbox in object_bboxes:
        cv2.rectangle(obj_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
        cv2.rectangle(pairwise_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
    obj_bbox_img = cv2.bitwise_and(parsed_img, obj_mask, mask=None)
    pairwise_bbox_img = cv2.bitwise_and(parsed_img, pairwise_mask, mask=None)
    
    # Resize images and prepare to be input to HOCNN model
    human_bbox_img = cv2.resize(human_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
    obj_bbox_img = cv2.resize(obj_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
    pairwise_bbox_img = cv2.resize(pairwise_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)

    human_bbox_img = human_bbox_img/255
    obj_bbox_img = obj_bbox_img/255
    pairwise_bbox_img = pairwise_bbox_img/255
    
    human_bbox_img = torch.from_numpy(human_bbox_img).to(device)
    obj_bbox_img = torch.from_numpy(obj_bbox_img).to(device)
    pairwise_bbox_img = torch.from_numpy(pairwise_bbox_img).to(device)

    res_human_input = human_bbox_img.unsqueeze(0)
    res_obj_input = obj_bbox_img.unsqueeze(0)
    res_pairwise_input = pairwise_bbox_img.unsqueeze(0)

    res_human_input = res_human_input.permute([0,3,1,2]).float().to(device)
    res_obj_input = res_obj_input.permute([0,3,1,2]).float().to(device)
    res_pairwise_input = res_pairwise_input.permute([0,3,1,2]).float().to(device)
    
    # Pass to HOCNN model
    with torch.no_grad(): # Disable gradients for validation
        outputs = hocnn_model.forward(res_human_input, res_obj_input, res_pairwise_input)
        
    # Prepare and return predictions
    confidences = F.sigmoid(outputs)
    top5preds = torch.topk(confidences, 5)
    top5_idx = top5preds[1].flatten()
    top5_confidence = top5preds[0].flatten()
    
    return_predictions = {}

    modelPreds, modelObjs = [], []
    for i in range(len(top5_idx)):
        prediction = hoi_list[top5_idx[i]]
        modelPreds.append(prediction)
        modelObjs.append(prediction['object'])

        prediction_str = prediction['object'] + ' ' + prediction['verb']
        prediction_conf = round(top5_confidence[i].item() * 100, 2)
        return_predictions[prediction_str] = prediction_conf
    
    # do faster rcnn integration logic
    if len(frcnn_objs) == 0:
        frcnn_pred = None
    else:
        frcnn_pred = frcnn_objs[0]
	
    if frcnn_pred in modelObjs:  # if fasterrcnn obj exists within predictions, likely that prediction is most correct
        ind = modelObjs.index(frcnn_pred)
        frcnn_str = modelPreds[ind]['object'] + ' ' + modelPreds[ind]['verb']
    else:
        if not frcnn_pred:
            frcnn_str = 'no objects detected by faster-rcnn!'
        else:                   # MAYBE TODO: sometimes multiple verbs correct, not always first one idk
            frcnn_str = frcnn_pred + ' ' + modelPreds[0]['verb']
    return_predictions['fasterrcnn_object'] = frcnn_str # idk, change

    return return_predictions

'''
HOPOSECNN prediction route
'''
@app.post("/api/hoposecnn/predict")
async def HOPOSECNN_predict(image: UploadFile = File(...)):
    try:
        img_contents = await image.read()
        # Convert string data to numpy array
        np_img = np.fromstring(img_contents, np.uint8)
        # Convert numpy array to image
        parsed_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'error': str(e)}

    # Prepare for and pass through Faster RCNN
    frcnn_inp_img = parsed_img / 255 # Faster RCNN needs image normalized on all channels
    frcnn_img_tensor = torch.from_numpy(frcnn_inp_img)
    frcnn_img_tensor = frcnn_img_tensor.permute([2,0,1]).float().to(device) # chw format
    rcnn_input = [frcnn_img_tensor]
    
    frcnn_output = faster_rcnn(rcnn_input)[0]
    
    human_bboxes, object_bboxes = [], []
    object_dic = {}
    for i in range(len(frcnn_output['boxes'])):
        if frcnn_output['scores'][i] < .7:
            break
        bbox = list(map(int, frcnn_output['boxes'][i]))
        if frcnn_output['labels'][i] == 1: # human
            human_bboxes.append(bbox)
        else: 
            object_bboxes.append(bbox)
            object_dic[coco_dict[frcnn_output['labels'][i]]] = bbox
            
    # Apply masks using Faster RCNN output
    human_mask = np.zeros_like(parsed_img)
    for bbox in human_bboxes:
        cv2.rectangle(human_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
    human_bbox_img = cv2.bitwise_and(parsed_img, human_mask, mask=None)

    obj_mask = np.zeros_like(parsed_img)
    pairwise_mask = human_mask
    for bbox in object_bboxes:
        cv2.rectangle(obj_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
        cv2.rectangle(pairwise_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), thickness=-1)
    obj_bbox_img = cv2.bitwise_and(parsed_img, obj_mask, mask=None)
    pairwise_bbox_img = cv2.bitwise_and(parsed_img, pairwise_mask, mask=None)
    
    # Apply masks using OpenPose output
    candidate, subset = body_estimation(parsed_img)
    
    pose_img = np.zeros((parsed_img.shape[0], parsed_img.shape[1], 1))

    prev_point = None
    for inner_point in range(0, len(candidate)):
        pose_data_2d = candidate[inner_point]
        x, y, c = pose_data_2d[0], pose_data_2d[1], pose_data_2d[2]
        if c > 0:
            pose_img = cv2.circle(pose_img, (int(x), int(y)), 10, (255, 0,0 ), \
                                 thickness=-1, lineType=cv2.FILLED)
            if prev_point:
                pose_img = cv2.line(pose_img, (int(prev_point[0]), int(prev_point[1])), \
                                    (int(x), int(y)), (255, 0, 0), 3)
            prev_point = (x, y)
    
    # Resize images and prepare to be input to HOCNN model
    human_bbox_img = cv2.resize(human_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
    obj_bbox_img = cv2.resize(obj_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
    pairwise_bbox_img = cv2.resize(pairwise_bbox_img, (64, 64), interpolation=cv2.INTER_AREA)
    pose_img = cv2.resize(pose_img, (64, 64), interpolation=cv2.INTER_AREA)

    human_bbox_img = human_bbox_img/255
    obj_bbox_img = obj_bbox_img/255
    pairwise_bbox_img = pairwise_bbox_img/255
    pose_img = pose_img/255
    
    human_bbox_img = torch.from_numpy(human_bbox_img).to(device)
    obj_bbox_img = torch.from_numpy(obj_bbox_img).to(device)
    pairwise_bbox_img = torch.from_numpy(pairwise_bbox_img).to(device)
    pose_img = torch.from_numpy(pose_img).to(device)

    res_human_input = human_bbox_img.unsqueeze(0)
    res_obj_input = obj_bbox_img.unsqueeze(0)
    res_pairwise_input = pairwise_bbox_img.unsqueeze(0)
    res_pose_input = pose_img.unsqueeze(0)

    res_human_input = res_human_input.permute([0,3,1,2]).float().to(device)
    res_obj_input = res_obj_input.permute([0,3,1,2]).float().to(device)
    res_pairwise_input = res_pairwise_input.permute([0,3,1,2]).float().to(device)
    res_pose_input = res_pose_input.unsqueeze(3).permute([0,3,1,2]).float().to(device)
    
    # Pass to HOPOSECNN model
    with torch.no_grad(): # Disable gradients for validation
        outputs = hoposecnn_model.forward(res_human_input, res_obj_input, res_pairwise_input, res_pose_input)
        
    # Prepare and return predictions
    confidences = F.sigmoid(outputs)
    top5preds = torch.topk(confidences, 5)
    top5_idx = top5preds[1].flatten()
    top5_confidence = top5preds[0].flatten()
    
    return_predictions = {}

    for i in range(len(top5_idx)):
        prediction = hoi_list[top5_idx[i]]
        prediction_str = prediction['object'] + ' ' + prediction['verb']
        prediction_conf = round(top5_confidence[i].item() * 100, 2)
        return_predictions[prediction_str] = prediction_conf
        
    return return_predictions

'''
Cool Background Net prediction route
'''
@app.post("/api/cbgn/predict")
async def CBGN_predict(image: UploadFile = File(...)):
    try:
        img_contents = await image.read()
        # Convert string data to numpy array
        np_img = np.fromstring(img_contents, np.uint8)
        # Convert numpy array to image
        parsed_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    except Exception as e:
        return {'error': str(e)}

    node_num = []
    features = None
    spatial_feat = None
    word2vec_emb = None
    roi_labels = None
    bg = None

    # Prepare for and pass through Faster RCNN
    frcnn_inp_img = parsed_img / 255 # Faster RCNN needs image normalized on all channels
    frcnn_img_tensor = torch.from_numpy(frcnn_inp_img)
    frcnn_img_tensor = frcnn_img_tensor.permute([2,0,1]).float().to(device) # chw format
    rcnn_input = [frcnn_img_tensor]
    
    out = faster_rcnn_cbgn(rcnn_input)[0]
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

    img_wh = [parsed_img.shape[1], parsed_img.shape[0]]
    spatial_feat = spatial.calculate_spatial_feats(bboxes, img_wh)
    spatial_feat = torch.Tensor(spatial_feat).to(device)

    word2vec_emb = np.empty((0,300))
    for id in roi_labels[0]:
        vec = word2vec[coco_dict[id]]
        word2vec_emb = np.vstack((word2vec_emb, vec))
    word2vec_emb = torch.Tensor(word2vec_emb).to(device)

    mask = np.ones_like(parsed_img) * 255
    for bbox in bboxes:
        bbox = bbox.detach().cpu().numpy()
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 0), thickness=-1)
    background_img = cv2.bitwise_and(parsed_img, mask, mask=None)
    background_img_resize = cv2.resize(background_img, (64, 64), interpolation=cv2.INTER_AREA)
    background_img_tensor = torch.from_numpy(background_img)
    res_background_input = background_img_tensor.unsqueeze(0)
    res_background_input = res_background_input.permute([0,3,1,2]).float().to(device)

    with torch.no_grad():
        node_num = [len(roi_labels[0])]
        model_preds, batch_graph = cbgn_model(node_num, features, spatial_feat, word2vec_emb, roi_labels, bg=res_background_input, validation=True)
        
    model_preds = model_preds.detach().cpu()
    model_preds = torch.sigmoid(model_preds)
    preds = torch.Tensor()
    confs = torch.Tensor()

    edge_num = []
    
    for i in range(len(model_preds)):
        conf, pred = torch.topk(model_preds[i], 5)
        preds = torch.cat((preds, pred.float()))
        confs = torch.cat((confs, conf.float()))
        edge_num += [i for _ in range(5)]
    
    return_predictions = {}
    _, top5_ids = torch.topk(confs, 5)
    
    for id in top5_ids:
        verb = verbs[int(preds[id.item()].item())]
        class_ids = batch_graph.find_edges(edge_num[id.item()])
	id_1, id_2 = class_ids[0].item(), class_ids[1].item()
	label_1, label_2 = roi_labels[0][id_1], roi_labels[0][id_2]
	object = coco_dict[label_2] if label_1 == 1 else coco_dict[label_1]
        return_predictions[verb + ' ' + object] = round(confs[id].item(), 4)

    return return_predictions
