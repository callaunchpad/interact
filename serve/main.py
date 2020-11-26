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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from openpose.body import Body
from model.cnn_model import HOCNN
from model.cnn_with_pose import HOPOSECNN

print('[*] Beginning server startup!')

os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='1' # Change this ID to an unused GPU
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
faster_rcnn.eval()
print('Faster rcnn loaded')

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

with open('../datasets/processed/hico/hoi_list.json') as f:
    hoi_list = json.load(f)
    
print('COCO dictionary and HOI list loaded')

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

    for i in range(len(top5_idx)):
        prediction = hoi_list[top5_idx[i]]
        prediction_str = prediction['object'] + ' ' + prediction['verb']
        prediction_conf = round(top5_confidence[i].item() * 100, 2)
        return_predictions[prediction_str] = prediction_conf
        
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