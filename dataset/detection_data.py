import os
import ipdb
from tqdm import tqdm

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import json

import torchvision
import torch

def vis_img(img, bboxs, labels, scores=None):
    try:
        if len(bboxs)== 0:
            return img    

        if scores is not None:
            keep = np.where(scores > 0.8)
            bboxs = bboxs[keep]
            labels = labels[keep]
            scores = scores[keep]
        
        score_idx = 0 
        line_width = 1
        for (bbox, label) in zip(bboxs, labels):
            Drawer = ImageDraw.Draw(img)
            # ipdb.set_trace()
            Drawer.rectangle(list(bbox), outline='red', width=line_width)
            text = COCO_INSTANCE_CATEGORY_NAMES[label]
            if scores is None:
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red')
            else:
                text = text + " " + '{:.3f}'.format(scores[score_idx])
                Drawer.text((bbox[0]+line_width+1, bbox[1]+line_width+1), text, 'red')
                score_idx +=1
        return img

    except Exception as e:
        print("Error:" ,e)
        print("bboxs: {}, labels: {}" .format(bboxs, labels))
    finally:
        pass

COCO_INSTANCE_CATEGORY_NAMES = [
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
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

if __name__ == "__main__":
    # set up model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, rpn_post_nms_top_n_test=300, box_batch_size_per_image=128, box_score_thresh=0.05, box_nms_thresh=0.3)
    devise = torch.device('cuda:0')
    model.cuda()
    model.eval()

    # print('Begining...')
    # data = json.loads(open('anno_list.json', 'r').read())
    # half_data = int(len(data)/2)
    # for ind in tqdm(range(0, half_data)):
    #     # if data == 'train2015': continue
    #     # save_dir = os.path.join('detection_data', data) 
    #     save_dir = '/home/birl/ml_dl_projects/bigjun/hio/no_frills_hoi_det/data_symlinks/hico_processed/faster_rcnn_boxes'
    #     if not os.path.exists(save_dir):
    #         os.mkdir(save_dir)
    #     if 'train' in data[ind]['global_id']:
    #         data_path = 'train2015'
    #     else:
    #         data_path = 'test2015'
    #     root = 'hico/images/' + data_path
    #     file_list = sorted(os.listdir(root))
    #     image = Image.open(os.path.join(root, data[ind]['global_id']+'.jpg')).convert('RGB')
    #     input = torchvision.transforms.functional.to_tensor(image)
    #     input = input.to(devise)
    #     outputs = model([input], save_feat=True)
    #     np.save(os.path.join(save_dir, '{}_fc7.npy'.format(data[ind]['global_id'])), outputs[0]['box_feat'].cpu().detach().numpy())
    #     np.save(os.path.join(save_dir, '{}_boxes.npy'.format(data[ind]['global_id'])), outputs[0]['boxes'].cpu().detach().numpy())
    #     np.save(os.path.join(save_dir, '{}_scores.npy'.format(data[ind]['global_id'])), outputs[0]['scores'].cpu().detach().numpy())
    #     nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(data[ind]['global_id']))
    #     with open(nms_keep_indices_path,'w') as file:
    #         json.dump(outputs[0]['labels'], file)
    # print('Make detection data successfully!')
    #         # ipdb.set_trace()
    #         # print(len(outputs[1][0]["labels"]))
    #         # img = vis_img(image, outputs[1][0]['boxes'].cpu().detach().numpy(), outputs[1][0]['labels'].cpu().detach().numpy(), outputs[1][0]['scores'].cpu().detach().numpy())
    #         # plt.subplot(1,1,1)
    #         # plt.imshow(np.array(img))
    #         # # plt.xticks([])
    #         # # plt.yticks([])
    #         # plt.axis('off')
    #         # plt.ion()
    #         # plt.pause(5)
    #         # plt.close()

    print('Begining...')
    for data in ['train2015', 'test2015']:
        # if data == 'train2015': continue
        save_dir = os.path.join('detection_data', data) 
        # save_dir = '/home/birl/ml_dl_projects/bigjun/hio/no_frills_hoi_det/data_symlinks/hico_processed/faster_rcnn_boxes'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        root = 'hico/images/' + data
        file_list = sorted(os.listdir(root))
        for idx in tqdm(range(len(file_list))):
            image = Image.open(os.path.join(root, file_list[idx])).convert('RGB')
            # plt.figure(figsize=(100,100))
            # plt.subplot(1,2,1)
            # plt.imshow(np.array(image))
            # plt.axis('off')
            # plt.xticks([])
            # plt.yticks([])
            # ipdb.set_trace()
            input = torchvision.transforms.functional.to_tensor(image)
            input = input.to(devise)
            outputs = model([input], save_feat=True)
            # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['pool'], output_size=7, sampling_ratio=2)   #featmap_names: 0 1 2 3 'pool'
            # roi_feat = roi_pooler(outputs[2], [outputs[0]], outputs[3])
            roi_feat = model.roi_heads.box_roi_pool(outputs[2], [outputs[0]], outputs[3])
            save_feat = model.roi_heads.box_head(roi_feat)
            np.save(os.path.join(save_dir, '{}_boxes_feature.npy'.format(file_list[idx].split('.')[0])), save_feat.cpu().detach().numpy())
            np.save(os.path.join(save_dir, '{}_det_class.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['labels'].cpu().detach().numpy())
            np.save(os.path.join(save_dir, '{}_det_boxes.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['boxes'].cpu().detach().numpy())
            np.save(os.path.join(save_dir, '{}_det_scores.npy'.format(file_list[idx].split('.')[0])), outputs[1][0]['scores'].cpu().detach().numpy())
            # np.save(os.path.join(save_dir, '{}_fc7.npy'.format(file_list[idx].split('.')[0])), outputs[0]['box_feat'].cpu().detach().numpy())
            # np.save(os.path.join(save_dir, '{}_boxes.npy'.format(file_list[idx].split('.')[0])), outputs[0]['boxes'].cpu().detach().numpy())
            # np.save(os.path.join(save_dir, '{}_scores.npy'.format(file_list[idx].split('.')[0])), outputs[0]['scores'].cpu().detach().numpy())
            # nms_keep_indices_path = os.path.join(save_dir,'{}_nms_keep_indices.json'.format(file_list[idx].split('.')[0]))
            # with open(nms_keep_indices_path,'w') as file:
            #     json.dump(outputs[0]['labels'], file)
    print('Make detection data successfully!')
            # ipdb.set_trace()
            # print(len(outputs[1][0]["labels"]))
            # img = vis_img(image, outputs[1][0]['boxes'].cpu().detach().numpy(), outputs[1][0]['labels'].cpu().detach().numpy(), outputs[1][0]['scores'].cpu().detach().numpy())
            # plt.subplot(1,1,1)
            # plt.imshow(np.array(img))
            # # plt.xticks([])
            # # plt.yticks([])
            # plt.axis('off')
            # plt.ion()
            # plt.pause(5)
            # plt.close()

