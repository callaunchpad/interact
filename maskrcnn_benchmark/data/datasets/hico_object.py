# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import pickle
import random
import numpy as np
from PIL import Image
from random import randint
import torch
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.bbox_utils import *
import json
import cv2


class HICODatasetObject(torch.utils.data.Dataset):
    def __init__(
        self, ann_file, pose_dir, root, train_val_neg_file, word_embedding_file, negative_sample_ratio, split, transforms=None
    ):
        self.img_dir = root
        self.pose_dir = pose_dir
        self.annotations = pickle.load(open(ann_file, "rb"), encoding='latin1')
        self.word_embeddings = pickle.load(open(word_embedding_file, "rb"), encoding='latin1')
        self.negative_sample_ratio = negative_sample_ratio
        self.tran_val_neg_file = pickle.load(open(train_val_neg_file, "rb"), encoding='latin1')
        self.num_classes = 117
        self.split = split
        self._transforms = transforms

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        image_id = anno[0]['image_id']
        img_path = os.path.join(self.img_dir, "HICO_train2015_%08d.jpg" % image_id)
        img = Image.open(img_path)
        img = img.convert('RGB')
        # when using Image.open to read images, img.size= (640, 480), while using cv2.imread, im.shape = (480, 640)
        # to be consistent with previous code, I used img.height, img.width here
        im_shape = (img.height, img.width) # (480, 640)

        blobs = self.bbox_augmentation(anno, image_id, im_shape)

        # create a BoxList from the boxes
        human_boxlist = BoxList(blobs['human_boxes'], img.size, mode="xyxy") # image_size=(width, height)
        object_boxlist = BoxList(blobs['object_boxes'], img.size, mode="xyxy") # image_size=(width, height)

        if self._transforms is not None:
            img, human_boxlist, object_boxlist = self._transforms(img, human_boxlist, object_boxlist)

        spatials = []
        for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
            ho_spatial = self.generate_spatial(human_box.numpy(), object_box.numpy()).reshape(1, 2, 64, 64)
            spatials.append(ho_spatial)

        blobs['spatials_object_centric'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
        blobs['human_boxes'], blobs['object_boxes'] = (human_boxlist,), (object_boxlist,)
        return img, blobs, image_id

    def get_img_info(self, index):
        img_id = self.annotations[index][0]['image_id']
        img_path = os.path.join(self.img_dir, "HICO_train2015_%08d.jpg" % img_id)
        img = Image.open(img_path)
        # width, height = img.size
        height, width = img.height, img.width
        return {
            "height": height,
            "width": width,
            "idx": index,
            "img_path": img_path,
            "ann": self.annotations[index],
        }

    def get_pose_box(self, pose):
        valid_mask = pose[:,2] > 0  # consider points with non-zero confidence
        if not np.any(valid_mask):
            return np.zeros([4])
        keypoints = pose[valid_mask,:2]
        x1,y1 = np.amin(keypoints,0)
        x2,y2 = np.amax(keypoints,0)
        box = np.array([x1,y1,x2,y2])
        return box

    def assign_pose(self, human_box, pose_boxes):
        max_idx = -1
        max_frac_inside = 0
        found_match = False
        for i, pose_box in enumerate(pose_boxes):
            iou,intersection,union = compute_iou(human_box,pose_box,True)
            pose_area = compute_area(pose_box)
            frac_inside = intersection / pose_area
            if frac_inside > max_frac_inside:
                max_frac_inside = frac_inside
                max_idx = i
                found_match = True
        return max_idx

    POSE_PAIRS = ((1,8),(1,11),(1,2),(1,5),(5,6),(6,7),(2,3),(3,4),(8,9),(9,10),(11,12),(12,13),(0,1),(0,14),(14,16),(0,15),(15,16))
    # Check other ways to approach this if we have enough time
    def get_pose_image(self, image_id, human_box, im_shape):
        pose_img_path = os.path.join(self.pose_dir, "HICO_train2015_%08d_keypoints.json" % image_id)
        with open(pose_img_path) as f:
            pose_data = json.load(f)
        ret = {}
        human_poses = [
            np.reshape(np.array(pose['pose_keypoints_2d']),(-1,3)) 
            for pose in pose_data['people']]
        pose_boxes = [self.get_pose_box(pose) for pose in human_poses]
        pose_boxes_aug = [self.augment_box_one(box, im_shape) for box in pose_boxes]
        person_id = self.assign_pose(human_box, pose_boxes)
        if person_id == -1:
            return np.zeros((64, 64, 1))
        person = pose_data['people'][person_id]
        pose_img = np.zeros((64, 64, 1))
        pose_data_2d = person['pose_keypoints_2d']
        width = im_shape[1]
        height = im_shape[0]
        for inner_point in range(0, len(pose_data_2d), 3):
            x, y, c = pose_data_2d[inner_point], pose_data_2d[inner_point + 1], pose_data_2d[inner_point + 2]
            if c > 0:
                pose_img = cv2.circle(pose_img, (int(64*x//width), int(64*y//height)), 10, 1, \
                                        thickness=-1, lineType=cv2.FILLED)
        for i in HICODatasetObject.POSE_PAIRS:
            point0 = i[0]
            point1 = i[1]
            x0, y0, c0 = pose_data_2d[point0], pose_data_2d[point0 + 1], pose_data_2d[point0 + 2]
            x1, y1, c1 = pose_data_2d[point1], pose_data_2d[point1 + 1], pose_data_2d[point1 + 2]
            if (c0 >0 and c1>0):
                pose_img = cv2.line(pose_img, (int(64*x0//width), int(64*y0//height)), \
                                    (int(64*x1//width), int(64*y1//height)), 1, 3)
        return pose_img

    def __len__(self):
        return len(self.annotations)

    def bbox_augmentation(self, anno, image_id, im_shape):
        # initialization
        human_boxes = []
        object_boxes = []

        human_labels = []
        object_labels = []
        ho_pair_labels = []

        object_word_embeddings = []
        mask = []
        human_poses = []

        # ground truth box augmentation
        for human_object_pair in anno:
            human_box_aug = self.augment_box_one(human_object_pair['human_box'], im_shape)
            human_boxes.append(human_box_aug)

            pose_im_aug = self.get_pose_image(image_id, human_box_aug, im_shape)
            human_poses.append(pose_im_aug)

            object_box_aug = self.augment_box_one(human_object_pair['object_box'], im_shape)
            object_boxes.append(object_box_aug)

            human_verbs_to_vector = self.verb_list_to_vector(human_object_pair['human_action_id_list'])
            human_labels.append(human_verbs_to_vector)

            object_verbs_to_vector = self.verb_list_to_vector(human_object_pair['object_action_id_list'])
            object_labels.append(object_verbs_to_vector)

            ho_verbs_to_vector = self.verb_list_to_vector(human_object_pair['verb_id_list'])
            ho_pair_labels.append(ho_verbs_to_vector)

            object_class = human_object_pair['object_class']
            object_word_embeddings.append(self.word_embeddings[object_class].reshape(300))

            #@Done made bbox to shape (1,4), not (1,5), so modified the code here
            #@Done changed to (1, 2, 64, 64)
            # ho_spatial = self.generate_spatial(human_box_aug, object_box_aug).reshape(1, 2, 64, 64)
            # spatials.append(ho_spatial)

            curr_mask = self.generate_mask(human_object_pair['possible_verb_with_object'])
            mask.append(curr_mask)

        num_pos = len(human_boxes)

        if image_id in self.tran_val_neg_file.keys():

            human_boxes_neg = []
            object_boxes_neg = []
            object_word_embeddings_neg = []
            human_poses_neg = []
            # spatials_neg = []
            mask_neg = []

            for negative_pair in self.tran_val_neg_file[image_id]:
                if self.bbox_iou(negative_pair['human_box'], anno[0]['human_box']) > 0.6:
                    human_box_neg_aug = self.augment_box_one(negative_pair['human_box'], im_shape)
                    human_boxes_neg.append(human_box_neg_aug)

                    object_box_neg_aug = self.augment_box_one(negative_pair['object_box'], im_shape)
                    object_boxes_neg.append(object_box_neg_aug)

                    object_class = negative_pair['object_class']
                    object_word_embeddings_neg.append(self.word_embeddings[object_class].reshape(300))

                    pose_im_aug_neg = self.get_pose_image(image_id, human_box_neg_aug, im_shape)
                    human_poses_neg.append(pose_im_aug_neg)

                    curr_mask = self.generate_mask(negative_pair['possible_verb_with_object'])
                    mask_neg.append(curr_mask)

            # use all Neg example
            if self.negative_sample_ratio != -1:
                # subsample negative examples if we have too many
                if len(human_boxes_neg) >= self.negative_sample_ratio * num_pos:
                    idx_list = random.sample(range(len(human_boxes_neg)), len(human_boxes_neg))
                    idx_list = idx_list[:self.negative_sample_ratio * num_pos]

                    human_boxes_neg = [human_boxes_neg[i] for i in idx_list]
                    human_poses_neg = [human_poses_neg[i] for i in idx_list]
                    object_boxes_neg = [object_boxes_neg[i] for i in idx_list]
                    object_word_embeddings_neg = [object_word_embeddings_neg[i] for i in idx_list]
                    mask_neg = [mask_neg[i] for i in idx_list]

                # generate more negative examples if we have too few
                if len(human_boxes_neg) < self.negative_sample_ratio * num_pos and len(human_boxes_neg) != 0:
                    idx_list = np.random.choice(len(human_boxes_neg), self.negative_sample_ratio * num_pos - len(human_boxes_neg)).tolist()

                    human_boxes_neg += [human_boxes_neg[i] for i in idx_list]
                    human_poses_neg += [human_poses_neg[i] for i in idx_list]
                    object_boxes_neg += [object_boxes_neg[i] for i in idx_list]
                    object_word_embeddings_neg += [object_word_embeddings_neg[i] for i in idx_list]
                    mask_neg += [mask_neg[i] for i in idx_list]

            human_boxes += human_boxes_neg
            object_boxes += object_boxes_neg
            object_word_embeddings += object_word_embeddings_neg
            human_poses += human_poses_neg
            mask += mask_neg

        num_pos_neg = len(human_boxes)

        if cfg.DATASETS.NEG_VERB_ALLZERO == 1:
            for _ in range(num_pos_neg - num_pos):
                ho_pair_labels.append(np.zeros(self.num_classes))
        elif cfg.DATASETS.NEG_VERB_ALLZERO == 0:
            for _ in range(num_pos_neg - num_pos):
                ho_pair_labels.append(self.verb_list_to_vector([57]))
        else:
            assert(0)

        assert len(ho_pair_labels)==num_pos_neg


        blobs = {}
        # @Done made bbox to shape (1,4), not (1,5), so modified the code here
        blobs['poses'] = torch.FloatTensor(human_poses).reshape(num_pos_neg, 1, 64, 64)
        blobs['human_boxes'] = torch.FloatTensor(human_boxes).reshape(num_pos_neg, 4)
        blobs['object_boxes'] = torch.FloatTensor(object_boxes).reshape(num_pos_neg, 4)
        blobs['ho_pair_labels_object_centric'] = torch.FloatTensor(ho_pair_labels).reshape(num_pos_neg, self.num_classes)
        blobs['object_word_embeddings_object_centric'] = torch.FloatTensor(object_word_embeddings).reshape(num_pos_neg, 300)
        blobs['mask_ho'] = torch.FloatTensor(mask).reshape(num_pos_neg, self.num_classes)
        blobs['human_labels'] = torch.FloatTensor(human_labels).reshape(num_pos, self.num_classes)
        blobs['object_labels'] = torch.FloatTensor(object_labels).reshape(num_pos, self.num_classes)
        blobs['pos_num'] = num_pos

        return blobs


    def augment_box_one(self, bbox, shape):

        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        y_center = (bbox[3] + bbox[1]) / 2
        x_center = (bbox[2] + bbox[0]) / 2

        thres = 0.7

        for count in range(20):

            ratio = 1 + randint(-10, 10) * 0.01

            y_shift = randint(-np.floor(height), np.floor(height)) * 0.1
            x_shift = randint(-np.floor(width), np.floor(width)) * 0.1

            x1 = max(0, x_center + x_shift - ratio * width / 2)
            x2 = min(shape[1] - 1, x_center + x_shift + ratio * width / 2)
            y1 = max(0, y_center + y_shift - ratio * height / 2)
            y2 = min(shape[0] - 1, y_center + y_shift + ratio * height / 2)

            if self.bbox_iou(bbox, np.array([x1, y1, x2, y2])) > thres:
                box = np.array([x1, y1, x2, y2]).astype(np.float32)
                return box

        # @Done removed the first 0 here,
        # because I want to directly use the transform function
        # and it can be directly processed by pooler
        # return np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5).astype(np.float32)
        return bbox

    def bbox_iou(self, boxA, boxB):

        ixmin = np.maximum(boxA[0], boxB[0])
        iymin = np.maximum(boxA[1], boxB[1])
        ixmax = np.minimum(boxA[2], boxB[2])
        iymax = np.minimum(boxA[3], boxB[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
               (boxA[2] - boxA[0] + 1.) *
               (boxA[3] - boxA[1] + 1.) - inters)

        overlaps = inters / uni
        return overlaps

    def verb_list_to_vector(self, action_list):
        action_ = np.zeros(self.num_classes)
        for GT_idx in action_list:
            action_[GT_idx] = 1
        # action_ = action_.reshape(1, self.num_classes)
        return action_

    def generate_mask(self, mask_list):
        mask_ = np.zeros(self.num_classes)
        for GT_idx in mask_list:
            mask_[GT_idx] = 1
        # mask_ = mask_.reshape(1, self.num_classes)
        return mask_

    def generate_spatial(self, human_box, object_box):

        # InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
        #                       max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        # height = InteractionPattern[3] - InteractionPattern[1] + 1
        # width = InteractionPattern[2] - InteractionPattern[0] + 1
        # if height > width:
        #     H, O = self.bbox_trans(human_box, object_box, 'height')
        # else:
        #     H, O = self.bbox_trans(human_box, object_box, 'width')

        # human_box = human_box.numpy()
        # object_box = object_box.numpy()
        H, O = self.bbox_trans(human_box, object_box)

        Pattern = np.zeros((2, 64, 64))
        Pattern[0, int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1] = 1
        Pattern[1, int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1] = 1

        return Pattern

    def bbox_trans(self, human_box_ori, object_box_ori, size=64):
        human_box = human_box_ori.copy()
        object_box = object_box_ori.copy()

        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                              max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

        height = InteractionPattern[3] - InteractionPattern[1] + 1
        width = InteractionPattern[2] - InteractionPattern[0] + 1

        if height > width:
            ratio = 'height'
        else:
            ratio = 'width'

        # shift the top-left corner to (0,0)

        human_box[0] -= InteractionPattern[0]
        human_box[2] -= InteractionPattern[0]
        human_box[1] -= InteractionPattern[1]
        human_box[3] -= InteractionPattern[1]
        object_box[0] -= InteractionPattern[0]
        object_box[2] -= InteractionPattern[0]
        object_box[1] -= InteractionPattern[1]
        object_box[3] -= InteractionPattern[1]

        if ratio == 'height':  # height is larger than width

            human_box[0] = 0 + size * human_box[0] / height
            human_box[1] = 0 + size * human_box[1] / height
            human_box[2] = (size * width / height - 1) - size * (width - 1 - human_box[2]) / height
            human_box[3] = (size - 1) - size * (height - 1 - human_box[3]) / height

            object_box[0] = 0 + size * object_box[0] / height
            object_box[1] = 0 + size * object_box[1] / height
            object_box[2] = (size * width / height - 1) - size * (width - 1 - object_box[2]) / height
            object_box[3] = (size - 1) - size * (height - 1 - object_box[3]) / height

            # Need to shift horizontally
            InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                                  max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
            # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
            if human_box[3] > object_box[3]:
                human_box[3] = size - 1
            else:
                object_box[3] = size - 1

            shift = size / 2 - (InteractionPattern[2] + 1) / 2

            human_box += [shift, 0, shift, 0]
            object_box += [shift, 0, shift, 0]



        else:  # width is larger than height

            human_box[0] = 0 + size * human_box[0] / width
            human_box[1] = 0 + size * human_box[1] / width
            human_box[2] = (size - 1) - size * (width - 1 - human_box[2]) / width
            human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width

            object_box[0] = 0 + size * object_box[0] / width
            object_box[1] = 0 + size * object_box[1] / width
            object_box[2] = (size - 1) - size * (width - 1 - object_box[2]) / width
            object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width

            # Need to shift vertically
            InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                                  max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]

            # assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)

            if human_box[2] > object_box[2]:
                human_box[2] = size - 1
            else:
                object_box[2] = size - 1

            shift = size / 2 - (InteractionPattern[3] + 1) / 2

            human_box = human_box + [0, shift, 0, shift]
            object_box = object_box + [0, shift, 0, shift]

        return np.round(human_box), np.round(object_box)

    def add_image_mean(self, image):
        # add back mean
        image = image + cfg.INPUT.PIXEL_MEAN
        return image

    def vis_bbox(self, image, H_box, O_box):
        import matplotlib.pyplot as plt
        dpi = 80
        # im_data = plt.imread(im_file)
        # im_data = np.concatenate((im_data, im_data), axis=1)
        from torchvision.transforms import functional as F
        try:
            image = self.add_image_mean(image)
            image = F.to_pil_image(image)
        except:
            image = image
        width, height = image.size
        GT_color = 0
        cc = plt.get_cmap('hsv', lut=6)
        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.imshow(image, interpolation='nearest') # imshow(im_data, interpolation='nearest')

        ax.add_patch(
            plt.Rectangle((H_box[0], H_box[1]),
                          H_box[2] - H_box[0],
                          H_box[3] - H_box[1], fill=False,
                          edgecolor='b', linewidth=3)
        )
        ax.add_patch(
            plt.Rectangle((O_box[0], O_box[1]),
                          O_box[2] - O_box[0],
                          O_box[3] - O_box[1], fill=False,
                          edgecolor='b', linewidth=3)
        )
        plt.plot([(H_box[0] + H_box[2]) / 2, (O_box[0] + O_box[2]) / 2],
                 [(H_box[1] + H_box[3]) / 2, (O_box[1] + O_box[3]) / 2],
                 marker='o', color='g', linewidth=3, markersize=6)
        plt.show()

        # ax.text(O_box[0] + 8, O_box[1] + 17,
        #         HICO_vb_inv[HICO_HOI[ele[1][0]]],
        #         bbox=dict(facecolor=cc(GT_color)[:3], alpha=0.5),
        #         fontsize=10, color='white')
