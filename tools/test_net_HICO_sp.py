# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
# import sys
# sys.path.insert(0,'..')
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import pickle
import logging
import glob
import json
from random import randint
import cv2

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.timer import Timer, get_time_str
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
from maskrcnn_benchmark.utils.Generate_HICO_detection import Generate_HICO_detection
from maskrcnn_benchmark.data.datasets.evaluation.hico.hico_compute_mAP import compute_hico_map
from maskrcnn_benchmark.utils.bbox_utils import *


# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def bbox_trans(human_box_ori, object_box_ori, size=64):
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

def generate_spatial(human_box_aug, object_box_aug):
    # human_box = human_box.numpy()
    # object_box = object_box.numpy()

    Pattern = np.zeros((2, 64, 64))
    Pattern[0, int(human_box_aug[1]):int(human_box_aug[3]) + 1, int(human_box_aug[0]):int(human_box_aug[2]) + 1] = 1
    Pattern[1, int(object_box_aug[1]):int(object_box_aug[3]) + 1, int(object_box_aug[0]):int(object_box_aug[2]) + 1] = 1

    return Pattern

def get_pose_image(pose_dir, image_id, human_box, im_shape):
    POSE_PAIRS = ((1,8),(1,11),(1,2),(1,5),(5,6),(6,7),(2,3),(3,4),(8,9),(9,10),(11,12),(12,13),(0,1),(0,14),(14,16),(0,15),(15,16))
    pose_img_path = os.path.join(pose_dir, "HICO_test2015_%08d_keypoints.json" % image_id)
    with open(pose_img_path) as f:
        pose_data = json.load(f)
    ret = {}
    human_poses = [
        np.reshape(np.array(pose['pose_keypoints_2d']),(-1,3)) 
        for pose in pose_data['people']
    ]
    pose_boxes = [get_pose_box(pose) for pose in human_poses]
    pose_boxes_aug = [augment_box_one(box, im_shape) for box in pose_boxes]
    person_id = assign_pose(human_box, pose_boxes)
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
    for i in POSE_PAIRS:
        point0 = i[0]
        point1 = i[1]
        x0, y0, c0 = pose_data_2d[point0], pose_data_2d[point0 + 1], pose_data_2d[point0 + 2]
        x1, y1, c1 = pose_data_2d[point1], pose_data_2d[point1 + 1], pose_data_2d[point1 + 2]
        if (c0 >0 and c1>0):
            pose_img = cv2.line(pose_img, (int(64*x0//width), int(64*y0//height)), \
                                (int(64*x1//width), int(64*y1//height)), 1, 3)
    return pose_img


def augment_box_one(bbox, shape):

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

        if bbox_iou(bbox, np.array([x1, y1, x2, y2])) > thres:
            box = np.array([x1, y1, x2, y2]).astype(np.float32)
            return box
    return bbox

def bbox_iou(boxA, boxB):
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

def get_pose_box(pose):
        valid_mask = pose[:,2] > 0  # consider points with non-zero confidence
        if not np.any(valid_mask):
            return np.zeros([4])
        keypoints = pose[valid_mask,:2]
        x1,y1 = np.amin(keypoints,0)
        x2,y2 = np.amax(keypoints,0)
        box = np.array([x1,y1,x2,y2])
        return box

def assign_pose(human_box, pose_boxes):
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

def im_detect(model, pose_dir, image_id, Test_RCNN, word_embeddings, object_thres, human_thres, detection, device, opt):
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    POSE_DIR = os.path.abspath(os.path.join(pose_dir))
    im_file = os.path.join(DATA_DIR, 'hico_20160224_det', 'images', 'test2015', 'HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg')
    img_original = Image.open(im_file)
    img_original = img_original.convert('RGB')
    im_shape = (img_original.height, img_original.width)  # (480, 640)
    transforms = build_transforms(cfg, is_train=False)

    This_human = []

    for Human in Test_RCNN[image_id]:

        if (np.max(Human[5]) > human_thres) and (Human[1] == 'Human'):  # This is a valid human

            O_box = np.empty((0, 4), dtype=np.float32)
            O_vec = np.empty((0, 300), dtype=np.float32)
            Pattern = np.empty((0, 2, 64, 64), dtype=np.float32)
            O_score = np.empty((0, 1), dtype=np.float32)
            O_class = np.empty((0, 1), dtype=np.int32)

            for Object in Test_RCNN[image_id]:
                if opt['use_thres_dic'] == 1:
                    object_thres_ = opt['thres_dic'][Object[4]]
                else:
                    object_thres_ = object_thres

                if (np.max(Object[5]) > object_thres_) and not (np.all(Object[2] == Human[2])):  # This is a valid object

                    O_box_ = np.array([Object[2][0], Object[2][1], Object[2][2], Object[2][3]]).reshape(1, 4)
                    O_box = np.concatenate((O_box, O_box_), axis=0)

                    O_vec_ = word_embeddings[Object[4]]
                    O_vec = np.concatenate((O_vec, O_vec_), axis=0)

                    Pattern_ = generate_spatial(Human[2], Object[2]).reshape(1, 2, 64, 64)
                    Pattern = np.concatenate((Pattern, Pattern_), axis=0)

                    O_score = np.concatenate((O_score, np.max(Object[5]).reshape(1, 1)), axis=0)
                    O_class = np.concatenate((O_class, np.array(Object[4]).reshape(1, 1)), axis=0)

            if len(O_box) == 0:
                continue
            H_box = np.array([Human[2][0], Human[2][1], Human[2][2], Human[2][3]]).reshape(1, 4)

            blobs = {}
            blobs['pos_num'] = len(O_box)
            pos_num = len(O_box)
            human_boxes_cpu = np.tile(H_box, [len(O_box), 1]).reshape(pos_num, 4)
            human_boxes = torch.FloatTensor(human_boxes_cpu)
            object_boxes_cpu = O_box.reshape(pos_num, 4)
            object_boxes = torch.FloatTensor(object_boxes_cpu)

            human_boxlist = BoxList(human_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)
            object_boxlist = BoxList(object_boxes, img_original.size, mode="xyxy")  # image_size=(width, height)

            img, human_boxlist, object_boxlist = transforms(img_original, human_boxlist, object_boxlist)
            
            spatials = []
            human_poses = []
            for human_box, object_box in zip(human_boxlist.bbox, object_boxlist.bbox):
                h_aug, o_aug = bbox_trans(human_box.numpy(), object_box.numpy())
                ho_spatial = generate_spatial(h_aug, o_aug).reshape(1, 2, 64, 64)
                spatials.append(ho_spatial)
                pose_im_aug = get_pose_image(pose_dir, image_id, h_aug, im_shape)
                human_poses.append(pose_im_aug)
            
            num_humans = len(human_boxlist)
            blobs['spatials'] = torch.FloatTensor(spatials).reshape(-1, 2, 64, 64)
            blobs['poses'] = torch.FloatTensor(human_poses).reshape(num_humans, 1, 64, 64)
            blobs['human_boxes'], blobs['object_boxes'] = (human_boxlist,), (object_boxlist,)
            blobs['object_word_embeddings'] = torch.FloatTensor(O_vec).reshape(pos_num, 300)

            for key in blobs.keys():
                if not isinstance(blobs[key], int) and not isinstance(blobs[key], tuple):
                    blobs[key] = blobs[key].to(device)
                elif isinstance(blobs[key], tuple): 
                    blobs[key] = [boxlist.to(device) for boxlist in blobs[key]]

            image_list = to_image_list(img, cfg.DATALOADER.SIZE_DIVISIBILITY)
            image_list = image_list.to(device)

            # compute predictions
            model.eval()
            with torch.no_grad():
                prediction_HO, prediction_H, prediction_O, prediction_sp = model(image_list, blobs)

            # convert to np.array
            prediction_HO = prediction_HO.data.cpu().numpy()
            prediction_H = prediction_H.data.cpu().numpy()
            # prediction_O = prediction_O.data.cpu().numpy()
            prediction_sp = prediction_sp.data.cpu().numpy()

            for idx in range(len(prediction_HO)):
                temp = []
                temp.append(Human[2])  # Human box
                temp.append(O_box[idx])  # Object box
                temp.append(O_class[idx])  # Object class
                temp.append(prediction_HO[idx])  # Score
                temp.append(Human[5])  # Human score
                temp.append(O_score[idx])  # Object score
                This_human.append(temp)

    detection[image_id] = This_human

def run_test(
            model,
            dataset_name=None,
            pose_dir=None,
            test_detection=None,
            word_embeddings=None,
            output_file=None,
            object_thres=0.4,
            human_thres=0.6,
            device=None,
            cfg=None,
            opt=None
):
    logger = logging.getLogger("DRG.inference")
    logger.info("Start evaluation on {} dataset.".format(dataset_name))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))

    image_list = glob.glob(os.path.join(DATA_DIR, 'hico_20160224_det', 'images', 'test2015', '*.jpg'))
    np.random.seed(cfg.TEST.RNG_SEED)
    detection = {}

    for idx, line in enumerate(tqdm(image_list)):

        image_id = int(line[-9:-4])

        if image_id in test_detection:
            im_detect(model, pose_dir, image_id, test_detection, word_embeddings, object_thres, human_thres, detection, device, opt)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)

    num_devices = 1
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(image_list), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(image_list),
            num_devices,
        )
    )

    pickle.dump(detection, open(output_file, "wb"))


def main():
    #     apply_prior   prior_mask
    # 0        -             -
    # 1        Y             -
    # 2        -             Y
    # 3        Y             Y
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument('--num_iteration', dest='num_iteration',
                        help='Specify which weight to load',
                        default=-1, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.4, type=float)  # used to be 0.4 or 0.05
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.6, type=float)
    parser.add_argument('--prior_flag', dest='prior_flag',
                        help='whether use prior_flag',
                        default=1, type=int)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1 and torch.cuda.is_available()

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    print('prior flag: {}'.format(args.prior_flag))

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'Data'))
    args.config_file = os.path.join(ROOT_DIR, args.config_file)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("DRG.inference", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)

    if args.num_iteration != -1:
        args.ckpt = os.path.join(cfg.OUTPUT_DIR, 'model_%07d.pth' % args.num_iteration)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    logger.info("Testing checkpoint {}".format(ckpt))
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            if args.num_iteration != -1:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_sp", dataset_name,
                                             "model_%07d" % args.num_iteration)
            else:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_sp", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    opt = {}
    opt['word_dim'] = 300
    opt['use_thres_dic'] = 1
    for output_folder, dataset_name in zip(output_folders, dataset_names):
        data = DatasetCatalog.get(dataset_name)
        data_args = data["args"]
        test_detection = pickle.load(open(data_args['test_detection_file'], "rb"), encoding='latin1')
        pose_dir = data_args['pose_dir'] # definitely not the right place to put this
        word_embeddings = pickle.load(open(data_args['word_embedding_file'], "rb"), encoding='latin1')
        opt['thres_dic'] = pickle.load(open(data_args['threshold_dic'], "rb"), encoding='latin1')
        output_file = os.path.join(output_folder, 'detection.pkl')
        # hico_folder = os.path.join(output_folder, 'HICO')
        output_map_folder = os.path.join(output_folder, 'map')

        logger.info("Output will be saved in {}".format(output_file))
        logger.info("Start evaluation on {} dataset.".format(dataset_name))

        run_test(
            model,
            dataset_name=dataset_name,
            pose_dir=pose_dir,
            test_detection=test_detection,
            word_embeddings=word_embeddings,
            output_file=output_file,
            object_thres=args.object_thres,
            human_thres=args.human_thres,
            device=device,
            cfg=cfg,
            opt=opt
        )

        # Generate_HICO_detection(output_file, hico_folder)
        compute_hico_map(output_map_folder, output_file, 'test')


if __name__ == "__main__":
    main()
