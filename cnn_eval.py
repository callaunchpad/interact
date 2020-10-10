from __future__ import print_function
import sys
import os
import ipdb
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

from model.model import AGRNN
from datasets.hico_constants import HicoConstants
from datasets.hico_dataset import HicoDataset, collate_fn
from datasets import metadata
import utils.io as io

# Evaluation script for CNN models.
# Should take in the following arguments:
    # GPU/CPU
    # Checkpoint path

# Check out hico_eval for refernece!


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
        model = HOCNN(feat_type=checkpoint['feat_type'], bias=checkpoint['bias'], bn=checkpoint['bn'], dropout=checkpoint['dropout'], multi_attn=checkpoint['multi_head'], layer=checkpoint['layers'], diff_edge=checkpoint['diff_edge']) #2 )
        # ipdb.set_trace()
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        print('Constructed model successfully!')
    except Exception as e:
        print('Failed to load checkpoint or construct model!', e)
        sys.exit(1)
    