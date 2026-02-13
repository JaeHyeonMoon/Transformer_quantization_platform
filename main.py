import argparse
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn

from quant import *
from quant.quantizer import UpperBitBoundQuantizer_attn, Log2Quantizer

from utils.datasets import ViTImageNetLoaderGenerator
from utils.models import get_net
from utils.test_utils import test_classification

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='PyTorch Implementation of transformer quantization')
    parser.add_argument("--data_dir", type=str, default='/data/hdd/ILSVRC2012', help='data directory')
    parser.add_argument("--seed", type=int, default=1, help='seed to use')
    parser.add_argument("--device", type=str, default='0', help='device to use')
    parser.add_argument("--name", type=str, default="vit_tiny", choices=["vit_tiny", "vit_small", "vit_base", "deit_tiny", "deit_small", "deit_base", "swin_tiny", "swin_small", "swin_base"])
    parser.add_argument("--n_groups", type=int, default=8, help='number of groups')
    parser.add_argument("--n_bits", type=int, default=4, help='bit setting for weights/activations')
    parser.add_argument("--channel_wise", type=str2bool, default=None, help='use output_channel-wise weight quantization')
    parser.add_argument("--head_wise", type=str2bool, default=None, help='use output_channel-wise weight quantization')
    # init quant model
    parser.add_argument("--scale_method", type=str, default='percentile', help='scaling method')
    parser.add_argument("--eps", type=float, default=1e-3, help='for init of quant params')
    parser.add_argument("--sequential", type=str2bool, default=True, help='sequential/parallel')
    # calibration settings
    parser.add_argument("--calib_size", type=int, default=512, help='number of calibration images')
    parser.add_argument("--calib_batch_size", type=int, default=8, help='batch size')
    parser.add_argument("--iters", type=int, default=5000, help='number of iterations')
    parser.add_argument("--lr", type=float, default=1e-4, help='learning rate for optimization')
    parser.add_argument("--p", type=float, default=2.4, help='L_p norm minimization')
    parser.add_argument("--head_stem_8bit", type=str2bool, default=False, help='use 8-bit quantization for head and stem')
    parser.add_argument("--calibration_granularity", type=str, default='block', choices = ['subblock', 'block'], help = 'granularity to calibrate')
    parser.add_argument("--test_before_calibration", action='store_true')
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    return args

def init_quant_model(cali_data, model, sequential=True):
    device = next(model.parameters()).device
    with torch.no_grad():
        model.set_quant_state(True, True)
        _ = model(cali_data.to(device))


def run(args, calib_loader, test_loader, net):
    # quantizer config
    wq_params = {'n_bits': args.n_bits, 'dim': 0, 'scale_method': (args.scale_method, args.eps), 'leaf_param': True}
    aq_params = {'n_bits': args.n_bits, 'dim': -1 if args.channel_wise else None, 'scale_method': (args.scale_method, args.eps), 'leaf_param': True}
    matmul_params = {'n_bits': args.n_bits, 'dim': None, 'scale_method': (args.scale_method, args.eps), 'leaf_param': True}
    # Initialize quantized model
    qnn = QuantModel(net, wq_params, aq_params, matmul_params, args.n_groups, args.calibration_granularity)
    qnn.cuda()
    qnn.eval()

    if "swin" in args.name:
        for module in qnn.modules():
            if isinstance(module, (UpperBitBoundQuantizer_attn, Log2Quantizer)):
                module.swin_on = True

    if args.head_stem_8bit:
        qnn.set_first_last_layer_to_8bit()

    cali_data = next(iter(calib_loader))[0]
    init_quant_model(cali_data[:32], qnn, sequential=args.sequential)
    
    channel_num = 0
    token_num = 0

    for m in qnn.model.modules():
        if hasattr(m, 'channel_num'):
            channel_num += m.channel_num
            token_num += m.token_num

    if args.test_before_calibration:
        acc = test_classification(qnn, test_loader, description='eval')
        print('Test accuracy before calibration :', acc)


if __name__ == "__main__":
    args = parse_args()
    seed_all(args.seed)
    names = {
        "vit_tiny": "vit_tiny_patch16_224",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",

        "deit_tiny": "deit_tiny_patch16_224",
        "deit_small": "deit_small_patch16_224",
        "deit_base": "deit_base_patch16_224",

        "swin_tiny": "swin_tiny_patch4_window7_224",
        "swin_small": "swin_small_patch4_window7_224",
        "swin_base": "swin_base_patch4_window7_224",
    }

    os.environ["CUDA_VISIBLE_DEVICES"]= args.device

    name = names[args.name]
    # load model
    net = get_net(name)
    for param in net.parameters():
        param.requires_grad = False
    # build data loader
    g = ViTImageNetLoaderGenerator(args.data_dir, 'imagenet', 32, 32, 4, kwargs={"model":net})
    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=args.calib_size, seed=args.seed)
    # main process
    run(args, calib_loader, test_loader, net)
