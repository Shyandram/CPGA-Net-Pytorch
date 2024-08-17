import torch
import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def weight_init_IAAF(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, std=0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
