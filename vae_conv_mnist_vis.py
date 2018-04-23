from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import numpy as np
import collections
from collections import OrderedDict

import datetime
import os

import vae_conv_model_mnist

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--model', default='model.pth', help='saved model file')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("cuda", args.cuda, args.no_cuda, torch.cuda.is_available())

params = 20

model = vae_conv_model_mnist.VAE(params)
model.have_cuda = args.cuda
if args.cuda:
    model.cuda()

if args.cuda:
    model.load_state_dict(torch.load(args.model))
else:
    model.load_state_dict(torch.load(args.model, map_location={'cuda:0': 'cpu'}))

np.set_printoptions(threshold=500000,linewidth=1000)
print(model)

# Summarize Model
from pytorch_summary import Summary
s = Summary(model.encoder, input_size=(1, 1, 28, 28))
s = Summary(model.decoder, input_size=(1, 1024, 1, 1))

side_x = 40
side_y = 20
z_input = np.full((side_x*side_y,params), 0.0)
# print(z_input.shape)

for i in range(side_y):
    for j in range(side_x):
        z_input[i*side_x+j][i] = (j-side_x/2.0) * 0.1
        # z_input[i*side+j][1] = (j-side/2.0) * 0.1

# for i in range(side):
#     for j in range(side):
#         z_input[i*side+j][0] = (i-side/2.0) * 0.1
#         z_input[i*side+j][1] = (j-side/2.0) * 0.1

# print(z_input)

if args.cuda:
    z_batch = torch.cuda.FloatTensor(z_input)
else:
    z_batch = torch.FloatTensor(z_input)
z_batch = Variable(z_batch)
vis_batch = model.decode(z_batch)

outf = args.outf
save_image(vis_batch.data.cpu(), outf + '/test.png', nrow=side_x)
