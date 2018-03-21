from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

import vae_conv_model_mnist

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--dataroot', help='path to dataset')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print("cuda", args.cuda, args.no_cuda, torch.cuda.is_available())

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# folder dataset
# dataset = datasets.ImageFolder(root=args.dataroot,
#                            transform=transforms.Compose([
#                             #    transforms.Scale(opt.imageSize),
#                             #    transforms.CenterCrop(opt.imageSize),
#                                transforms.ToTensor(),
#                             #    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
#                                          shuffle=True, **kwargs)
# test_loader = train_loader

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)

model = vae_conv_model_mnist.VAE()
model.have_cuda = args.cuda
if args.cuda:
    model.cuda()

print(model)


reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False

def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD
    # return BCE + 3 * KLD
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# print("model")

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        # break

        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

            print(data.data.size(), recon_batch.data.size())

            vutils.save_image(recon_batch.data.view(data.data.size()),
                    '%s/samples_%03d_%03d.png' % (args.outf, epoch, batch_idx),
                    normalize=True)
            vutils.save_image(data.data.view(data.data.size()),
                    '%s/real_%03d_%03d.png' % (args.outf, epoch, batch_idx),
                    normalize=True)


            # torch.save(model.state_dict(), '%s/vae_epoch_%d_%d.pth' % (args.outf, epoch, batch_idx))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

# def test(epoch):
#     model.eval()
#     test_loss = 0
#     for data, _ in test_loader:
#         if args.cuda:
#             data = data.cuda()
#         data = Variable(data, volatile=True)
#         recon_batch, mu, logvar = model(data)
#         test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#
#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

for epoch in range(1, args.epochs + 1):
    train(epoch)
# train(1)
