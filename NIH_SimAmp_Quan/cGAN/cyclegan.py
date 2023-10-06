import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from tqdm import tqdm

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--generator_type", type=str, default='resnet', help="generator_type")
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_steps", type=int, default=1000000, help="number of steps of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
dir = './noise_enhance'
sample_dir = os.path.join(dir, f'samples_cgan_{opt.generator_type}')
ckpt_dir = os.path.join(dir, f'ckpt_cgan_{opt.generator_type}')
clean_train_dir = os.path.join(dir, 'data/clean_trainset')
clean_test_dir = os.path.join(dir, 'data/clean_testset')
noisy_train_dir = os.path.join(dir, 'data/noisy_trainset')
noisy_test_dir = os.path.join(dir, 'data/noisy_testset')
train_sample = os.path.join(noisy_train_dir, 'p226_001.wav')
test_sample = os.path.join(noisy_test_dir, 'p226_024.wav')
os.makedirs(sample_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)
print(f'{dir}\t{sample_dir}\t{ckpt_dir}')
print(f'{clean_train_dir}\t{clean_test_dir}\t{noisy_train_dir}\t{noisy_test_dir}')
print(f'{train_sample}\t{test_sample}')

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
if opt.generator_type == 'unet':
    G_AB = UnetGenerator(input_nc=1, output_nc=1, num_downs=7, use_dropout=True)
    G_BA = UnetGenerator(input_nc=1, output_nc=1, num_downs=7, use_dropout=True)
else:
    G_AB = ResnetGenerator(
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=opt.n_residual_blocks,
        padding_type='reflect'
    )
    G_BA = ResnetGenerator(
        input_nc=1,
        output_nc=1,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=opt.n_residual_blocks,
        padding_type='reflect'
    )
D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.to(device)
    G_BA = G_BA.to(device)
    D_A = D_A.to(device)
    D_B = D_B.to(device)
    criterion_GAN.to(device)
    criterion_cycle.to(device)
    criterion_identity.to(device)

# Initialize weights
G_AB.apply(weights_init_normal)
G_BA.apply(weights_init_normal)
D_A.apply(weights_init_normal)
D_B.apply(weights_init_normal)

# Training data loader
dataset = SpecDataset(noisy_dir=noisy_train_dir, gt_dir=clean_train_dir, device=device)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
n_epochs = (opt.n_steps // len(dataloader)) + 1
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()


def sample_image(steps_done):
    # Sample from test set
    input_img = dataset.getitem_helper(test_sample)
    gen_imgs = G_AB(input_img.unsqueeze(0).to(device))
    save_image(gen_imgs.squeeze(0).data, f"./{sample_dir}/test_step%d.png" % steps_done, normalize=True)
    # Sample from train set
    input_img = dataset.getitem_helper(train_sample)
    gen_imgs = G_AB(input_img.unsqueeze(0).to(device))
    save_image(gen_imgs.squeeze(0).data, f"./{sample_dir}/train_step%d.png" % steps_done, normalize=True)

# ----------
#  Training
# ----------
n_epochs = (opt.n_steps // len(dataloader)) + 1
curr_step = 0
isContinue = True
G_loss = 0.0
D_loss = 0.0

prev_time = time.time()
with tqdm(total=opt.n_steps) as pbar:
    for epoch in range(opt.epoch, n_epochs):
        for i, batch in enumerate(dataloader):
            # count step
            curr_step += 1
            isContinue = curr_step < opt.n_steps
            if not isContinue: break

            # Set model input
            real_A = Variable(batch["noisy"].type(Tensor))
            real_B = Variable(batch["gt"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False).to(device)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False).to(device)

            # ------------------
            #  Train Generators
            # ------------------

            G_AB.train()
            G_BA.train()

            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss
            fake_B = G_AB(real_A)
            loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
            fake_A = G_BA(real_B)
            loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()
            G_loss += loss_G.item()

            # -----------------------
            #  Train Discriminator A
            # -----------------------

            optimizer_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_A(real_A), valid)
            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
            # Total loss
            loss_D_A = (loss_real + loss_fake) / 2

            loss_D_A.backward()
            optimizer_D_A.step()

            # -----------------------
            #  Train Discriminator B
            # -----------------------

            optimizer_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(D_B(real_B), valid)
            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2
            D_loss += loss_D.item()
            
            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()


            if curr_step % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, D_loss/opt.sample_interval, G_loss/opt.sample_interval)
                )
                G_loss = D_loss = 0.0
                sample_image(curr_step)

                # Save model checkpoints
                torch.save(G_AB.state_dict(), "ckpt_cycleGAN/G_AB.pt")
                torch.save(G_BA.state_dict(), "ckpt_cycleGAN/G_BA.pt")
                torch.save(D_A.state_dict(), "ckpt_cycleGAN/D_A.pt")
                torch.save(D_B.state_dict(), "ckpt_cycleGAN/D_B.pt")
            pbar.update()
        
        if not isContinue: break