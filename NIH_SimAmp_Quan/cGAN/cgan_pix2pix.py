import argparse
import os
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
import torchaudio.transforms as T

from torch.cuda.amp import GradScaler, autocast
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from models import ResnetGenerator, Discriminator, UnetGenerator, ContextEncoder, NLayerDiscriminator
from models import weights_init_normal
from datasets import SpecDataset
from utils import save_spectrogram, normalize_spec, denormalize_spec, db_to_power, power_to_db
###################
# START EDITING
###################
''' NOTE:
- use_context_loss -> use_audio_processor
- run: use_mixed_precision -> NO use_context_loss
'''
parser = argparse.ArgumentParser()
parser.add_argument("--my_id", type=str, default='test')
parser.add_argument("--generator_type", type=str, default='resnet')
parser.add_argument("--dir", type=str, default='./PD_enhance')
parser.add_argument("--pretrain_dir", type=str, default=None)
parser.add_argument("--use_context_embed", action='store_true', default=False)
parser.add_argument("--use_context_loss", action='store_true', default=False)
parser.add_argument("--use_audio_processor", action='store_true', default=True) # this is very slow
parser.add_argument("--hop_length_rate", type=int, default=4, help="rate which hop_length smaller than n_fft, if =2, spec.shape=(n_mels, 128), =8, shape=(n_mels, 512), better sound")
parser.add_argument("--use_DDP", action='store_true', default=False) # somehow it appears error after a while
parser.add_argument("--n_steps", type=int, default=10, help="number of training steps") # 100000
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument('--verbose_interval', type=int, default=100, help='interval between printing loss')
parser.add_argument("--num_residual_blocks", type=int, default=8, help="num_residual_blocks if use cycleGAN generator")
parser.add_argument("--n_conv_layers", type=int, default=5, help="num_conv_layers for NLayerDiscriminator")
parser.add_argument("--lambda_L1", type=float, default=5.0, help="lambda_L1 for G loss") # Pix2Pix = 100.
parser.add_argument("--lambda_ctx", type=float, default=1.0, help="lambda_ctx for G loss") # Pix2Pix = 100.
parser.add_argument("--gamma", type=float, default=0.1, help="gamma for LR scheduler")
opt = parser.parse_args()

dir = opt.dir
sample_dir = os.path.join(dir, f'cgan_{opt.generator_type}_{opt.my_id}_samp')
ckpt_dir = os.path.join(dir, f'cgan_{opt.generator_type}_{opt.my_id}_ckpt')
clean_train_dir = os.path.join(dir, 'data/clean_trainset')
clean_test_dir = os.path.join(dir, 'data/clean_testset')
noisy_train_dir = os.path.join(dir, 'data/noisy_trainset')
noisy_test_dir = os.path.join(dir, 'data/noisy_testset')

sample = os.listdir(noisy_test_dir)[0]
sample_A = os.path.join(noisy_test_dir, sample)
sample_B = os.path.join(clean_test_dir, sample)

###################
# END EDITING
###################

cuda = True if torch.cuda.is_available() else False
device = f'cuda' if cuda else 'cpu'
master_process = True

# Enable DDP
# Initialize the process group
if opt.use_DDP:
    init_process_group(backend="nccl")
    # Get the DDP rank
    ddp_rank = int(os.environ['RANK'])
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    # Get the DDP local rank
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    # Set the cuda device
    device = f'cuda:{ddp_local_rank}'
    if master_process: print('ddp_rank:', ddp_rank)

# Print configs
if master_process:
    os.makedirs(dir, exist_ok=True)
    print (f'Current time: {datetime.now().strftime("DATE %d/%m/%Y - TIME: %H:%M:%S")}')
    print('\n\t'.join(['Config:']+[f'{k} = {v}' for k, v in vars(opt).items()])) # print args
    print(f'Find {torch.cuda.device_count()} gpus available')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f'{dir}\n\t{sample_dir}\n\t{ckpt_dir}')
    print(f'{clean_train_dir}\n\t{clean_test_dir}\n\t{noisy_train_dir}\n\t{noisy_test_dir}')
    print(f'{sample_A}\n\t{sample_B}')
    for f in os.listdir(sample_dir):
        try:
            os.remove(os.path.join(sample_dir, f))
        except:
            pass

# Configure data loader
dataset = SpecDataset(noisy_dir=noisy_train_dir,
                      gt_dir=clean_train_dir,
                      device=device,
                      use_audio_processor=opt.use_audio_processor,
                      hop_length_rate=opt.hop_length_rate)
if opt.use_DDP:
    dataloader = DataLoader(dataset, sampler=DistributedSampler(dataset), batch_size=opt.batch_size, shuffle=False)
else:
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)
if master_process: print(f'CREATED Dataset size={len(dataset)} - DataLoader size={len(dataloader)}')

# Loss functions
l1_loss = nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)
if master_process: print('CREATED Losses')

# Compute in/output_nc
output_nc = 1
input_nc = 1
add_input_nc = 0
if opt.use_context_embed:
    enc_h, enc_w = ContextEncoder.output_size # (203, 1024)
    spec, _, _, _ = dataset[0]['gt']
    inp_h, inp_w = spec.shape[-2:]
    add_input_nc = int((enc_h*enc_w) // (inp_h*inp_w))

# Initialize generator and discriminator
if opt.use_context_embed or opt.use_context_loss:
    # Must keep some weights.requires_grad=True to have its output as gradient
    encoder = ContextEncoder(device).to(device)
    if opt.use_DDP: encoder = DDP(encoder, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    if master_process: print('CREATED ContextEncoder')
if opt.generator_type == 'unet': # not check if it runs
    generator = UnetGenerator(input_nc=input_nc, output_nc=1, num_downs=7, use_dropout=False).to(device)
    if master_process: print('CREATED UnetGenerator')
else:
    if master_process: print('input num channels = ', input_nc)
    generator = ResnetGenerator(
        input_nc=input_nc + add_input_nc,
        output_nc=output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=opt.num_residual_blocks,
        padding_type='reflect'
    ).to(device)
    if master_process: print('CREATED ResnetGenerator')

discriminator = NLayerDiscriminator(
    input_nc=input_nc + output_nc,
    n_layers=opt.n_conv_layers
).to(device)
if master_process: print('CREATED NLayerDiscriminator')

if opt.pretrain_dir is not None:
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    try:
        state_dict_G = torch.load(os.path.join(opt.pretrain_dir, 'G_model.pt'))
        state_dict_D = torch.load(os.path.join(opt.pretrain_dir, 'D_model.pt'))
        generator.load_state_dict(state_dict_G)
        discriminator.load_state_dict(state_dict_D)
        if master_process: print('LOADED WEIGHTS for G & D')
    except Exception as e:
        try:
            new_state_dict_G = OrderedDict()
            for k, v in state_dict_G.items():
                name = k[7:] # remove "module."
                new_state_dict_G[name] = v
            generator.load_state_dict(new_state_dict_G)

            new_state_dict_D = OrderedDict()
            for k, v in state_dict_D.items():
                name = k[7:] # remove "module."
                new_state_dict_D[name] = v
            discriminator.load_state_dict(new_state_dict_D)
            if master_process: print('LOADED WEIGHTS for G & D')
        except Exception as e:
            if master_process: print(e, '\nCannot load model')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    if master_process: print('INITIALIZED WEIGHT for G & D')

if opt.use_DDP:
    # https://github.com/pytorch/pytorch/issues/22095
    generator = DDP(generator, device_ids=[ddp_local_rank], output_device=ddp_local_rank, broadcast_buffers=False)
    discriminator = DDP(discriminator, device_ids=[ddp_local_rank], output_device=ddp_local_rank, broadcast_buffers=False)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
if master_process: print('CREATED Optimizers')

# Schedulers
milestones=[int(opt.n_steps*0.5), int(opt.n_steps*0.75)]
scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=milestones, gamma=opt.gamma)
if master_process: print('CREATED Schedulers')

def set_requires_grad(model, requires_grad=False):
    for p in model.parameters():
        p.requires_grad = requires_grad

def sample_image(steps_done):
    generator.eval()
    if opt.use_context_embed: encoder.eval()
    with torch.no_grad():
        if steps_done == 0:
            for i, sample in zip(['A', 'B'], [sample_A, sample_B]):
                samplename = os.path.basename(sample).replace('.wav', '')
                inp_A, wav_A, _, _ = dataset.getitem_helper(sample) # input_img = [1, 128, 128]
                inp_A = inp_A[0].detach().clone().cpu().numpy()
                outpath = os.path.join(sample_dir, f'{samplename}_{i}.png')
                save_spectrogram(inp_A, outpath)
            return
        sample = sample_A
        samplename = os.path.basename(sample).replace('.wav', '')
        inp_A, wav_A, _, _ = dataset.getitem_helper(sample) # input_img = [1, 128, 128]
        inp_A = inp_A.unsqueeze(0).to(device) # add batch
        wav_A = wav_A.unsqueeze(0).to(device)

        emb_A = encoder(wav_A) if opt.use_context_embed else None
        out_B = generator(inp_A, emb_A).squeeze(0) # remove batch
        out_B = out_B[0].detach().clone().cpu().numpy()
        out_B, _ = normalize_spec(out_B)

        outpath = os.path.join(sample_dir, f"step{steps_done}.png")
        save_spectrogram(out_B, outpath)

if master_process: sample_image(0)

def get_target_tensor(prediction, target_is_real):
    if target_is_real:
        target_tensor = torch.tensor(0.0, device=device, requires_grad=False)
    else:
        target_tensor = torch.tensor(1.0, device=device, requires_grad=False)
    return target_tensor.expand_as(prediction)

def forward_G(generator, real_A_spec, real_A_wav):
    """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    emb = encoder(real_A_wav) if opt.use_context_embed else None
    fake_B_spec = generator(real_A_spec, emb)  # G(A)
    return fake_B_spec

def backward_D(discriminator, real_A_spec, real_B_spec, fake_B_spec):
    loss_D_fake = loss_D_real = torch.tensor(0.)
    # --------------------
    #  ERROR IN backward_D
    #  REAL & FAKE works individualy
    # --------------------
    """Calculate GAN loss for the discriminator"""
    torch.autograd.set_detect_anomaly(True)

    # Real
    real_AB = torch.cat((real_A_spec, real_B_spec), 1)
    pred_real = discriminator(real_AB)
    target_tensor1 = get_target_tensor(pred_real, True)
    loss_D_real = adversarial_loss(pred_real.clone(), target_tensor1.clone())
    
    # Fake; stop backprop to the generator by detaching fake_B
    fake_AB = torch.cat((real_A_spec, fake_B_spec), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    pred_fake = discriminator(fake_AB.detach())
    target_tensor2 = get_target_tensor(pred_fake, False)
    loss_D_fake = adversarial_loss(pred_fake.clone(), target_tensor2.clone())
    
    # combine loss and calculate gradients
    loss_D = (loss_D_fake + loss_D_real) * 0.5
    loss_D.backward()
    return loss_D.item()

def backward_G(discriminator, real_A_spec, real_B_spec, fake_B_spec):
    """Calculate GAN and L1 loss for the generator"""
    # First, G(A) should fake the discriminator
    fake_AB = torch.cat((real_A_spec, fake_B_spec), 1)
    pred_fake = discriminator(fake_AB)
    target_tensor = get_target_tensor(pred_fake, True)
    loss_G_GAN = adversarial_loss(pred_fake, target_tensor)
    # Second, G(A) = B
    loss_G_L1 = l1_loss(fake_B_spec, real_B_spec) * opt.lambda_L1
    # Third, 
    loss_G_ctx = torch.tensor(0.)
    if opt.use_context_loss:
        fake_B_wav = dataset.spec_to_audio(fake_B_spec, filer, scale)
        fake_B_wav_emb, real_B_wav_emb = encoder(fake_B_wav), encoder(real_B_wav)
        loss_G_ctx = adversarial_loss(fake_B_wav_emb, real_B_wav_emb) * opt.lambda_ctx
    # combine loss and calculate gradients
    loss_G = loss_G_GAN + loss_G_L1 + loss_G_ctx
    loss_G.backward()
    return loss_G.item()

# ----------
#  Training
# ----------
n_epochs = (opt.n_steps // len(dataloader)) + 1
if master_process: print(f'Model will be trained for: {opt.n_steps} steps or {n_epochs} EPOCHS')
curr_step = 0
isContinue = True
G_loss = 0.0
D_loss = 0.0

with torch.autograd.set_detect_anomaly(True):

    discriminator.train()
    for epoch in range(n_epochs):
        if master_process:
            train_progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [train {opt.my_id}]", position=0, leave=True)
        else:
            train_progress_bar = dataloader
        for batch in train_progress_bar:
            generator.train()
            if opt.use_context_embed: encoder.train()
            
            #  train model domain A -> B
            real_A_spec, real_A_wav, filer, scale = batch['noisy']
            real_B_spec, real_B_wav, _, _ = batch['gt']
            
            # forward G
            fake_B_spec = forward_G(generator, real_A_spec, real_A_wav)
            
            # update D
            set_requires_grad(discriminator, True)
            optimizer_D.zero_grad()
            G_loss += backward_D(discriminator, real_A_spec, real_B_spec, fake_B_spec)
            optimizer_D.step()

            # update G
            set_requires_grad(discriminator, False) # D requires no gradients when optimizing G
            optimizer_G.zero_grad()
            G_loss += backward_G(discriminator, real_A_spec, real_B_spec, fake_B_spec)
            optimizer_G.step()

            scheduler_G.step()
            
            # Print result
            if master_process:
                if curr_step % opt.sample_interval == 0:
                    sample_image(curr_step)
                    torch.save(generator.state_dict(), f'./{ckpt_dir}/G_model.pt')
                    torch.save(discriminator.state_dict(), f'./{ckpt_dir}/D_model.pt')
                if curr_step % opt.verbose_interval == 0:
                    print("[Step %d/%d] [D loss: %f] [G loss: %f] [time: %s]"
                        % (curr_step, opt.n_steps, D_loss/opt.verbose_interval, G_loss/opt.verbose_interval, f'{datetime.now().strftime("%d/%m/%Y-%H:%M:%S")}'))
                    G_loss = D_loss = 0.0
            
            # count step
            curr_step += 1
            isContinue = curr_step < opt.n_steps
            if not isContinue: break
        
        if not isContinue: break

if opt.use_DDP: destroy_process_group()
# https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/164814