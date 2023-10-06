import argparse
import os
import numpy as np
import math
import shutil
from tqdm import tqdm
from datetime import datetime

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

from models import ResnetGenerator, Discriminator, UnetGenerator, ContextEncoder
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
parser.add_argument("--my_id", type=str, default='pix_ctx-loss_1')
parser.add_argument("--generator_type", type=str, default='resnet')
parser.add_argument("--dir", type=str, default='./PD_enhance')
parser.add_argument("--infer_dir", type=str, default='data/noisy_testset')
parser.add_argument("--pretrain_dir", type=str, default='./PD_enhance/cgan_resnet_pix_ctx-loss_1_ckpt')
parser.add_argument("--use_context_embed", action='store_true', default=False)
parser.add_argument("--use_context_loss", action='store_true', default=True)
parser.add_argument("--use_audio_processor", action='store_true', default=True) # this is very slow
parser.add_argument("--hop_length_rate", type=int, default=4, help="rate which hop_length smaller than n_fft, if =2, spec.shape=(n_mels, 128), =8, shape=(n_mels, 512), better sound")
parser.add_argument("--use_DDP", action='store_true', default=False) # somehow it appears error after a while
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument('--verbose_interval', type=int, default=1, help='interval between printing loss')
parser.add_argument("--num_residual_blocks", type=int, default=8, help="num_residual_blocks if use cycleGAN generator")
opt = parser.parse_args()

dir = opt.dir
save_dir = os.path.join(dir, f'cgan_{opt.generator_type}_infer')
if opt.my_id is not None:
    save_dir = f'{save_dir}_{opt.my_id}'
infer_dir = os.path.join(dir, opt.infer_dir)
tmp_noisyset = os.path.join(dir, 'data/noisy_testset')
tmp_cleanset = os.path.join(dir, 'data/clean_testset')
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

# Print configs
if master_process:
    print (f'Current time: {datetime.now().strftime("DATE %d/%m/%Y - TIME: %H:%M:%S")}')
    print('\n\t'.join(['Config:']+[f'{k} = {v}' for k, v in vars(opt).items()]))
    print(f'Find {torch.cuda.device_count()} gpus available')
    # assert opt.pretrain_dir is not None
    os.makedirs(save_dir, exist_ok=True)
    print(f'{dir}\n\t{save_dir}\n\t{infer_dir}')
    for f in os.listdir(save_dir):
        try:
            os.remove(os.path.join(save_dir, f))
        except:
            pass

# Configure data loader
dataset = SpecDataset(noisy_dir=tmp_noisyset,
                      gt_dir=tmp_cleanset,
                      device=device,
                      use_audio_processor=True,
                      n_iters=100,
                      hop_length_rate=opt.hop_length_rate)
if master_process: print('CREATED Dataset')

# Compute input_nc
input_nc = 1
if opt.use_context_embed:
    enc_h, enc_w = ContextEncoder.output_size # (203, 1024)
    spec, _, _, _ = dataset[0]['gt']
    inp_h, inp_w = spec.shape[-2:]
    add_input_nc = int((enc_h*enc_w) // (inp_h*inp_w))
    input_nc += add_input_nc

# Initialize generator and discriminator
if opt.use_context_embed or opt.use_context_loss:
    # Must keep some weights.requires_grad=True to have its output as gradient
    encoder = ContextEncoder(device).to(device)
    if opt.use_DDP: encoder = DDP(encoder, device_ids=[ddp_local_rank], output_device=ddp_local_rank)
    if master_process: print('CREATED ContextEncoder')
if opt.generator_type == 'unet': # not check if it runs
    generator = UnetGenerator(input_nc=input_nc, output_nc=1, num_downs=7, use_dropout=True).to(device)
    if master_process: print('CREATED UnetGenerator')
else:
    if master_process: print('input num channels = ', input_nc)
    generator = ResnetGenerator(
        input_nc=input_nc,
        output_nc=1,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        n_blocks=opt.num_residual_blocks,
        padding_type='reflect'
    ).to(device)
    if master_process: print('CREATED ResnetGenerator')

if opt.pretrain_dir is not None:
    try:
        generator.load_state_dict(torch.load(os.path.join(opt.pretrain_dir, 'G_model.pt')))
        if master_process: print('Loaded model WEIGHTS')
    except Exception as e:
        if master_process: print(e, '\nCannot load model')
if opt.use_DDP:
    generator = DDP(generator, device_ids=[ddp_local_rank], output_device=ddp_local_rank)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

exit()

# ----------
#  Inference
# ----------
generator.eval()
if opt.use_context_loss or opt.use_context_embed: encoder.eval()
with torch.no_grad():
    files = os.listdir(infer_dir)
    if master_process:
        files = tqdm(files, desc=f"Infer {opt.my_id}:", position=0, leave=True)
    else: files = files
    for f in files:
        tmp_dir = os.path.join(save_dir, f.replace('.wav', ''))
        os.makedirs(tmp_dir, exist_ok=True)
        
        infer_file = os.path.join(infer_dir, f)
        infer_file_new_loc = os.path.join(tmp_dir, f)
        shutil.copy(infer_file, infer_file_new_loc)

        save_wav = os.path.join(tmp_dir, f.replace('.wav', '_clean.wav'))
        save_spec = save_wav.replace('.wav', '.png')
        
        # print('filename: ', f)
        # print('infer_file        ', infer_file)
        # print('infer_file_new_loc', infer_file_new_loc)
        # print('\tinfer_dir:', infer_dir, '\n\tinfer_file:', infer_file)
        # print('\ttmp_dir:  ', tmp_dir)
        # print('\tsave_wav: ', save_wav, '\n\tsave_spec:', save_spec)
        
        melspec, wav, filterbank, scale = dataset.getitem_helper(infer_file)
        # ([1, 128, 128]), ([66560]), ([1, 128, 513]), ([1])
        melspec, wav = melspec.unsqueeze(0).to(device), wav.unsqueeze(0).to(device)
        # ([1, 1, 128, 128]), ([1, 66560])
        filterbank, scale = filterbank.unsqueeze(0).to(device), scale.unsqueeze(0).to(device)
        # ([1, 1, 128, 513]), ([1, 1])
        
        emb = encoder(wav) if opt.use_context_embed else None
        gen_img = generator(melspec, emb)
        # ([1, 1, 128, 128])
        
        gen_wav = dataset.spec_to_audio(gen_img, filterbank, scale)
        # ([1, 66560])
        gen_melspec, _, _ = dataset.audio_to_spec(gen_wav)
        # ([1, 128, 128])
        
        # save audio
        gen_wav = gen_wav[0:1].detach().clone().cpu()
        dataset.save_audio(save_wav, gen_wav)
        # save melspec
        gen_melspec = gen_melspec[0].detach().clone().cpu()
        save_spectrogram(gen_melspec.numpy(), save_spec)
        


if opt.use_DDP: destroy_process_group()
