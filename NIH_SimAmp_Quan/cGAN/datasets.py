import glob
import random
import os
import librosa
from PIL import Image
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from audio_process import AudioProcessor, AudioProcessorTorch, normalize_wav, change_range
from utils import normalize_spec, denormalize_spec, power_to_db, db_to_power, save_spectrogram

class SpecDataset(Dataset):
    def __init__(
            self,
            noisy_dir,
            gt_dir,
            device,
            mode='db',
            is_normal_spec=True,
            findMinMaxSpec=False,
            use_audio_processor=False,
            n_iters=10,
            hop_length_rate=2
    ):
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.filenames = os.listdir(noisy_dir)
        
        assert len(set(os.listdir(noisy_dir)) - set(os.listdir(gt_dir))) == 0, 'x and y does not match'
        
        self.device = device
        self.mode = mode
        self.is_normal_spec = is_normal_spec
        self.findMinMaxSpec = findMinMaxSpec
        self.use_audio_processor = use_audio_processor
        self.n_iters = n_iters

        self.sample_rate = 16000
        self.n_fft = 1024
        self.n_stft = int((self.n_fft//2) + 1)
        self.win_length = None
        self.hop_length_rate = hop_length_rate
        self.hop_length = self.n_fft // self.hop_length_rate
        self.n_mels = 128
        if use_audio_processor:
            self.max_len = 64000 + 64*40 # to ensure img.shape = (1, 128, 512)
        else:
            self.max_len = 64000 + 64*20 # to ensure img.shape = (1, 128, 128)

        self.spec_min, self.spec_max = self.get_minmax_spec(findMinMaxSpec)

        self.processor = self.get_processor()
        
        # ONLY DO THIS SINCE DATASET IS SMALL
        # self.arrs = [self.__getitem__(i) for i in range(len(self.filenames))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        noisy_file = os.path.join(self.noisy_dir, filename)
        gt_file = os.path.join(self.gt_dir, filename)
                        
        return {
            'noisy': self.getitem_helper(noisy_file),
            'gt': self.getitem_helper(gt_file)
        }
    
    def getitem_helper(self, filename):
        wav, sr = self.read_audio(filename)
        wav = wav.to(self.device)
        wav = self.clean_signal(wav, sr)
        spec, filterbank, scale = self.audio_to_spec(wav)
        return spec, wav.squeeze(0), filterbank, scale

    def read_audio(self, filename):
        wav, sr = torchaudio.load(filename)
        return wav, sr
    
    def save_audio(self, path, waveform):
        assert waveform.ndim == 2, waveform.ndim
        assert waveform.shape[0] == 1, 'only 1 sample, not a batch'
        torchaudio.save(
            filepath=path,
            src=change_range(normalize_wav(waveform)).detach().clone().cpu(), 
            sample_rate=self.sample_rate)
    
    def audio_to_spec(self, wav):
        # wav.shape = [1, num_samples]
        # spec = self.mel_transform(wav)  # melspec.shape = (1, h, w)
        spec, filterbank, scale = self.processor.wav_to_melspec(wav)
        if self.mode == 'db':
            spec = power_to_db(spec)
        if self.is_normal_spec:
            spec, minmaxval = normalize_spec(spec, self.spec_min, self.spec_max)
        return spec, filterbank, scale
    
    def spec_to_audio(self, spec, filter, scale):
        spec = torch.clamp(spec, min=0.0, max=1.0)
        if self.is_normal_spec:
            spec = denormalize_spec(spec, self.spec_min, self.spec_max)
        if self.mode == 'db':
            spec = db_to_power(spec)
        wav = self.processor.melspec_to_wav(spec, filter, scale)
        return wav
    
    def get_processor(self):
        if self.use_audio_processor:
            processor = AudioProcessor(self.n_fft, self.hop_length, self.n_mels, self.sample_rate, self.n_iters)
        else:
            processor = AudioProcessorTorch(
                self.n_fft, self.n_stft, self.n_mels, self.sample_rate,
                self.win_length, self.hop_length, self.device)
        return processor

    def get_minmax_spec(self, findMinMaxSpec=False):
        # min/max val of db melspectrogram on noise_enhance/data/noisy_trainset/*.wav
        if not findMinMaxSpec:
            minval, maxval = -100.0, 50.0
        else:
            maxval, minval = -float('inf'), float('inf')
            for f in tqdm(os.listdir(self.noisy_dir), desc="get_minmax_spectrogram"):
                x, _ = self.getitem_helper(os.path.join(self.noisy_dir, f))[0]
                maxval = max(x.max(), maxval)
                minval = min(x.min(), minval)
        return minval, maxval
    
    def clean_signal(self, wav, sr):
        wav = self.resample_if_necessary(wav, sr)
        wav = self.mix_down_if_necessary(wav)
        wav = self.pad_or_trim(wav)
        return wav
        
    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    
    def resample_if_necessary(self, signal, sr):
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    
    def pad_or_trim(self, signal):
        # cut_if_necessary
        signal = signal[:, :min(signal.shape[1], self.max_len)]  # cut by time axis
        # pad_if_necessary, pad with 0
        num_missing_samples = max(0, self.max_len - signal.shape[1])
        signal = torch.nn.functional.pad(signal, (0, num_missing_samples), value=0)
        return signal

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    dataset = SpecDataset(
        './PD_enhance/data/noisy_trainset',
        './PD_enhance/data/clean_trainset',
        device,
        use_audio_processor=True,
        hop_length_rate=2
    )
    filename = './PD_enhance/data/noisy_testset/PD01_sit10.wav'
    melspec, wav, filer, scale = dataset.getitem_helper(filename)
    print(melspec.shape, wav.shape, filer.shape, scale.shape)
    # ([1, 128, 128]), ([65280]), ([]), ([])
    # ([1, 128, 512]), ([66560]), ([1, 128, 513]), ([1])

    re_wav = dataset.spec_to_audio(melspec, filer, scale)
    print('re_wav.shape', re_wav.shape)
    # ([1, 65024])
    # ([1, 66560])

    save_spectrogram(melspec[0].cpu().numpy(), 'PD01_sit10.png')
    dataset.save_audio('z.wav', wav.unsqueeze(0))
    dataset.save_audio('z_re.wav', re_wav)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in loader:
        break
    melspec, wav, filer, scale = batch['gt']

    print(melspec.shape, wav.shape, filer.shape, scale.shape)
    # ([4, 1, 128, 128]), ([4, 65280]), ([4]), ([4])
    # ([4, 1, 128, 512]), ([4, 66560]), ([4, 1, 128, 513]), ([4, 1])

    re_wav = dataset.spec_to_audio(melspec, filer, scale)
    print('re_wav.shape', re_wav.shape)
    # ([4, 65024])
    # ([4, 66560])

    # dataset.save_audio('z.wav', wav[0:1])
    # dataset.save_audio('z_re.wav', re_wav[0:1])
    
    # sample = dataset[0]
    # noisy, gt = sample['noisy'], sample['gt']
    # melspec, _, _, _ = noisy
    # save_spectrogram(melspec[0].cpu().numpy(), 'z-noisy.png')
    # melspec, _, _, _ = gt
    # save_spectrogram(melspec[0].cpu().numpy(), 'z-gt.png')