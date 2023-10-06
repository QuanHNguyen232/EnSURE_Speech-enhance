import random
import time
import datetime
import sys
import matplotlib.pyplot as plt

from torch.autograd import Variable
import torch
import numpy as np

from torchvision.utils import save_image

from typing import Callable, Optional, Union

def power_to_db(S, ref: Union[float, Callable] = 1.0, amin: float = 1e-10, top_db: Optional[float] = 80.0):
  # https://librosa.org/doc/latest/_modules/librosa/core/spectrum.html#power_to_db
  magnitude = S
  if callable(ref): # User supplied a function to calculate reference power
      ref_value = ref(magnitude)
  else:
      ref_value = torch.tensor(abs(ref))
  amin = torch.tensor(amin)
  log_spec = 10.0 * torch.log10(torch.maximum(amin, magnitude))
  log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))
  if top_db is not None:
    if top_db < 0: raise Exception("top_db must be non-negative")
    log_spec = torch.maximum(log_spec, log_spec.max() - top_db)
  return log_spec
  
def db_to_power(S_db, ref: float = 1.0):
  # https://librosa.org/doc/latest/_modules/librosa/core/spectrum.html#db_to_power
  return ref * torch.pow(torch.tensor(10.), 0.1 * S_db)

def normalize_spec(melspec, minval=None, maxval=None):
  if (minval is None) and (maxval is None):
    maxval, minval = melspec.max(), melspec.min()
  melspec -= minval
  melspec /= (maxval - minval)
  return melspec, (minval, maxval)

def denormalize_spec(melspec, minval, maxval):
  melspec *= (maxval - minval)
  melspec += minval
  return melspec

def save_spectrogram(specgram, outname):
    plt.clf()
    fig, axs = plt.subplots(1, 1)
    fig.frameon = False
    axs.set_axis_off()
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    plt.savefig(outname, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_spectrogram(specgram, outname=None):
    plt.clf()
    fig, axs = plt.subplots(1, 1)
    fig.frameon = False
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    plt.show(block=False)
    
def plot_waveform(waveform, sr, title="Waveform"):
    plt.clf()
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr
    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)