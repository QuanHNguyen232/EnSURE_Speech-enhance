# https://github.com/bkvogel/griffin_lim.git

import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import librosa

class AudioProcessor:
    def __init__(self, n_fft, hop_length, n_mels, sample_rate, n_iters=10):
        self.n_fft = n_fft
        self.hopsamp = hop_length # n_fft // 8
        self.n_mels = n_mels
        self.sr = sample_rate
        self.n_iters = n_iters
    
    def wav_to_melspec(self, wavs):
        # wavs.shape = (wav_lens)
        # mel_spec.shape = (1, h, w)
        # filterbank.shape = (1, h, w)
        # scale.shape = () (scalar)
        batch = wavs.shape[0]
        _, magnitude, scale = compute_STFT(wavs, self.n_fft, self.hopsamp)
        mel_spec, filterbank = compute_melspec(
            magnitude, self.n_fft, self.sr, batch=batch,
                    min_freq_hz=70, max_freq_hz=8000, mel_bin_count=self.n_mels
        )
        return mel_spec, filterbank, scale
    
    def melspec_to_wav(self, mel_spec, filterbank, scale):
        '''
        mel_spec.shape = (N, mel_bin_count, h)
        filterbank.shape = (N, mel_bin_count, w)
        scale.shape = (N)
        '''
        ndim = mel_spec.ndim
        if ndim == 4: # remove extra dim when use batch
            mel_spec = mel_spec.squeeze(1)
            filterbank = filterbank.squeeze(1)
            scale = scale.squeeze(1)
        inv_linear_spec = mel_to_linear_spec(mel_spec, filterbank)
        wav_reconstruct, _ = linspec_to_audio(inv_linear_spec, scale, self.n_fft, self.hopsamp, self.n_iters)
        return wav_reconstruct

class AudioProcessorTorch:
    def __init__(self, n_fft, n_stft, n_mels, sample_rate, win_length, hop_length, device):
        self.n_fft = n_fft
        self.n_stft = n_stft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.device = device

        self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                center=True, normalized=True, onesided=True,
                pad_mode="reflect", norm="slaney", mel_scale="htk",
                power=2.0,).to(self.device)
        self.invers_transform = T.InverseMelScale(
                sample_rate=self.sample_rate,
                n_stft=self.n_stft,
                n_mels=self.n_mels,
                mel_scale='htk',max_iter=1000).to(self.device)
        self.grifflim_transform = T.GriffinLim(
                n_fft=self.n_fft,
                win_length=self.win_length,
                power=2.0).to(self.device)
    
    def wav_to_melspec(self, wavs):
        specs = self.mel_transform(wavs)
        return specs, torch.as_tensor(0), torch.as_tensor(0)  # to match with AudioProcessor output
    
    def melspec_to_wav(self, spec, empty1, empty2):
        spec = self.invers_transform(spec)
        wavs = self.grifflim_transform(spec)
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        return wavs
#########################################################
#########################################################
###################### MAIN FUNCS #######################
#########################################################
#########################################################
def compute_STFT(wavs, fft_size, hopsamp):
    '''
    wav.shape = (N, wav_len)
    return: stft.shape = (N, h, w), magnitude.shape = (N, h, w), scale.shape = (N)
    '''
    batch, wav_lens = wavs.shape
    stft = stft_for_reconstruction(wavs, fft_size, hopsamp) # shape = (N, h, w)
    magnitude = torch.abs(stft)**2.0 # shape = (N, h, w)
    scale = 1.0 / torch.max(magnitude.view(magnitude.shape[0], -1), dim=1).values # shape = (N)
    # Rescale to put all values in the range [0, 1].
    for b in range(batch):
        magnitude[b] *= scale[b] # shape = (N, h, w)
    return stft, magnitude, scale

def compute_melspec(magnitude, fft_size, sample_rate_hzs, batch, min_freq_hz=70, max_freq_hz=8000, mel_bin_count=128):
    '''
    magnitude.shape = (N, h, w)
    return: mel_spec.shape = (N, mel_bin_count, h), filterbank.shape = (N, mel_bin_count, w)
    '''
    batch, _, _ = magnitude.shape
    device = magnitude.device
    linear_bin_count = 1 + fft_size//2
    filterbank = make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count, linear_bin_count , sample_rate_hzs, batch).to(device)
    mel_spec = torch.matmul(filterbank, magnitude.permute(0, 2, 1).to(torch.float32))
    return mel_spec, filterbank

def mel_to_linear_spec(mel_spec, filterbank):
    '''
    mel_spec.shape = (N, mel_bin_count, h), filterbank.shape = (N, mel_bin_count, w)
    return: inv_linear_spec.shape = (N, w, h)
    '''
    inv_linear_spec = torch.matmul(filterbank.permute(0, 2, 1), mel_spec)
    return inv_linear_spec

def linspec_to_audio(inv_linear_spec, scale, fft_size, hopsamp, iterations):
    '''
    inv_linear_spec.shape = (N, w, h); scale.shape = (N); fft_size, hopsamp, iterations are scalars
    return: wav_reconstruct.shape = (N, wav_lens), stft_modified.shape = (N, h, w)
    '''
    batch, _, _ = inv_linear_spec.shape
    device = inv_linear_spec.device
    stft_modified = inv_linear_spec.permute(0, 2, 1) # shape = (N, h, w)
    stft_modified_scaled = torch.zeros_like(stft_modified, device=device)

    for b in range(batch):
        stft_modified_scaled[b] = torch.pow(stft_modified[b] / scale[b], 0.5)

    wav_reconstruct = reconstruct_signal_griffin_lim(stft_modified_scaled, fft_size, hopsamp, iterations)
    wav_reconstruct = change_range(normalize_wav(wav_reconstruct))
    return wav_reconstruct, stft_modified

#########################################################
#########################################################
##################### SUPPORT FUNCS #####################
#########################################################
#########################################################
def normalize_wav(x):
    # x.shape = (N, wav_lens)
    mi, ma = torch.min(x, dim=1, keepdim=True).values, torch.max(x, dim=1, keepdim=True).values
    x = x - mi
    x = x / (ma - mi)
    return x

def change_range(x):
    # x.shape = (N, wav_lens)
    # change range from 0 -> 1 to -1 -> 1
    x = x * 2.0
    x = x - 1.0
    return x

def hz_to_mel(f_hz):
    return 2595*torch.log10(torch.as_tensor(1.0 + f_hz/700.0))

def mel_to_hz(m_mel):
    return 700*(10**(m_mel/2595.0) - 1.0)

def fft_bin_to_hz(n_bin, sample_rate_hz, fft_size):
    n_bin = float(n_bin)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    return n_bin*sample_rate_hz/(2.0*fft_size)

def hz_to_fft_bin(f_hz, sample_rate_hz, fft_size):
    f_hz = float(f_hz)
    sample_rate_hz = float(sample_rate_hz)
    fft_size = float(fft_size)
    fft_bin = round(f_hz*2.0*fft_size/sample_rate_hz)
    if fft_bin >= fft_size:
        fft_bin = fft_size-1
    return fft_bin

def stft_for_reconstruction(x, fft_size=1024, hopsamp=1024//8):
    '''
    x.shape = [batch, wav_lens]
    fft_size, hopsamp: scalars
    return: (batch, h, w)
    '''
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    batch, wav_len = x.shape
    device = x.device
    windows = torch.stack([torch.hann_window(fft_size, device=device) for _ in range(batch)], dim=0)
    arr = torch.stack([torch.fft.rfft(windows*x[:, i:i+fft_size]) for i in range(0, wav_len-fft_size, hopsamp)], dim=1)
    return arr

def istft_for_reconstruction(X, fft_size=1024, hopsamp=1024//8):
    '''
    x.shape = [batch, h, w]
    fft_size, hopsamp: scalars
    return: (batch, wav_lens)
    '''
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    batch, time_slices, _ = X.shape
    device = X.device
    windows = torch.stack([torch.hann_window(fft_size, device=device) for _ in range(batch)], dim=0)

    len_samples = int(time_slices*hopsamp + fft_size)
    x = torch.zeros(batch, len_samples, device=device)
    for n,i in enumerate(range(0, len_samples-fft_size, hopsamp)):
        x[:, i:i+fft_size] += windows * torch.real(torch.fft.irfft(X[:, n]))
    return x

def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    '''
    magnitude_spectrogram.shape = [batch, h, w]
    fft_size, hopsamp, iterations: scalars
    return (batch, wav_lens)
    '''
    batch, time_slices, _ = magnitude_spectrogram.shape
    device = magnitude_spectrogram.device
    len_samples = int(time_slices*hopsamp + fft_size)
    x_reconstruct = torch.rand(batch, len_samples, device=device)
    for _ in range(iterations):
      reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
      reconstruction_angle = torch.angle(reconstruction_spectrogram)
      proposal_spectrogram = magnitude_spectrogram*torch.exp(1.0j*reconstruction_angle)
      x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
    return x_reconstruct

def make_mel_filterbank(min_freq_hz, max_freq_hz, mel_bin_count, linear_bin_count, sample_rate_hz, batch):
    '''
    min_freq_hz, max_freq_hz, mel_bin_count, linear_bin_count: scalars
    sample_rate_hzs: [list, Tensor] of all samples in batch
    '''
    # batch = len(sample_rate_hzs)
    min_mels = hz_to_mel(min_freq_hz)
    max_mels = hz_to_mel(max_freq_hz)
    # Create mel_bin_count linearly spaced values between these extreme mel values.
    mel_lin_spaced = torch.tensor(np.linspace(min_mels, max_mels, num=mel_bin_count))
    # Map each of these mel values back into linear frequency (Hz).
    center_frequencies_hz = torch.tensor([mel_to_hz(n) for n in mel_lin_spaced])

    mels_per_bin = float(max_mels - min_mels)/float(mel_bin_count - 1)
    mels_start = min_mels - mels_per_bin
    hz_start = mel_to_hz(mels_start)
    fft_bin_start = torch.tensor([hz_to_fft_bin(hz_start, sample_rate_hz, linear_bin_count)
                              for _ in range(batch)])

    mels_end = max_mels + mels_per_bin
    hz_stop = mel_to_hz(mels_end)

    fft_bin_stop = torch.tensor([hz_to_fft_bin(hz_stop, sample_rate_hz, linear_bin_count)
                            for _ in range(batch)])

    # Map each center frequency to the closest fft bin index.
    linear_bin_indices = torch.stack(
        [torch.tensor([hz_to_fft_bin(f_hz, sample_rate_hz, linear_bin_count) for f_hz in center_frequencies_hz])
        for _ in range(batch)], dim=0)
    # Create filterbank matrix.
    filterbank = torch.zeros((batch, mel_bin_count, linear_bin_count))

    for b in range(batch):
        for mel_bin in range(mel_bin_count):
            center_freq_linear_bin = linear_bin_indices[b, mel_bin]
            center_freq_linear_bin = int(center_freq_linear_bin)
            if center_freq_linear_bin > 1:
                # It is possible to create the left triangular filter.
                if mel_bin == 0:
                    # Since this is the first center frequency, the left side must start ramping up from linear bin 0 or 1 mel bin before the center freq.
                    left_bin = max(0, fft_bin_start[b])
                else:
                    # Start ramping up from the previous center frequency bin.
                    left_bin = linear_bin_indices[b, mel_bin - 1]
                left_bin = int(left_bin)
                for f_bin in range(int(left_bin), int(center_freq_linear_bin)+1):
                    if (center_freq_linear_bin - left_bin) > 0:
                        response = float(f_bin - left_bin)/float(center_freq_linear_bin - left_bin)
                        filterbank[b, mel_bin, f_bin] = response
            # Create the right side of the triangular filter that ramps down from 1 to 0.
            if center_freq_linear_bin < linear_bin_count-2:
                # It is possible to create the right triangular filter.
                if mel_bin == mel_bin_count - 1:
                    # Since this is the last mel bin, we must ramp down to response of 0 at the last linear freq bin.
                    right_bin = min(linear_bin_count - 1, fft_bin_stop[b])
                else:
                    right_bin = linear_bin_indices[b, mel_bin + 1]
                right_bin = int(right_bin)
                for f_bin in range(center_freq_linear_bin, right_bin+1):
                    if (right_bin - center_freq_linear_bin) > 0:
                        response = float(right_bin - f_bin)/float(right_bin - center_freq_linear_bin)
                        filterbank[b, mel_bin, f_bin] = response
            filterbank[b, mel_bin, center_freq_linear_bin] = 1.0
    return filterbank

if __name__ == '__main__':
    wav, sr = torchaudio.load('./z.wav')

    n_fft = 1024
    n_stft = int((n_fft//2) + 1)
    hop_length = hopsamp = n_fft // 8 # n_fft // 8
    n_mels = 128
    n_iters = 10
    sample_rate = sr

    processor = AudioProcessor(n_fft, hop_length, n_mels, sample_rate, n_iters)
    spec, filterbank, scale = processor.wav_to_melspec(wav)
    print('spec.shape', spec.shape)
    wav_re = processor.melspec_to_wav(spec, filterbank, scale)
    torchaudio.save('z_test_org.wav', wav_re, sample_rate=16000)
    print('done org')

    # new method
    batch = 1
    stft, magnitude, scale = compute_STFT(wav, n_fft, hopsamp)
    mel_spec, filterbank = compute_melspec(
        magnitude, n_fft, sr, batch=batch,
                min_freq_hz=70, max_freq_hz=8000, mel_bin_count=n_mels
    )
    
    ndim = mel_spec.ndim
    if ndim == 4: # remove extra dim when use batch
        mel_spec = mel_spec.squeeze(1)
        filterbank = filterbank.squeeze(1)
        scale = scale.squeeze(1)
    # inv_linear_spec = mel_to_linear_spec(mel_spec, filterbank)

    def istft(mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix)
    def spec2wav(spectrogram, phase):
        S = spectrogram
        return istft(S, phase)

    # wav_reconstruct, _ = linspec_to_audio(inv_linear_spec, scale, n_fft, hopsamp, n_iters)
    S_inv = librosa.feature.inverse.mel_to_stft(mel_spec.numpy(), sr=sr, n_fft=n_fft)
    S_inv = np.transpose(S_inv, (0, 2, 1))
    purified = spec2wav(S_inv, stft.numpy())
   

    torchaudio.save('z_test_lbs.wav', torch.from_numpy(purified), sample_rate=16000)