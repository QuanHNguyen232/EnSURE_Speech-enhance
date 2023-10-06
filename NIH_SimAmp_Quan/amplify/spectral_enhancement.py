import os
os.chdir('../')
import glob
import numpy as np
import librosa
from tqdm import tqdm

def spectral_shaping(speech_file, alpha):
    '''
    Define the spectral_shaping function that applies spectral shaping to enhance the speech signal. It computes the Short-Time Fourier Transform (STFT) magnitude of the input signal, applies spectral shaping by raising the magnitude to the power of alpha, and reconstructs the enhanced signal using inverse STFT (ISTFT).
    TODO:
    - Experiment with different alpha values
    - Combine with other enhancement methods.
    '''
    # Read the audio file
    signal, rate = librosa.load(speech_file, sr=48000)

    # Ensure mono audio
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Compute the Short-Time Fourier Transform (STFT) magnitude
    stft = librosa.stft(signal)
    magnitude = np.abs(stft)

    # Apply spectral shaping
    enhanced_magnitude = magnitude ** alpha

    # Reconstruct the enhanced signal using inverse STFT (ISTFT)
    enhanced_signal = librosa.istft(enhanced_magnitude * np.exp(1j * np.angle(stft)))

    return signal, enhanced_signal, rate

def sub_band_spectral_enhancement(speech_file, num_subbands, alpha):
    ''' Sub-band Spectral Enhancement
    In this code snippet, we define the sub_band_spectral_enhancement function that applies sub-band spectral enhancement to the speech signal. It computes the Short-Time Fourier Transform (STFT) magnitude of the input signal, divides the magnitude spectrogram into sub-bands, applies spectral shaping using the alpha parameter to each sub-band, combines the enhanced sub-bands, and reconstructs the enhanced signal using inverse STFT (ISTFT).
    TODO:
    Experiment with different num_subbands and alpha.
    Args:
        - Number of sub-bands
        - Spectral enhancement factor
    '''
    # Read the audio file
    signal, rate = librosa.load(speech_file, sr=48000)

    # Ensure mono audio
    if signal.ndim > 1:
        signal = signal[:, 0]
    
    # Compute the Short-Time Fourier Transform (STFT) magnitude
    stft = librosa.stft(signal)
    magnitude = np.abs(stft)

    # Divide the magnitude spectrogram into sub-bands
    freq_bins = np.linspace(0, magnitude.shape[0], num_subbands + 1, dtype=int)
    sub_bands = np.split(magnitude, freq_bins[1:-1], axis=0)

    # Apply spectral shaping to each sub-band
    enhanced_sub_bands = []
    for band in sub_bands:
        enhanced_band = band ** alpha
        enhanced_sub_bands.append(enhanced_band)

    # Combine the enhanced sub-bands
    enhanced_magnitude = np.concatenate(enhanced_sub_bands, axis=0)

    # Reconstruct the enhanced signal using inverse STFT (ISTFT)
    enhanced_signal = librosa.istft(enhanced_magnitude * np.exp(1j * np.angle(stft)))

    return signal, enhanced_signal, rate

def nmf_spectral_enhancement(speech_file, num_components, alpha):
    ''' Non-negative Matrix Factorization (NMF)
    In this code snippet, we define the nmf_spectral_enhancement function that applies NMF-based spectral enhancement to the speech signal. It computes the Short-Time Fourier Transform (STFT) magnitude of the input signal, performs NMF decomposition on the magnitude spectrogram, applies spectral shaping using the alpha parameter to the NMF components, reconstructs the enhanced magnitude spectrogram, and finally reconstructs the enhanced signal using inverse STFT (ISTFT).
    TODO:
    Experiment with different num_components and alpha
    Args:
        - Number of NMF components
        - Spectral enhancement factor
    '''
    # Read the audio file
    signal, rate = librosa.load(speech_file, sr=48000)

    # Ensure mono audio
    if signal.ndim > 1:
        signal = signal[:, 0]

    # Compute the Short-Time Fourier Transform (STFT) magnitude
    stft = librosa.stft(signal)
    magnitude = np.abs(stft)

    # Perform NMF on the magnitude spectrogram
    components, activations = librosa.decompose.decompose(magnitude, n_components=num_components, sort=True)

    # Apply spectral shaping to the components
    enhanced_components = components ** alpha

    # Reconstruct the enhanced magnitude spectrogram
    enhanced_magnitude = np.dot(enhanced_components, activations)

    # Reconstruct the enhanced signal using inverse STFT (ISTFT)
    enhanced_signal = librosa.istft(enhanced_magnitude * np.exp(1j * np.angle(stft)))

    return signal, enhanced_signal, rate

if __name__ == '__main__':
    # Use DL conda env
    source_folder = "data/no-noise/*.wav"

    # START EDIT
    # path is 'spectral_shaping' OR 'sub_band' OR 'nmf'
    save_path = 'amplify/nmf'
    alpha = 1.5
    num_subbands = 8
    num_components = 10
    # END EDIT

    param = {'alpha': alpha}
    save_path = save_path + f'_a{alpha}'
        
    if 'spectral_shaping' in save_path:
        enhance_method = spectral_shaping
    elif 'sub_band' in save_path:
        enhance_method = sub_band_spectral_enhancement
        save_path = save_path + f'_band{num_subbands}'
        param['num_subbands'] = num_subbands
    elif 'nmf' in save_path:
        enhance_method = nmf_spectral_enhancement
        save_path = save_path + f'_comp{num_components}'
        param['num_components'] = num_components
    else:
        raise Exception("No method avalaible")

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    audios = glob.glob(source_folder)
    for audio in tqdm(audios):
        amp_status = audio.split('_')[2]
        if amp_status == "na":  # We only select the non-amplified audio
            try:
                param['speech_file'] = audio
                _, enhanced_speech, rate = enhance_method(**param)
                filename = os.path.join(save_path, os.path.basename(audio))
                librosa.output.write_wav(filename, enhanced_speech, rate)
            except:
                print(f'can NOT enhance {audio}')
            
        
    
