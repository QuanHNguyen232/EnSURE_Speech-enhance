import librosa
import numpy as np
from scipy.fft import fft, ifft
import os
import glob

def amply_freq(y, sr, fl, fh, factor):
    # Perform the FFT
    Y = fft(y)

    # Convert to absolute value (magnitude) and phase
    Y_mag = np.abs(Y)
    Y_phase = np.angle(Y)

    # Define the frequency band to amplify
    f_low = fl  # Lower frequency limit in Hz
    f_high = fh  # Upper frequency limit in Hz

    # Convert these frequencies to indices
    i_low = int(f_low * len(y) / sr)
    i_high = int(f_high * len(y) / sr)

    # Amplify the frequency band by a certain factor
    amp_factor = factor
    Y_mag[i_low:i_high] *= amp_factor

    # Convert back to complex numbers
    Y_amp = Y_mag * np.exp(1j * Y_phase)

    # Perform the inverse FFT
    y_amp = np.real(ifft(Y_amp))
    return y_amp


def amp_folder(source_folder, fl, fh, factor):
    source_wavefiles = glob.glob(source_folder)
    for wavfile in source_wavefiles:
        # wavfile = "../data/no-noise/sim_NM_amp_nn_oc02_h17_1.wav"  no noise
        amp_status = wavfile.split('_')[2]
        if amp_status == "na":  # We only select the non-amplified audio
            y, sr = librosa.load(wavfile, sr=48000)
            y_amp = amply_freq(y, sr, fl, fh, factor)
            save_dir = str(fl)+"_"+str(fh)+"_"+str(factor)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            audio_name = wavfile.split("/")[-1]
            save_path = os.path.join(save_dir, audio_name)
            librosa.output.write_wav(save_path, y_amp, sr)

if __name__=="__main__":
    source_folder = "../data/no-noise/*.wav"
    fl = 1000
    fh = 8000
    factor = 2
    amp_folder(source_folder, fl, fh, factor)