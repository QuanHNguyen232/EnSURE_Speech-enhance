import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import os
from tqdm import tqdm

def find_formant(filename):
    # Load the audio file
    # filename = 'your_audio_file.wav'  # Replace with your file path
    y, sr = librosa.load(filename, sr=48000)

    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    y = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Frame the signal into short frames
    frame_size = int(sr * 0.025)
    num_frames = len(y) // frame_size
    
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        frames[i, :] = y[i*frame_size : (i+1)*frame_size]

    # Compute the LPC coefficients for each frame
    order = 2 + sr // 1000
    lpc_coeffs = np.zeros((num_frames, order+1))

    for i in range(num_frames):
        lpc_coeffs[i, :] = librosa.lpc(frames[i, :], order) # [num_frames, order+1]

    # Get formant frequencies for each frame
    formants = np.zeros((num_frames, order))  # (128, 50)

    for i in range(num_frames):
        roots = np.roots(lpc_coeffs[i, :])  #  roots represent the resonant frequencies (or formants)
        roots = [root for root in roots if np.imag(root) >= 0 and np.abs(root) > 0.95 and np.abs(root) < 1.05]
        freqs = sorted(np.arctan2(np.imag(roots), np.real(roots)))
        formants[i, :len(freqs)] = [freq * sr / (2 * np.pi) for freq in freqs]

        # formants[i, :] = [freq * sr / (2 * np.pi) for freq in freqs]

    # Average formant frequencies across frames
    mean_formants = np.mean(formants, axis=0)
    return formants, mean_formants

def amplify(filename, mean_formants):
    audio, sr = librosa.load(filename, sr=48000)

    energy_factor = 2
    padding_length = 0.1

    # Calculate the number of samples for padding
    padding_samples = int(padding_length * sr)

    # Apply padding to the audio signal
    padded_audio = np.pad(audio, (padding_samples, padding_samples), mode='constant')
    # padded_audio = audio
    print(audio.shape, np.pad(audio, (padding_samples, padding_samples), mode='constant').shape, padding_samples)
    # Modify the amplitudes around each formant
    for formant in mean_formants:
        if formant !=0:
            # Define the bandpass filter for the formant frequency range
            lowcut = (formant - 100) if (formant-200>0) else formant+100   # Adjust as needed
            highcut = (formant + 100)  # Adjust as needed
            # print (lowcut, highcut)
            order = 4  # Filter order
            b, a = butter(order, [lowcut, highcut], btype='band', fs=sr)

            # Apply the bandpass filter to the audio wave
            filtered_audio = filtfilt(b, a, padded_audio)

            # Remove the padding from the filtered audio
            filtered_audio = filtered_audio[padding_samples:-padding_samples]
            # filtered_audio = filtfilt(b, a, audio)

            # Calculate the amplification factor for the formant
            amplification_factor = energy_factor * 1

            # Scale the amplitudes within the formant frequency range
            filtered_audio *= amplification_factor
            print(audio.shape, filtered_audio.shape)
            # Add the modified formant amplitudes to the original audio
            audio += filtered_audio
            # print (audio)
    save_dir = "amp_formant"
    audio_name = filename.split("/")[-1]
    save_path = os.path.join(save_dir, audio_name)
    # librosa.output.write_wav(save_path, audio, sr)

def amplify_by_frame(filename, formants, verbose=False):
    # formants.shape = (num_frame, formants)
    audio, sr = librosa.load(filename, sr=48000)
    
    # Frame the signal into short frames
    frame_size = int(sr * 0.025)
    if verbose: print('audio len', len(audio), '\t', 'frame_size', frame_size)

    energy_factor = 3
    padding_length = 0.1

    # Calculate the number of samples for padding
    padding_samples = int(padding_length * sr)
    
    # Modify the amplitudes around each formant
    for i, frame in enumerate(formants):
        # slicing
        audio_slice = audio[i*frame_size : (i+1)*frame_size]
        # # Apply padding to the audio signal
        padded_audio = np.pad(audio_slice, (padding_samples, padding_samples), mode='constant')
        if verbose: print('audio_slice.shape', audio_slice.shape, '\t', 'padded_audio.shape', padded_audio.shape)

        # amp by formant
        for formant in frame:
            if formant !=0:
                # Define the bandpass filter for the formant frequency range
                lowcut = (formant - 100) if (formant-200>0) else formant+100   # Adjust as needed
                highcut = (formant + 100)  # Adjust as needed
                if lowcut > highcut or highcut >= sr/2: continue  # must: highcut < fs/2
                
                order = 4  # Filter order
                b, a = butter(order, [lowcut, highcut], btype='band', fs=sr)

                # MUST ADD BY FRAME
                # Apply the bandpass filter to the audio wave
                filtered_audio = filtfilt(b, a, padded_audio)

                # Remove the padding from the filtered audio
                filtered_audio = filtered_audio[padding_samples:-padding_samples]
                # filtered_audio = filtfilt(b, a, audio)

                # Calculate the amplification factor for the formant
                amplification_factor = energy_factor * 1

                # Scale the amplitudes within the formant frequency range
                filtered_audio *= amplification_factor
                if verbose: print('filtered_audio.shape', filtered_audio.shape)
                
                # Add the modified formant amplitudes to the original audio
                audio[i*frame_size : (i+1)*frame_size] += filtered_audio
                
    save_dir = "amp_frame_formant"
    audio_name = filename.split("/")[-1]
    save_path = os.path.join(save_dir, audio_name)
    librosa.output.write_wav(save_path, audio, sr)

def enhance_by_formant(filename):
    formants, mean_formants = find_formant(filename)
    # amplify(filename, mean_formants)
    amplify_by_frame(filename, formants)

def enhance_formant_folder(source_folder):
    import glob
    audios = glob.glob(source_folder)
    for audio in tqdm(audios):
        amp_status = audio.split('_')[2]
        if amp_status == "na":  # We only select the non-amplified audio
            try:
                enhance_by_formant(audio)
            except:
                print(f'error file {audio}')

if __name__=="__main__":
    source_folder = "../data/no-noise/*.wav"
    enhance_formant_folder(source_folder)

    