import subprocess
import glob
import os
from scipy.io import wavfile
from tqdm import tqdm

def convert_16k(folder):
    wavfiles = glob.glob(f'{folder}/*.wav')
    for filename in tqdm(wavfiles):
        basename = os.path.basename(filename)
        dirname = os.path.dirname(filename)
        save_dir = f'{dirname}_16k'
        subprocess.run(f"mkdir -p {save_dir}", shell=True, check=True)
        try:
            subprocess.run(f"ffmpeg -hide_banner -loglevel fatal -nostats -i {filename} -ar 16k {os.path.join(save_dir, basename)}", shell=True, check=True)
        except Exception as e:
            print(e)
            print(filename)
            break

def check_sr(folder):
    wavfiles = glob.glob(f'{folder}/*.wav')
    for filename in tqdm(wavfiles):
        sr, y = wavfile.read(filename)
        assert sr == 16000, sr

if __name__ == '__main__':
    folder = 'no-noise'
    convert_16k('no-noise')