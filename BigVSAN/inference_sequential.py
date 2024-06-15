# MODIFIED TO DO A SEQUENTIAL GENERATION PROCESS WHERE EACH GROUP OF GENERATED FILES BECOME THE INPUT FOR NEW GENERATION


from __future__ import absolute_import, division, print_function, unicode_literals

import os
import argparse
import json
import torch
from scipy.io.wavfile import write
import librosa
import shutil
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE
from models import BigVSAN as Generator

h = None
device = None
torch.backends.cudnn.benchmark = False

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def get_mel(x, h):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def clean_filename(filename, iteration):
    """
    Cleans the filename by removing any previous iteration suffix before adding the new one.
    """
    if iteration > 1: 
        parts = filename.split('_')
        cleaned_parts = [part for part in parts if not part.startswith('iter') and not part.endswith('generated.wav')]
        return '_'.join(cleaned_parts)
    return os.path.splitext(filename)[0] 

def inference(input_wavs_dir, output_dir, temp_output_dir, checkpoint_file, iteration, h, device):
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)

    with torch.no_grad():
        count = 0
        for filename in os.listdir(input_wavs_dir):
            count +=1
            wav, sr = librosa.load(os.path.join(input_wavs_dir, filename), h.sampling_rate, mono=True)
            wav = torch.FloatTensor(wav).to(device)
            x = get_mel(wav.unsqueeze(0), h)

            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            base_filename = clean_filename(filename, iteration)
            output_filename = f"{base_filename}_iter{iteration}_generated.wav"
            output_file = os.path.join(output_dir, output_filename)
            temp_output_file = os.path.join(temp_output_dir, output_filename)

            write(output_file, h.sampling_rate, audio)
            write(temp_output_file, h.sampling_rate, audio)
            print(count, output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_file', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1)

    a = parser.parse_args()

    config_file = os.path.join(os.path.dirname(a.checkpoint_file), 'config.json')
    with open(config_file) as f:
        data = f.read()
    
    global h
    h = AttrDict(json.loads(data))

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    current_input_dir = a.input_wavs_dir
    for i in range(a.iterations):
        print(f'Iteration {i+1}/{a.iterations}')

        # Temporal directory
        temp_output_dir = os.path.join("temp_dir_for_iteration", f"iteration_{i+1}")
        if os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)  # Limpia el directorio temporal si ya existe
        os.makedirs(temp_output_dir, exist_ok=True)

        inference(current_input_dir, a.output_dir, temp_output_dir, a.checkpoint_file, i + 1, h, device)

        current_input_dir = temp_output_dir

    shutil.rmtree("temp_dir_for_iteration")

if __name__ == '__main__':
    main()


