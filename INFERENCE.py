import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import firwin
import librosa
import librosa.display
import os
from pathlib import Path

from PRISM_LITE2 import PrismLite


MODEL_PATH = "./experiment_outputs_EMB_LITE_128/model_4bands.pth"
EMB_CSV_PATH = ""
SWEEP_METADATA_CSV = ""
INPUT_AUDIO = ""
OUTPUT_DIR = "./inference_outputs"

band_settings = [
    {'pedal': 'honeybee', 'gain': 5, 'tone': 5},
    {'pedal': 'zendrive', 'gain': 0, 'tone': 1},
    {'pedal': 'honeybee', 'gain': 5, 'tone': 5},
    {'pedal': 'zendrive', 'gain': 0, 'tone': 0},
]


pastel_cmap = LinearSegmentedColormap.from_list("pastel_blue_pink", [(0,"#FFFFFF"), (0.4,"#94D6F8"), (1,"#FA8583")])
plt.register_cmap(name="pastel_blue_pink", cmap=pastel_cmap)


class MelBandFilter(nn.Module):
    def __init__(self, n_bands, filter_len=513, sr=48000, min_freq=40.0, max_freq=None):
        super().__init__()
        self.n_bands = n_bands
        self.filter_len = filter_len
        self.sr = sr

        if max_freq is None:
            max_freq = (sr / 2.0) - 4000.0

        mel_points = np.linspace(librosa.hz_to_mel(min_freq), librosa.hz_to_mel(max_freq), n_bands + 1)
        hz_points = librosa.mel_to_hz(mel_points)
        self.band_edges = [(hz_points[i], hz_points[i + 1]) for i in range(n_bands)]

        filters = []
        for low, high in self.band_edges:
            bw = high - low
            low_eff = max(0.0, low - 0.05 * bw)
            high_eff = min(sr / 2.0, high + 0.05 * bw)
            if low_eff <= 0:
                h = firwin(filter_len, high_eff, pass_zero=True, fs=sr, window='hamming')
            else:
                h = firwin(filter_len, [low_eff, high_eff], pass_zero=False, fs=sr, window='hamming')
            filters.append(h)
        filters = np.stack(filters)
        self.register_buffer("filters", torch.from_numpy(filters).float())

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, _, T = x.shape
        pad = (self.filter_len - 1) // 2
        y = []
        for i in range(self.n_bands):
            h = self.filters[i].view(1, 1, -1)
            y_i = F.conv1d(x, h, padding=pad)
            y.append(y_i)
        y = torch.cat(y, dim=1)
        return y[..., :T]

    def inverse(self, x):
        return x.sum(dim=1, keepdim=True)
    
    def get_band_info(self):
        info = []
        for i, (low, high) in enumerate(self.band_edges):
            center = (low + high) / 2
            info.append({
                'band': i,
                'low_hz': low,
                'high_hz': high,
                'center_hz': center,
                'bandwidth_hz': high - low
            })
        return info


class PrismInference:
    def __init__(self, model_path, emb_csv_path, sweep_metadata_csv, device='cuda'):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=device)
        self.n_bands = checkpoint['n_bands']
        
        self.model = PrismLite(
            inp_channel=1,
            out_channel=1,
            channels=64,
            kernel_size=3,
            n_blocks=9,
            cond_dim=128,
            sample_rate=48000,
            n_bands=self.n_bands,
        ).to(device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.df_emb = pd.read_csv(emb_csv_path)
        self.df_emb['latents'] = self.df_emb['latents'].apply(
            lambda s: np.array(eval(s), dtype=np.float32) if isinstance(s, str) else np.array(s, dtype=np.float32)
        )
        
        self.df_sweeps = pd.read_csv(sweep_metadata_csv)
        
        self.filterbank = MelBandFilter(n_bands=self.n_bands).to(device)
        
        print(f"Model loaded: {self.n_bands} bands")
        print(f"Parameters: {self.model.compute_num_of_params():,}")
        
    def get_latent_vector(self, pedal, gain, tone):
        row = self.df_emb[
            (self.df_emb['label'] == pedal) &
            (self.df_emb['g'] == gain) &
            (self.df_emb['t'] == tone)
        ]
        if len(row) == 0:
            raise ValueError(f"No latent found for pedal={pedal}, gain={gain}, tone={tone}")
        return torch.from_numpy(row.iloc[0]['latents'])
    
    def get_sweep_path(self, pedal, gain, tone):
        row = self.df_sweeps[
            (self.df_sweeps['label'] == pedal) &
            (self.df_sweeps['g'] == gain) &
            (self.df_sweeps['t'] == tone)
        ]
        if len(row) == 0:
            raise ValueError(f"No sweep found for pedal={pedal}, gain={gain}, tone={tone}")
        return row.iloc[0]['sweep_path']
    
    def compose_ground_truth(self, band_settings, length_samples):
        ground_truth_bands = []
        
        print("Loading ground truth sweeps for each band...")
        for i, setting in enumerate(band_settings):
            sweep_path = self.get_sweep_path(
                setting['pedal'],
                setting['gain'],
                setting['tone']
            )
            
            sweep_audio = np.load(sweep_path).astype(np.float32)
            original_sr = 32000
            target_sr = 48000
            
            print(f"  Band {i+1}: {setting['pedal']} (g={setting['gain']}, t={setting['tone']})")
            print(f"    Original length: {len(sweep_audio)} samples @ {original_sr}Hz")
            
            sweep_audio_resampled = librosa.resample(sweep_audio, orig_sr=original_sr, target_sr=target_sr)
            print(f"    Resampled length: {len(sweep_audio_resampled)} samples @ {target_sr}Hz")
            
            if len(sweep_audio_resampled) < length_samples:
                sweep_audio_resampled = np.pad(sweep_audio_resampled, (0, length_samples - len(sweep_audio_resampled)), mode='constant')
            else:
                sweep_audio_resampled = sweep_audio_resampled[:length_samples]
            
            sweep_tensor = torch.from_numpy(sweep_audio_resampled).to(self.device)
            sweep_bands = self.filterbank(sweep_tensor.unsqueeze(0).unsqueeze(1))
            
            ground_truth_bands.append(sweep_bands[:, i:i+1, :])
        
        ground_truth_bands = torch.cat(ground_truth_bands, dim=1)
        ground_truth_audio = self.filterbank.inverse(ground_truth_bands)
        
        return ground_truth_audio.squeeze().cpu().numpy()
    
    def prepare_conditioning(self, band_settings):
        if len(band_settings) != self.n_bands:
            raise ValueError(f"Expected {self.n_bands} band settings, got {len(band_settings)}")
        
        conditioning_list = []
        for setting in band_settings:
            latent = self.get_latent_vector(
                setting['pedal'],
                setting['gain'],
                setting['tone']
            )
            conditioning_list.append(latent)
        
        conditioning = torch.stack(conditioning_list, dim=0).unsqueeze(0).to(self.device)
        return conditioning
    
    def process_audio(self, audio_path, band_settings, chunk_size=2048, overlap=0.5):
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        
        print(f"Input audio: {len(audio)} samples @ {sr}Hz")
        
        if sr != 48000:
            print(f"Resampling input from {sr}Hz to 48000Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
            print(f"Resampled audio: {len(audio)} samples @ {sr}Hz")
        
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        audio_tensor = audio_tensor[24000:-10000]
        print(f"Audio after trimming: {len(audio_tensor)} samples")

        input_audio = self.filterbank(audio_tensor.unsqueeze(0).unsqueeze(1))
        input_audio = self.filterbank.inverse(input_audio).squeeze().cpu().numpy()
        
        ground_truth_audio = self.compose_ground_truth(band_settings, len(audio_tensor))
        
        conditioning = self.prepare_conditioning(band_settings)
        
        total_samples = len(audio_tensor)
        hop_size = int(chunk_size * (1 - overlap))
        num_chunks = (total_samples - chunk_size) // hop_size + 1
        if total_samples > chunk_size:
            num_chunks = max(1, num_chunks)
        else:
            num_chunks = 1
        
        output_audio = torch.zeros(total_samples, device=self.device)
        overlap_count = torch.zeros(total_samples, device=self.device)
        
        hidden_state = None
        
        print(f"Processing {num_chunks} chunks with {overlap*100:.0f}% overlap...")
        
        window = torch.hann_window(chunk_size, device=self.device)
        
        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * hop_size
                end_idx = min(start_idx + chunk_size, total_samples)
                
                chunk = audio_tensor[start_idx:end_idx]
                current_chunk_size = len(chunk)
                
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                
                chunk = chunk.unsqueeze(0).unsqueeze(-1)
                
                if hidden_state is not None:
                    pred_bands, hidden_state = self.model(chunk, conditioning, hidden_state)
                else:
                    pred_bands, hidden_state = self.model(chunk, conditioning)
                
                pred_bands = pred_bands.permute(0, 2, 1)
                pred_bands = self.filterbank(pred_bands)
                
                pred_audio = self.filterbank.inverse(pred_bands)
                pred_audio = pred_audio.squeeze()[:current_chunk_size]
                
                current_window = window[:current_chunk_size]
                
                output_audio[start_idx:end_idx] += pred_audio * current_window
                overlap_count[start_idx:end_idx] += current_window
        
        overlap_count = torch.clamp(overlap_count, min=1e-8)
        output_audio = output_audio / overlap_count
        
        output_audio = output_audio.cpu().numpy()
        
        return output_audio, sr, input_audio, ground_truth_audio
    
    def save_audio(self, audio, sr, output_path):
        audio = np.clip(audio * 3, -1.0, 1.0)
        sf.write(output_path, audio, sr)
        print(f"Audio saved to: {output_path}")
    
    def plot_spectrogram(self, audio, sr, output_path=None):
        plt.figure(figsize=(14, 8))
        
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        
        img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='pastel_blue_pink')
        plt.colorbar(img, format='%+2.0f dB')
        
        band_info = self.filterbank.get_band_info()
        for i, info in enumerate(band_info):
            if i < len(band_info) - 1:
                freq_hz = info['high_hz']
                plt.axhline(y=freq_hz, color='black', linestyle='dashdot', linewidth=4, alpha=0.7)
                
                '''mid_freq = (info['high_hz'] + band_info[i+1]['low_hz']) / 2
                plt.text(plt.xlim()[1] * 0.02, freq_hz, 
                        f"Band {i+1}|{i+2}", 
                        color='black', fontsize=25, va='bottom', 
                        bbox=dict(boxstyle='round, pad=0.1', facecolor='white', alpha=0.5))'''
        
        plt.title('Mel Spectrogram with Band Divisions')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Spectrogram saved to: {output_path}")
        
        plt.close()


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("PRISM Inference")
    print("="*80)
    print(f"\nModel: {MODEL_PATH}")
    print(f"Input: {INPUT_AUDIO}")
    print(f"\nBand Settings:")
    for i, setting in enumerate(band_settings):
        print(f"  Band {i+1}: {setting['pedal']} (gain={setting['gain']}, tone={setting['tone']})")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    inferencer = PrismInference(MODEL_PATH, EMB_CSV_PATH, SWEEP_METADATA_CSV, device=device)
    
    output_audio, sr, input_audio, ground_truth_audio = inferencer.process_audio(INPUT_AUDIO, band_settings)
    
    output_name = Path(INPUT_AUDIO).stem
    audio_out_path = os.path.join(OUTPUT_DIR, f"{output_name}_processed.wav")
    audio_gt_path = os.path.join(OUTPUT_DIR, f"{output_name}_groundtruth.wav")
    spec_out_path = os.path.join(OUTPUT_DIR, f"{output_name}_processed_spectrogram.png")
    spec_input_path = os.path.join(OUTPUT_DIR, f"{output_name}_input_spectrogram.png")
    spec_gt_path = os.path.join(OUTPUT_DIR, f"{output_name}_groundtruth_spectrogram.png")
    
    inferencer.save_audio(output_audio, sr, audio_out_path)
    inferencer.save_audio(ground_truth_audio, sr, audio_gt_path)
    
    print("\nGenerating spectrograms...")
    print("  1. Input spectrogram...")
    inferencer.plot_spectrogram(input_audio, sr, spec_input_path)
    print("  2. Ground truth spectrogram...")
    inferencer.plot_spectrogram(ground_truth_audio, sr, spec_gt_path)
    print("  3. Processed spectrogram...")
    inferencer.plot_spectrogram(output_audio, sr, spec_out_path)
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()