
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import firwin
import torch.nn as nn
import pandas as pd
import os
from scipy.signal import spectrogram
import matplotlib.pyplot as plt


OUTPUT_FOLDER = ""


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


class Prism_Dataset(Dataset):
    def __init__(self, dataframe, dataframe_emb, unprocessed_wav_path, chunk_size=2048, n_bands=16, allowed_indices=None):
        self.df = dataframe
        self.df_emb = dataframe_emb.copy()  # <--- latent table
        self.unprocessed_audio, self.sr = sf.read(unprocessed_wav_path)
        if self.unprocessed_audio.ndim > 1:
            self.unprocessed_audio = self.unprocessed_audio[:, 0]

        self.chunk_size = chunk_size
        self.n_bands = n_bands

        self.pedals_list = sorted(self.df['effect'].unique().tolist())
        self.pedal2id = {p: i for i, p in enumerate(self.pedals_list)}
        self.gain_list = sorted(self.df['gain'].unique().tolist())
        self.tone_list = sorted(self.df['tone'].unique().tolist())

        if allowed_indices is None:
            raise ValueError("You must provide allowed_indices!")
        self.allowed_indices = np.array([i for i in allowed_indices if i in self.df['index'].unique()])
        if len(self.allowed_indices) == 0:
            raise ValueError("No valid indices!")

        if self.n_bands > 1:
            self.filter = MelBandFilter(n_bands=n_bands, filter_len=513, sr=self.sr)

        # --- preprocess the embedding dataframe for fast lookup ---
        # Convert stringified latent lists to real tensors
        self.df_emb['latents'] = self.df_emb['latents'].apply(
            lambda s: np.array(eval(s), dtype=np.float32) if isinstance(s, str) else np.array(s, dtype=np.float32)
        )

    def __len__(self):
        return len(self.allowed_indices)

    def _get_latent_vector(self, effect, gain, tone):
        """Return latent vector for given (effect, gain, tone)."""
        row = self.df_emb[
            (self.df_emb['label'] == effect) &
            (self.df_emb['g'] == gain) &
            (self.df_emb['t'] == tone)
        ]
        if len(row) == 0:
            raise ValueError(f"No matching latent vector for ({effect}, g={gain}, t={tone})")
        return torch.from_numpy(row.iloc[0]['latents'])


    def __getitem__(self, idx):
        chunk_idx = self.allowed_indices[idx]
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        input_chunk = self.unprocessed_audio[start:end]
        if len(input_chunk) < self.chunk_size:
            input_chunk = np.pad(input_chunk, (0, self.chunk_size - len(input_chunk)), mode='constant')
        input_chunk = torch.from_numpy(input_chunk.astype(np.float32))

        df_chunk = self.df[self.df['index'] == chunk_idx]
        if len(df_chunk) == 0:
            raise ValueError(f"No processed samples for chunk {chunk_idx}")

        # ---------- SINGLE-BAND CASE ----------
        if self.n_bands == 1:
            row = df_chunk.sample(n=1).iloc[0]
            processed_audio = np.load(row['path']).astype(np.float32)
            if len(processed_audio) < self.chunk_size:
                processed_audio = np.pad(processed_audio, (0, self.chunk_size - len(processed_audio)), mode='constant')
            processed_tensor = torch.from_numpy(processed_audio)

            latent_vec = self._get_latent_vector(row['effect'], row['gain'], row['tone']).unsqueeze(0)

            return {
                "input_chunk": input_chunk,
                "conditioning": latent_vec,
                "target_bands": processed_tensor.unsqueeze(0)
            }

        # ---------- MULTI-BAND CASE (UNIQUENESS-BIASED SAMPLING) ----------

        # 1️⃣ Select candidate chunks (global pool)
        candidate_rows = self.df.sample(n=min(self.n_bands * 2, len(self.df)), replace=len(self.df) < self.n_bands * 2)
        candidate_indices = candidate_rows['index'].unique().tolist()

        # 2️⃣ Determine number of unique chunks (k_unique) with exponential bias
        #     - α controls how strongly we prefer unique combinations
        #     - higher α = fewer repetitions
        alpha = 0.4
        probs = np.exp(alpha * np.arange(1, self.n_bands + 1))
        probs /= probs.sum()
        k_unique = np.random.choice(np.arange(1, self.n_bands + 1), p=probs)

        # 3️⃣ Pick k_unique unique chunks from the pool
        unique_chunks = np.random.choice(candidate_indices, size=k_unique, replace=False)

        # 4️⃣ Randomly assign each band one of those unique chunks (with replacement)
        assigned_indices = np.random.choice(unique_chunks, size=self.n_bands, replace=True)

        # 5️⃣ Load each unique chunk only once
        loaded_audio = {}
        loaded_latents = {}

        for cidx in set(assigned_indices):
            row = self.df[self.df['index'] == cidx].sample(n=1).iloc[0]
            processed_audio = np.load(row['path']).astype(np.float32)
            if len(processed_audio) < self.chunk_size:
                processed_audio = np.pad(processed_audio, (0, self.chunk_size - len(processed_audio)), mode='constant')
            processed_tensor = torch.from_numpy(processed_audio).unsqueeze(0).unsqueeze(0)
            loaded_audio[cidx] = processed_tensor
            loaded_latents[cidx] = self._get_latent_vector(row['effect'], row['gain'], row['tone'])

        # 6️⃣ Generate per-band outputs (filter only the required band)
        target_bands = []
        conditioning_list = []

        for band_idx, assigned_chunk_idx in enumerate(assigned_indices):
            proc_audio = loaded_audio[assigned_chunk_idx]
            latent_vec = loaded_latents[assigned_chunk_idx]

            h = self.filter.filters[band_idx].view(1, 1, -1)
            pad = (self.filter.filter_len - 1) // 2
            y_i = F.conv1d(proc_audio, h, padding=pad)
            target_band = y_i.squeeze(0).squeeze(0)[:self.chunk_size]

            target_bands.append(target_band)
            conditioning_list.append(latent_vec)

        target_bands = torch.stack(target_bands, dim=0)
        conditioning = torch.stack(conditioning_list, dim=0)

        return {
            "input_chunk": input_chunk,
            "conditioning": conditioning,
            "target_bands": target_bands
        }




if __name__ == "__main__":
    df = pd.read_csv("/home/ardan/ARDAN/MORPHDRIVE_2/TCN_PRISM/preprocessed_data/tcn_disc_metadata.csv")
    df_emb = pd.read_csv("/home/ardan/ARDAN/MORPHDRIVE_2/TCN_PRISM/vae_output/vae_latents.csv")
    dataset = Prism_Dataset(
        dataframe=df,
        dataframe_emb=df_emb,
        unprocessed_wav_path="/home/ardan/ARDAN/MORPHDRIVE_2/a_0-input.wav",
        chunk_size=2048,
        n_bands=16, 
        allowed_indices=range(0, 500)
    )

    print("Dataset length:", len(dataset))

    sample = dataset[100]
    print("\nSample shapes:")
    print("  input_chunk.shape:", sample['input_chunk'].shape)
    print("  conditioning.shape:", sample['conditioning'].shape)
    print("  target_bands.shape:", sample['target_bands'].shape)
    print("\nConditioning (first 3 bands):")
    print(sample['conditioning'][:10])

    output_folder = OUTPUT_FOLDER
    os.makedirs(output_folder, exist_ok=True)
    sr = dataset.sr

    def save_mel_spaced_spectrogram(signal, sr, title, save_path, n_mels=128):
        nperseg = min(512, len(signal))
        noverlap = nperseg // 4
        
        f, t, Sxx = spectrogram(signal, fs=sr, nperseg=nperseg, noverlap=noverlap)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)

        mel_freqs = librosa.hz_to_mel(f)
        mel_ticks = np.linspace(mel_freqs[0], mel_freqs[-1], num=8)
        hz_ticks = librosa.mel_to_hz(mel_ticks)

        plt.figure(figsize=(10, 4))
        plt.pcolormesh(t, mel_freqs, Sxx_db, shading="gouraud", cmap='viridis')
        plt.title(title, fontsize=12, fontweight='bold')
        plt.ylabel("Mel frequency (mel)")
        plt.xlabel("Time [sec]")
        plt.colorbar(label="Power (dB)")
        plt.yticks(mel_ticks, labels=[f"{hz:.0f}" for hz in hz_ticks])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    bands = sample["target_bands"].numpy()
    band_info = dataset.filter.get_band_info()
    
    for b in range(dataset.n_bands):
        info = band_info[b]
        save_path = os.path.join(output_folder, f"band_{b:02d}.png")
        title = f"Band {b}: {info['low_hz']:.0f}-{info['high_hz']:.0f} Hz"
        save_mel_spaced_spectrogram(bands[b], sr, title, save_path)

    print(f"\nSaved {dataset.n_bands} band spectrograms to: {output_folder}")

    with torch.no_grad():
        reconstructed = dataset.filter.inverse(sample["target_bands"].unsqueeze(0)).squeeze().numpy()
    input_clean = sample["input_chunk"].numpy()

    comparison_folder = os.path.join(output_folder, "comparison")
    os.makedirs(comparison_folder, exist_ok=True)

    save_mel_spaced_spectrogram(input_clean, sr, "Clean Input", 
                                os.path.join(comparison_folder, "input_clean.png"))
    save_mel_spaced_spectrogram(reconstructed, sr, "Reconstructed (Sum of Bands)", 
                                os.path.join(comparison_folder, "reconstructed.png"))

    bands_sum = bands.sum(axis=0)
    save_mel_spaced_spectrogram(bands_sum, sr, "Direct Sum of All Bands", 
                                os.path.join(comparison_folder, "bands_sum.png"))

    print(f"\nSaved comparison spectrograms to: {comparison_folder}")
    
    reconstruction_error = np.abs(reconstructed - input_clean).mean()
    print(f"\nReconstruction Quality:")
    print(f"  Mean absolute error: {reconstruction_error:.6f}")
    print(f"  Signal energy: {np.abs(input_clean).mean():.6f}")
    print(f"  Relative error: {reconstruction_error / (np.abs(input_clean).mean() + 1e-10) * 100:.2f}%")

    print("Input chunk:", sample["input_chunk"].min(), sample["input_chunk"].max())
    print("Target bands min/max:", sample["target_bands"].min(), sample["target_bands"].max())
    print("Sum of bands min/max:", sample["target_bands"].sum(dim=0).min(), sample["target_bands"].sum(dim=0).max())
    recon = dataset.filter.inverse(sample["target_bands"].unsqueeze(0)).squeeze().numpy()
    print("Reconstructed vs input max/min:", recon.max(), recon.min(), sample["input_chunk"].max(), sample["input_chunk"].min())


    
    print(f"\nBand energy distribution:")
    for b in range(dataset.n_bands):
        energy = np.abs(bands[b]).mean()
        info = band_info[b]
        print(f"  Band {b:2d} ({info['low_hz']:6.0f}-{info['high_hz']:6.0f}Hz): {energy:.6f}")


