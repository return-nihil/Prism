import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import firwin
import torch.nn as nn


class MelBandFilter(nn.Module):
    def __init__(self, n_bands, filter_len=513, sr=48000, min_freq=40.0, max_freq=None):
        super().__init__()
        self.n_bands = n_bands
        self.filter_len = filter_len
        self.sr = sr
        
        if max_freq is None:
            max_freq = (sr / 2.0) - 4000.0
        
        self.band_edges = self._compute_band_edges(n_bands, min_freq, max_freq)
        filters = self._create_filters()
        self.register_buffer("filters", torch.from_numpy(filters).float())
    
    def _compute_band_edges(self, n_bands, min_freq, max_freq):
        mel_points = np.linspace(
            librosa.hz_to_mel(min_freq),
            librosa.hz_to_mel(max_freq),
            n_bands + 1
        )
        hz_points = librosa.mel_to_hz(mel_points)
        return [(hz_points[i], hz_points[i + 1]) for i in range(n_bands)]
    
    def _create_filters(self):
        filters = []
        for low, high in self.band_edges:
            bw = high - low
            low_eff = max(0.0, low - 0.05 * bw)
            high_eff = min(self.sr / 2.0, high + 0.05 * bw)
            
            if low_eff <= 0:
                h = firwin(
                    self.filter_len, high_eff,
                    pass_zero=True, fs=self.sr, window='hamming'
                )
            else:
                h = firwin(
                    self.filter_len, [low_eff, high_eff],
                    pass_zero=False, fs=self.sr, window='hamming'
                )
            filters.append(h)
        
        return np.stack(filters)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, _, T = x.shape
        pad = (self.filter_len - 1) // 2
        
        band_outputs = []
        for i in range(self.n_bands):
            h = self.filters[i].view(1, 1, -1)
            band_output = F.conv1d(x, h, padding=pad)
            band_outputs.append(band_output)
        
        result = torch.cat(band_outputs, dim=1)
        return result[..., :T]
    
    def inverse(self, x):
        return x.sum(dim=1, keepdim=True)
    
    def get_band_info(self):
        return [
            {
                'band': i,
                'low_hz': low,
                'high_hz': high,
                'center_hz': (low + high) / 2,
                'bandwidth_hz': high - low
            }
            for i, (low, high) in enumerate(self.band_edges)
        ]


class Prism_Dataset(Dataset):
    def __init__(
        self,
        audio_dataframe,
        latent_dataframe,
        unprocessed_wav_path,
        chunk_size=2048,
        n_bands=8,
        allowed_indices=None,
    ):
        self.chunk_size = chunk_size
        self.n_bands = n_bands
        # self.df = dataframe.copy()

        # drop file column from both
        audio_df = audio_dataframe.drop(columns=['file'])
        latent_df = latent_dataframe.drop(columns=['file', 'sweep_path'])

        # create a merged dataframe using keys pedal, gain, tone
        self.df = audio_df.merge(
            latent_df,
            on=["pedal", "gain", "tone"],
            how="inner",
            suffixes=('_audio', '_latent')
        )
        print(f"Merged dataframe has {len(self.df)} entries.")

        self.unprocessed_audio, self.sr = self._load_unprocessed_audio(unprocessed_wav_path)

        self.pedals_list = sorted(self.df['pedal'].unique().tolist())
        self.pedal2id = {p: i for i, p in enumerate(self.pedals_list)}
        self.gain_list = sorted(self.df['gain'].unique().tolist())
        self.tone_list = sorted(self.df['tone'].unique().tolist())

        self.allowed_indices = allowed_indices
        self.df_by_chunk = self._index_chunks()

        if self.n_bands > 1:
            self.filter = MelBandFilter(n_bands=n_bands, filter_len=513, sr=self.sr)

    def _load_unprocessed_audio(self, path):
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio, sr

    def _index_chunks(self):
        return {
            chunk_idx: self.df[self.df['index'] == chunk_idx]
            for chunk_idx in self.allowed_indices
        }

    def _load_audio(self, path):
        processed_audio = np.load(path).astype(np.float32)
        if len(processed_audio) < self.chunk_size:
            processed_audio = np.pad(
                processed_audio,
                (0, self.chunk_size - len(processed_audio)),
                mode='constant'
            )
        return processed_audio

    def _get_input_chunk(self, chunk_idx):
        start = chunk_idx * self.chunk_size
        end = start + self.chunk_size
        input_chunk = self.unprocessed_audio[start:end]

        if len(input_chunk) < self.chunk_size:
            input_chunk = np.pad(
                input_chunk,
                (0, self.chunk_size - len(input_chunk)),
                mode='constant'
            )

        return torch.from_numpy(input_chunk.astype(np.float32))


    def _process_single_band(self, df_chunk):
        row = df_chunk.sample(n=1).iloc[0]

        processed_audio = self._load_audio(row['chunk_path'])
        processed_tensor = torch.from_numpy(processed_audio)

        latent = row['latents']
        # If it's a string, parse it manually
        if isinstance(latent, str):
            # Remove brackets and split by comma
            latent = latent.strip('[]').split(',')
            # Convert each element to float
            latent = [float(x.strip()) for x in latent]

        # Now convert to numpy array
        if not isinstance(latent, np.ndarray):
            latent = np.asarray(latent, dtype=np.float32)

        latent_vec = torch.from_numpy(latent).unsqueeze(0)

        return {
            'target_bands': processed_tensor.unsqueeze(0),
            'conditioning': latent_vec,
            'pedal_names': [row['pedal']],
            'gains': [row['gain']],
            'tones': [row['tone']]
        }

    def _sample_unique_effects(self, df_chunk):
        alpha = 0.4
        max_unique = min(self.n_bands, len(df_chunk))

        probs = np.exp(alpha * np.arange(1, max_unique + 1))
        probs /= probs.sum()

        k_unique = np.random.choice(np.arange(1, max_unique + 1), p=probs)
        sampled_rows = df_chunk.sample(n=k_unique, replace=False)

        sample_pool_indices = list(range(len(sampled_rows)))
        assigned_sample_indices = np.random.choice(
            sample_pool_indices, size=self.n_bands, replace=True
        )

        return sampled_rows, assigned_sample_indices

    def _load_samples(self, sampled_rows):
        loaded_audio = {}
        loaded_latents = {}
        pedal_names = {}
        gains = {}
        tones = {}

        for i, (_, row) in enumerate(sampled_rows.iterrows()):
            loaded_audio[i] = self._load_audio(row['chunk_path'])

            latent = row['latents']

            # If it's a string, parse it manually
            if isinstance(latent, str):
                # Remove brackets and split by comma
                latent = latent.strip('[]').split(',')
                # Convert each element to float
                latent = [float(x.strip()) for x in latent]

            # Now convert to numpy array
            if not isinstance(latent, np.ndarray):
                latent = np.asarray(latent, dtype=np.float32)

            loaded_latents[i] = torch.from_numpy(latent)

            pedal_names[i] = row['pedal']
            gains[i] = row['gain']
            tones[i] = row['tone']

        return loaded_audio, loaded_latents, pedal_names, gains, tones

    def _apply_band_filter(self, audio_tensor, band_idx):
        all_filters = self.filter.filters.view(self.n_bands, 1, -1)
        pad = (self.filter.filter_len - 1) // 2

        h = all_filters[band_idx:band_idx+1]
        y_i = F.conv1d(audio_tensor, h, padding=pad)

        return y_i.squeeze(0).squeeze(0)[:self.chunk_size]

    def _process_multi_band(self, df_chunk):
        sampled_rows, assigned_sample_indices = self._sample_unique_effects(df_chunk)
        loaded_audio, loaded_latents, pedal_names, gains, tones = self._load_samples(sampled_rows)

        target_bands = []
        conditioning_list = []
        output_pedal_names = []
        output_gains = []
        output_tones = []

        for band_idx, sample_idx in enumerate(assigned_sample_indices):
            proc_audio = loaded_audio[sample_idx]
            proc_audio_tensor = torch.from_numpy(proc_audio).view(1, 1, -1)

            target_band = self._apply_band_filter(proc_audio_tensor, band_idx)

            target_bands.append(target_band)
            conditioning_list.append(loaded_latents[sample_idx])
            output_pedal_names.append(pedal_names[sample_idx])
            output_gains.append(gains[sample_idx])
            output_tones.append(tones[sample_idx])

        return {
            'target_bands': torch.stack(target_bands, dim=0), 
            'conditioning': torch.stack(conditioning_list, dim=0), 
            'pedal_names': output_pedal_names,
            'gains': output_gains,
            'tones': output_tones
        }


    def __len__(self):
        return len(self.allowed_indices)

    def __getitem__(self, idx):
        chunk_idx = self.allowed_indices[idx]
        input_chunk = self._get_input_chunk(chunk_idx)

        df_chunk = self.df_by_chunk[chunk_idx]
        if len(df_chunk) == 0:
            raise ValueError(f"No processed samples for chunk {chunk_idx}")

        if self.n_bands == 1:
            processed_data = self._process_single_band(df_chunk)
        else:
            processed_data = self._process_multi_band(df_chunk)

        return {
            "input_chunk": input_chunk,
            "conditioning": processed_data['conditioning'],
            "target_bands": processed_data['target_bands'],
            "pedal_names": processed_data['pedal_names'],
            "gains": processed_data['gains'],
            "tones": processed_data['tones'],
            "_raw_dataset_idx": idx,
        }

def main():
    import pandas as pd
    TEST_AUDIO_DF = pd.read_csv("_prepared_data/audio_chunks_metadata.csv")
    TEST_LATENT_DF = pd.read_csv("_prepared_data/metadata_with_latents.csv")

    dataset = Prism_Dataset(
        audio_dataframe=TEST_AUDIO_DF,
        latent_dataframe=TEST_LATENT_DF,
        unprocessed_wav_path="DATA/input_unprocessed.wav",
        chunk_size=2048,
        n_bands=4,
        allowed_indices=[0, 1, 2, 3, 4, 5]
    )

if __name__ == "__main__":
    main()
    pass