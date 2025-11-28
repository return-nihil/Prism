import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
import random

from wntcn_dataset import Prism_Dataset, MelBandFilter
from wntcn_model import WN_TCN, init_weights
from wntcn_losses import WNTCN_Loss
from wntcn_train import train_one_epoch, test_model


CHUNK_SIZE = 2048
BAND_CONFIGS = [8, 6, 4, 3, 2, 1]
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 1e-3
MIN_LR = 1e-6
SEED = 42
DATA_CSV = ""
UNPROCESSED_WAV = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PEDAL_LIST = ["rr", "rm", "kot"]
LATENT_DIM = 8
DATA_EMB = ""
OUTPUT_DIR = "./OUTS"


os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def run_training_pipeline():
    df_full = pd.read_csv(DATA_CSV)
    df_emb_full = pd.read_csv(DATA_EMB)
    df = df_full[df_full['effect'].isin(PEDAL_LIST)].reset_index(drop=True)
    df_emb = df_emb_full[df_emb_full['label'].isin(PEDAL_LIST)].reset_index(drop=True)
    all_indices = df['index'].unique()
    max_index = all_indices.max()
    
    all_indices = np.arange(max_index + 1)
    np.random.seed(SEED)
    np.random.shuffle(all_indices)
    
    split = int(len(all_indices) * 0.8)
    train_indices = all_indices[:split]
    test_indices = all_indices[split:]
    
    print(f"Total chunks: {len(all_indices)}, Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"Split is fixed with SEED={SEED}")
    
    results_file = os.path.join(OUTPUT_DIR, "results.txt")
    
    with open(results_file, 'w') as f:
        f.write(f"Multi-band Training Experiment (Simplified v2)\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seed: {SEED}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Train Chunks: {len(train_indices)}, Test Chunks: {len(test_indices)}\n")
        f.write("="*80 + "\n\n")
    
    for n_bands in BAND_CONFIGS:
        print(f"\n{'='*80}")
        print(f"Training model with {n_bands} band(s)")
        print(f"{'='*80}\n")
        
        
        train_dataset = Prism_Dataset(
            dataframe=df,
            dataframe_emb=df_emb,
            unprocessed_wav_path=UNPROCESSED_WAV,
            chunk_size=CHUNK_SIZE,
            n_bands=n_bands,
            allowed_indices=train_indices
        )
        
        test_dataset = Prism_Dataset(
            dataframe=df,
            dataframe_emb=df_emb,
            unprocessed_wav_path=UNPROCESSED_WAV,
            chunk_size=CHUNK_SIZE,
            n_bands=n_bands,
            allowed_indices=test_indices
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, drop_last=True, pin_memory=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32, drop_last=True, pin_memory=True, prefetch_factor=2)
        

        model = WN_TCN(inp_channel=1, 
                      out_channel=1, 
                      channels=64, 
                      n_blocks=8, 
                      cond_dim=64, 
                      sample_rate=48000, 
                      band_hidden=64,
                      latent_dim=LATENT_DIM,
                      n_bands=n_bands 
                      ).to(DEVICE)
        model.apply(init_weights)

        num_params = model.compute_num_of_params()
        rf, rf_ms = model.compute_receptive_field()
        print(f"Model parameters: {num_params:,}")
        print(f"Receptive field: {rf} samples ({rf_ms:.2f} ms)")
        
        criterion = WNTCN_Loss().to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(MIN_LR / LEARNING_RATE, 1.0 - (epoch / EPOCHS))
        )
        
        filterbank = MelBandFilter(n_bands=n_bands).to(DEVICE)
        
        for epoch in range(EPOCHS):
            train_loss, train_stft, train_esr, train_aux = train_one_epoch(
                model, train_loader, optimizer, criterion, filterbank, DEVICE, n_bands, epoch, EPOCHS
            )
            
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f" TRAIN | Loss: {train_loss:.6f}, STFT: {train_stft:.6f}, ESR: {train_esr:.6f}, AUX: {train_aux:.6f}")
            print(f" LR: {current_lr:.6e}")
            
            scheduler.step()
        
        final_test_loss, final_test_stft, final_test_esr, final_test_aux, final_test_mse, final_test_mae = test_model(
            model, test_loader, criterion, filterbank, DEVICE
        )
    
        
        model_path = os.path.join(OUTPUT_DIR, f"model_{n_bands}bands.pth")
        torch.save({
            'n_bands': n_bands,
            'model_state_dict': model.state_dict(),
            'final_test_loss': final_test_loss,
            'final_test_stft': final_test_stft,
            'final_test_esr': final_test_esr,
            'final_test_aux': final_test_aux,
            'final_test_mse': final_test_mse,
            'final_test_mae': final_test_mae,
            'num_params': num_params
        }, model_path)
        print(f"Model saved: {model_path}")
        
        with open(results_file, 'a') as f:
            f.write(f"N_BANDS: {n_bands}\n")
            f.write(f"Parameters: {num_params:,}\n")
            f.write(f"Receptive field: {rf} samples ({rf_ms:.2f} ms)\n")
            f.write(f"Final Test Loss: {final_test_loss:.6f}\n")
            f.write(f"Final Test STFT: {final_test_stft:.6f}\n")
            f.write(f"Final Test ESR: {final_test_esr:.6f}\n")
            f.write(f"Final Test AUX: {final_test_aux:.6f}\n")
            f.write(f"Final Test MSE: {final_test_mse:.6f}\n")
            f.write(f"Final Test MAE: {final_test_mae:.6f}\n")
            f.write(f"Model saved: {model_path}\n")
            f.write("-"*80 + "\n\n")
        
        print(f"\nResults appended to {results_file}\n")
        
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved to: {results_file}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")



if __name__ == "__main__":
    run_training_pipeline()