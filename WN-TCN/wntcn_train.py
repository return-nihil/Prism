import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import random
from datetime import datetime

from wntcn_dataset import Prism_Dataset, MelBandFilter
from wntcn_model import WN_TCN, init_weights
from wntcn_losses import WNTCN_Loss

CHUNK_SIZE = 2048
BAND_CONFIGS = [8, 6, 4, 3, 2, 1]
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 1e-3
MIN_LR = 1e-6
SEED = 42
DATA_CSV = ""
UNPROCESSED_WAV = ""

PEDAL_LIST = ["RR", "RM", "KoT"]
LATENT_DIM = 8
DATA_EMB = ""
OUTPUT_DIR = "./OUTS"


os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



def train_one_epoch(model, loader, optimizer, criterion, filterbank, device, n_bands, epoch):
    model.train()
    running_loss, running_main = 0.0, 0.0
    running_stft, running_esr, running_aux = 0.0, 0.0, 0.0
    pbar = tqdm(loader, desc="Training")
    
    aux_weight = max(0.3 * (1 - epoch / EPOCHS), 0.05)
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        x = batch["input_chunk"].to(device)
        y_target = batch["target_bands"].to(device)
        cond = batch["conditioning"].to(device)
        
        B, T = x.shape
        x_in = x.unsqueeze(-1)
        
        y_pred, _ = model(x_in, cond)
        y_pred = y_pred.permute(0, 2, 1)
        y_pred_bands = filterbank(y_pred)
        
        y_pred_full = filterbank.inverse(y_pred_bands)
        y_target_full = filterbank.inverse(y_target)
        
        stft_loss_full, esr_loss_full = criterion(y_pred_full, y_target_full)
        main_loss = stft_loss_full + esr_loss_full
        
        aux_loss = 0.0
        for b in range(n_bands):
            mse_band = F.mse_loss(y_pred_bands[:, b:b+1, :], y_target[:, b:b+1, :])
            aux_loss += mse_band * 20
        aux_loss /= n_bands
        
        loss = main_loss + aux_weight * aux_loss
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_main += main_loss.item()
        running_aux += aux_loss.item()
        running_stft += stft_loss_full.item()
        running_esr += esr_loss_full.item()
        
        pbar.set_postfix({
            "loss": running_loss / (batch_idx + 1),
            "main": running_main / (batch_idx + 1),
            "aux": running_aux / (batch_idx + 1)
        })
    
    avg_loss = running_loss / len(loader)
    avg_stft = running_stft / len(loader)
    avg_esr = running_esr / len(loader)
    avg_aux = running_aux / len(loader)
    
    return avg_loss, avg_stft, avg_esr, avg_aux


def test_model(model, loader, criterion, filterbank, device):
    model.eval()
    running_loss, running_stft, running_esr, running_aux = 0.0, 0.0, 0.0, 0.0
    running_mse, running_mae = 0.0, 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            x = batch["input_chunk"].to(device)
            y_target = batch["target_bands"].to(device)
            cond = batch["conditioning"].to(device)
            
            B, T = x.shape
            x_in = x.unsqueeze(-1)
            
            y_pred, _ = model(x_in, cond)
            y_pred = y_pred.permute(0, 2, 1)
            y_pred_bands = filterbank(y_pred)
            
            y_pred_full = filterbank.inverse(y_pred_bands)
            y_target_full = filterbank.inverse(y_target)
            
            stft_loss_full, esr_loss_full = criterion(y_pred_full, y_target_full)
            mse_loss_full = F.mse_loss(y_pred_full, y_target_full)
            mae_loss_full = F.l1_loss(y_pred_full, y_target_full)
            
            loss = stft_loss_full + esr_loss_full
            
            running_loss += loss.item()
            running_stft += stft_loss_full.item()
            running_esr += esr_loss_full.item()
            running_mse += mse_loss_full.item()
            running_mae += mae_loss_full.item()
    
    avg_loss = running_loss / len(loader)
    avg_stft = running_stft / len(loader)
    avg_esr = running_esr / len(loader)
    avg_mse = running_mse / len(loader)
    avg_mae = running_mae / len(loader)
    avg_aux = running_aux / len(loader)
    
    return avg_loss, avg_stft, avg_esr, avg_aux, avg_mse, avg_mae


if __name__ == "__main__":
    
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
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
                      ).to(device)
        model.apply(init_weights)

        num_params = model.compute_num_of_params()
        rf, rf_ms = model.compute_receptive_field()
        print(f"Model parameters: {num_params:,}")
        print(f"Receptive field: {rf} samples ({rf_ms:.2f} ms)")
        
        criterion = WNTCN_Loss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(MIN_LR / LEARNING_RATE, 1.0 - (epoch / EPOCHS))
        )
        
        filterbank = MelBandFilter(n_bands=n_bands).to(device)
        
        for epoch in range(EPOCHS):
            train_loss, train_stft, train_esr, train_aux = train_one_epoch(
                model, train_loader, optimizer, criterion, filterbank, device, n_bands, epoch
            )
            
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f" TRAIN | Loss: {train_loss:.6f}, STFT: {train_stft:.6f}, ESR: {train_esr:.6f}, AUX: {train_aux:.6f}")
            print(f" LR: {current_lr:.6e}")
            
            scheduler.step()
        
        final_test_loss, final_test_stft, final_test_esr, final_test_aux, final_test_mse, final_test_mae = test_model(
            model, test_loader, criterion, filterbank, device
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