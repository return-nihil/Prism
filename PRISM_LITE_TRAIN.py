import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import wandb
from tqdm import tqdm
import os
import random
from datetime import datetime

from datasets_emb import Prism_Dataset, MelBandFilter
from PRISM_LITE2 import PrismLite, init_weights
from losses import TCN_Loss

CHUNK_SIZE = 2048
BAND_CONFIGS = [16]#[1, 2, 4, 8, 16]
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 1e-3
MIN_LR = 1e-6
SEED = 42
DATA_CSV = ""
DATA_EMB = ""
UNPROCESSED_WAV = ""
OUTPUT_DIR = "./experiment_outputs_EMB_LITE_128_NEW_DATALOADER"

os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def tensor_to_numpy_for_wandb(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze()
    tensor = tensor * 3
    return tensor.numpy().astype(np.float32)


def spectral_convergence_loss(pred, target, n_fft=2048, hop_length=512):
    pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                           window=torch.hann_window(n_fft).to(pred.device), 
                           return_complex=True)
    target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, hop_length=hop_length, 
                             window=torch.hann_window(n_fft).to(target.device), 
                             return_complex=True)
    
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    
    sc_loss = torch.norm(pred_mag - target_mag, p='fro') / torch.norm(target_mag, p='fro')
    return sc_loss



def train_one_epoch(model, loader, optimizer, criterion, filterbank, device, n_bands, epoch):
    model.train()
    running_loss, running_main, running_aux = 0.0, 0.0, 0.0
    running_stft, running_esr = 0.0, 0.0
    pbar = tqdm(loader, desc="Training")
    
    last_batch = None
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
        
        last_batch = (y_pred_bands, y_target)
    
    avg_loss = running_loss / len(loader)
    avg_main = running_main / len(loader)
    avg_aux = running_aux / len(loader)
    avg_stft = running_stft / len(loader)
    avg_esr = running_esr / len(loader)
    
    log_dict = {
        "train/loss": avg_loss,
        "train/main_loss": avg_main,
        "train/aux_loss": avg_aux,
        "train/stft": avg_stft,
        "train/esr": avg_esr,
        "train/aux_weight": aux_weight,
        "epoch": epoch
    }
    
    if last_batch is not None:
        y_pred_bands, y_target = last_batch
        y_pred_full = filterbank.inverse(y_pred_bands[0:1])
        y_target_full = filterbank.inverse(y_target[0:1])
        pred_audio = tensor_to_numpy_for_wandb(y_pred_full)
        target_audio = tensor_to_numpy_for_wandb(y_target_full)
        log_dict.update({
            "train/predicted_audio": wandb.Audio(pred_audio, sample_rate=48000),
            "train/target_audio": wandb.Audio(target_audio, sample_rate=48000)
        })
    
    wandb.log(log_dict)
    return avg_loss, avg_stft, avg_esr


def test_model(model, loader, criterion, filterbank, device, n_bands, epoch):
    model.eval()
    running_loss, running_stft, running_esr = 0.0, 0.0, 0.0
    running_mse, running_mae = 0.0, 0.0
    last_batch = None
    
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
            
            last_batch = (y_pred_bands, y_target)
    
    avg_loss = running_loss / len(loader)
    avg_stft = running_stft / len(loader)
    avg_esr = running_esr / len(loader)
    avg_mse = running_mse / len(loader)
    avg_mae = running_mae / len(loader)
    
    log_dict = {
        "test/loss": avg_loss,
        "test/stft": avg_stft,
        "test/esr": avg_esr,
        "test/mse": avg_mse,
        "test/mae": avg_mae,
        "epoch": epoch
    }
    
    if last_batch is not None:
        y_pred_bands, y_target = last_batch
        y_pred_full = filterbank.inverse(y_pred_bands[0:1])
        y_target_full = filterbank.inverse(y_target[0:1])
        pred_audio = tensor_to_numpy_for_wandb(y_pred_full)
        target_audio = tensor_to_numpy_for_wandb(y_target_full)
        log_dict.update({
            "test/predicted_audio": wandb.Audio(pred_audio, sample_rate=48000),
            "test/target_audio": wandb.Audio(target_audio, sample_rate=48000)
        })
    
    wandb.log(log_dict)
    return avg_loss, avg_stft, avg_esr, avg_mse, avg_mae


if __name__ == "__main__":
    
    df = pd.read_csv(DATA_CSV)
    df_emb = pd.read_csv(DATA_EMB)
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
        
        wandb.init(
            project="prism-multiband-v2",
            name=f"{n_bands}_bands_LITE_128_NEW_DATA",
            config={
                "n_bands": n_bands,
                "chunk_size": CHUNK_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "min_lr": MIN_LR,
                "seed": SEED,
                "train_chunks": len(train_indices),
                "test_chunks": len(test_indices)
            }
        )
        
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
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
        
        model = PrismLite(
            inp_channel=1,
            out_channel=1,
            channels=64,
            kernel_size=3,
            n_blocks=9,
            cond_dim=128, # 64
            sample_rate=48000,
            n_bands=n_bands,
        ).to(device)
        model.apply(init_weights)
        
        num_params = model.compute_num_of_params()
        rf, rf_ms = model.compute_receptive_field()
        print(f"Model parameters: {num_params:,}")
        print(f"Receptive field: {rf} samples ({rf_ms:.2f} ms)")
        
        criterion = TCN_Loss().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max(MIN_LR / LEARNING_RATE, 1.0 - (epoch / EPOCHS))
        )
        
        filterbank = MelBandFilter(n_bands=n_bands).to(device)
        
        for epoch in range(EPOCHS):
            train_loss, train_stft, train_esr = train_one_epoch(
                model, train_loader, optimizer, criterion, filterbank, device, n_bands, epoch
            )
            
            current_lr = scheduler.get_last_lr()[0]
            wandb.log({
                "learning_rate": current_lr,
                "epoch": epoch
            })
            
            print(f"Epoch {epoch+1}/{EPOCHS}")
            print(f" TRAIN | Loss: {train_loss:.6f}, STFT: {train_stft:.6f}, ESR: {train_esr:.6f}")
            print(f" LR: {current_lr:.6e}")
            
            scheduler.step()
        
        final_test_loss, final_test_stft, final_test_esr, final_test_mse, final_test_mae = test_model(
            model, test_loader, criterion, filterbank, device, n_bands, EPOCHS-1
        )
        
        wandb.run.summary["final_test_loss"] = final_test_loss
        wandb.run.summary["final_test_stft"] = final_test_stft
        wandb.run.summary["final_test_esr"] = final_test_esr
        wandb.run.summary["num_params"] = num_params
        
        model_path = os.path.join(OUTPUT_DIR, f"model_{n_bands}bands.pth")
        torch.save({
            'n_bands': n_bands,
            'model_state_dict': model.state_dict(),
            'final_test_loss': final_test_loss,
            'final_test_stft': final_test_stft,
            'final_test_esr': final_test_esr,
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
            f.write(f"Final Test MSE: {final_test_mse:.6f}\n")
            f.write(f"Final Test MAE: {final_test_mae:.6f}\n")
            f.write(f"Model saved: {model_path}\n")
            f.write("-"*80 + "\n\n")
        
        print(f"\nResults appended to {results_file}\n")
        
        wandb.finish()
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"Results saved to: {results_file}")
    print(f"Models saved to: {OUTPUT_DIR}")
    print(f"{'='*80}\n")