import torch
import torch.nn.functional as F
from tqdm import tqdm



def train_one_epoch(model, loader, optimizer, criterion, filterbank, device, n_bands, epoch, max_epochs):
    model.train()
    running_loss, running_main = 0.0, 0.0
    running_stft, running_esr, running_aux = 0.0, 0.0, 0.0
    pbar = tqdm(loader, desc="Training")
    
    aux_weight = max(0.3 * (1 - epoch / max_epochs), 0.05)
    
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
