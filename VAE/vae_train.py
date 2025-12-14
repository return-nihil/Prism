import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from vae_losses import VAE_Train_Loss, VAE_Valid_Loss


def train(model, 
          train_loader, 
          val_loader, 
          optimizer_model,
          epochs,
          device,
          output_path):

    train_loss = VAE_Train_Loss(device).to(device)
    scheduler = ReduceLROnPlateau(optimizer_model, 
                                  mode='min', 
                                  factor=0.5, 
                                  patience=15, 
                                  threshold=1e-7)

    best_val_l1 = float('inf')
    early_stopping_patience = 50
    epochs_no_improve = 0

    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_stft_loss = 0
        total_kl_loss = 0

        for batch in train_loader:
            model.train()
            audio = batch['audio'].float().to(device).unsqueeze(1)

            optimizer_model.zero_grad()
            noise = torch.randn(audio.size(0), 1, audio.size(2)).to(device) * 0.000001
            audio_noise = audio + noise
            output, mu, logvar, _ = model(audio_noise)

            loss, recon_loss, stft_loss, kl_loss = train_loss(audio, output, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer_model.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_stft_loss += stft_loss.item()
            total_kl_loss += kl_loss.item()

        _, _, val_l1 = evaluate(model, val_loader, device)
        scheduler.step(val_l1)

        if val_l1 < best_val_l1 - 1e-6:
            best_val_l1 = val_l1
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in {early_stopping_patience} epochs.")
                break

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Recon: {total_recon_loss / len(train_loader):.4f} | STFT: {total_stft_loss / len(train_loader):.4f} | "
              f"KL: {total_kl_loss / len(train_loader):.4f} | "
              f"Val L1: {val_l1:.4f}")

    torch.save(model.state_dict(), output_path.replace('.pth', '_final.pth'))


def evaluate(model, val_loader, device):
    model.eval()
    valid_loss = VAE_Valid_Loss(device)
    recon, stft, l1 = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].float().to(device).unsqueeze(1)
            output, _, _, _ = model(audio)
            recon_loss, stft_loss, l1_loss = valid_loss(audio, output)
            recon += recon_loss.item()
            stft += stft_loss.item()
            l1 += l1_loss.item()

        num_batches = len(val_loader)
        avg_recon = recon / num_batches
        avg_stft = stft / num_batches
        avg_l1 = l1 / num_batches

        return avg_recon, avg_stft, avg_l1