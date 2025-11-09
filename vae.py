import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import wandb
from scipy import signal
from torch.utils.data import Dataset


LOGS = False 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if LOGS:
    wandb.init(project="MorphDrive_VAE", name="VAE_Training")


OUTPUT_MODELS = ""
OUTPUT_VAE_LATENTS_CSV = ""
METADATA_CSV = ""



def weights_init(model):
    if isinstance(model, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(model.weight)


class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullyConnected, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation='leaky', bn=True):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels) if bn else nn.Identity()
        self.activation = nn.LeakyReLU() if activation == 'leaky' else nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    

class DeConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation='silu', bn=True):
        super(DeConvLayer, self).__init__()
        self.bn = bn
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU() if activation == 'silu' else nn.Tanh()

    def forward(self, x):
        x = self.deconv(x)
        x = self.batch_norm(x) if self.bn else x
        x = self.activation(x)
        return x


class LatentSpace(nn.Module):
    def __init__(self, latent_dim, expansion_factor, weigth=1):
        super(LatentSpace, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(latent_dim*expansion_factor, 256),
            nn.SiLU(),
            nn.Linear(256, 64),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.weigth = weigth
    
    def _reparametrization_trick(self, mu, logvar, weight):
        sigma = torch.sqrt(torch.exp(logvar))
        eps = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(mu.device) # perche' lo devo mandare a device?
        z = mu + weight * sigma * eps
        return z

    def forward(self, x):
        x = self.linear_block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = torch.tanh(self._reparametrization_trick(mu, logvar, self.weigth))
        return mu, logvar, z


class Encoder(nn.Module):
    def __init__(self, input_dim, pedal_latent_dim, expansion_factor=32):
        super(Encoder, self).__init__()
        self.conv1 = ConvLayer(input_dim, 8, kernel_size=4, stride=2, padding=1)
        self.conv1a = ConvLayer(8, 8, kernel_size=4, stride=2, padding=3, dilation=2)
        self.conv1b = ConvLayer(8, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvLayer(16, 16, kernel_size=6, stride=4, padding=1)
        self.conv3 = ConvLayer(16, 32, kernel_size=6, stride=3, padding=1)
        self.conv4 = ConvLayer(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv5 = ConvLayer(64, 64, kernel_size=4, stride=2, padding=1)
        self.conv6 = ConvLayer(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv7 = ConvLayer(128, 256, kernel_size=4, stride=2, padding=0)
        self.conv8 = ConvLayer(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv9 = ConvLayer(128, 128, kernel_size=4, stride=2, padding=1)
        self.flat = nn.Flatten()
        self.fully = FullyConnected(2560, pedal_latent_dim*expansion_factor) 

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv1a(x)
        x2 = self.conv1b(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.flat(x)
        x = self.fully(x)

        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim):
        super(Decoder, self).__init__()
        self.fully1 = FullyConnected(latent_dim, latent_dim * 16)
        self.fully2 = FullyConnected(latent_dim * 16, 7936) 
        self.deconv1 = DeConvLayer(128, 256, kernel_size=4, stride=2, padding=1, activation='silu')
        self.deconv2 = DeConvLayer(256, 512, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv3 = DeConvLayer(512, 256, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv4 = DeConvLayer(256, 256, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv5 = DeConvLayer(256, 128, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv6 = DeConvLayer(128, 64, kernel_size=5, stride=2, padding=1, activation='silu')
        self.deconv7 = DeConvLayer(64, 32, kernel_size=5, stride=2, padding=1, activation='silu', bn=False)
        self.deconv8 = DeConvLayer(32, 16, kernel_size=4, stride=2, padding=0, activation='silu', bn=False)
        self.deconv9 = DeConvLayer(16, 16, kernel_size=4, stride=2, padding=1, activation='silu', bn=False)
        self.deconv10 = DeConvLayer(16, 8, kernel_size=4, stride=2, padding=1, activation='tanh', bn=False)
        self.final_conv = ConvLayer(8, output_dim, kernel_size=3, stride=1, padding=1, activation='tanh')
        self.denoiser = nn.Sequential(
            nn.Conv1d(output_dim, 32, kernel_size=3, stride=1, padding=1, groups=1),
            nn.Tanh(),
            nn.Conv1d(32, output_dim, kernel_size=1, stride=1, padding=0),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fully1(z)
        x = self.fully2(x)
        x = rearrange(x, 'b (c h) -> b c h', c=128, h=62)  
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = self.deconv7(x)
        x = self.deconv8(x)
        x = self.deconv9(x)
        x = self.deconv10(x)
        x = self.final_conv(x)
        x = self.denoiser(x)
        return x
    

class MorphDrive_VAE(nn.Module):
    def __init__(self, 
                 input_dim=1,
                 latent_dim=8,
                 expansion_factor=32
                 ):
        super(MorphDrive_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor
        self.encoder = Encoder(input_dim, self.latent_dim, self.expansion_factor)
        self.decoder = Decoder(input_dim, self.latent_dim)
        self.pedal_latent_space = LatentSpace(self.latent_dim, self.expansion_factor)

    def forward(self, x):
        features = self.encoder(x)
        mu, logvar, z = self.pedal_latent_space(features)
        output = self.decoder(z)
        return output, mu, logvar, z
    


def extract_spectrogram(audio, sr=32000):
    extracted_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, fmax=sr//2)
    log_spectrogram = librosa.power_to_db(extracted_spectrogram, ref=np.max)
    return log_spectrogram

def plot_spectrograms_to_wandb(model_output, original_audio, sr):
    import wandb

    predicted = model_output.squeeze().cpu().detach().numpy()
    original = original_audio.squeeze().cpu().detach().numpy()

    predicted_spectrogram = extract_spectrogram(predicted, sr)
    original_spectrogram = extract_spectrogram(original, sr)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(predicted_spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title('Predicted Spectrogram')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.imshow(original_spectrogram, aspect='auto', origin='lower', cmap='magma')
    plt.title('Original Spectrogram')
    plt.colorbar()

    wandb.log({"Spectrograms": wandb.Image(plt)})
    plt.close()


def load_audio_to_wandb(original, predicted, sr=32000):
    import wandb
    original = original.squeeze().cpu().detach().numpy()
    predicted = predicted.squeeze().cpu().detach().numpy()
    wandb.log({
        "original_audio": wandb.Audio(original, caption="Original Audio", sample_rate=sr),
        "predicted_audio": wandb.Audio(predicted, caption="Predicted Audio", sample_rate=sr)
    })


def normalize_coordinates(coords):
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    coords[:, 0] = (coords[:, 0] - x_min) / (x_max - x_min)
    coords[:, 1] = (coords[:, 1] - y_min) / (y_max - y_min)
    return coords


def normalize_coordinates(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def visualize_latents(reduction, X_transformed, y, image_path, label_type="pedal"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if label_type == "pedal":
        labels = y["label"]
    elif label_type == "gain":
        labels = y["gain"]
    elif label_type == "tone":
        labels = y["tone"]

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    cmap = plt.get_cmap("tab10" if num_labels <= 10 else "hsv")
    colors = [cmap(i / num_labels) for i in range(num_labels)]
    label_to_color = {label: colors[i] for i, label in enumerate(unique_labels)}
    label_names = {label: label if label_type == "pedal" else str(label) for label in unique_labels}

    for label in unique_labels:
        indices = (labels == label)
        ax.scatter(X_transformed[indices, 0], X_transformed[indices, 1],
                        color=[label_to_color[label]], label=label_names[label], s=100)

    ax.set_title(f"{reduction} on Latents ({label_type.capitalize()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    
    plt.legend()
    plt.savefig(image_path)
    plt.show()


def tsne_on_latents(latents_csv_path, csv_path, image_path, label_type="pedal", perplexity=50, n_iter=1000):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval) 
    X = np.array(df["latents"].to_list())
    y = df[["label", "g", "t"]]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(y)-1), max_iter=n_iter, random_state=42) #perpl 80 #iter 2000
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalize_coordinates(X_tsne)
    df["coords"] = X_tsne.tolist()  

    df["label"] = df["label"]
    df = df[["label", "g", "t", "latents", "coords"]]
    df.to_csv(csv_path, index=False)
    
    visualize_latents("TSNE", X_tsne, y, image_path, label_type)
    

from losses import MRSTFT_Loss # sistemare con loss a parte per VAE

class KL_Loss(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class VAE_Train_Loss(nn.Module):
    def __init__(self, 
                 recon_weigth=0.05,
                 stft_weight=1, 
                 kl_weight=0.0025):
        super().__init__()
        self.recon_mse = nn.MSELoss(reduction='sum')
        self.recon_huber = nn.HuberLoss(reduction='sum')
        self.stft_loss = MRSTFT_Loss().to(DEVICE)
        self.kl_loss = KL_Loss()
        self.recon_weight = recon_weigth
        self.stft_weight = stft_weight
        self.kl_weight = kl_weight

    def forward(self, target_audio, predicted_audio, mu, logvar):
        mse = self.recon_mse(target_audio, predicted_audio)
        huber = self.recon_huber(target_audio, predicted_audio)
        recon = torch.mean(mse + huber, dim=0) * self.recon_weight

        stft = self.stft_loss(target_audio, predicted_audio) * self.stft_weight
        kl = self.kl_loss(mu, logvar) * self.kl_weight

        total = recon + stft + kl 
        return total, recon, stft, kl


class VAE_Valid_Loss(nn.Module):
    def __init__(self,
                 recon_weigth=1,
                 stft_weight=1):
        super().__init__()
        self.recon_mse = nn.MSELoss(reduction='sum')
        self.recon_huber = nn.HuberLoss(reduction='sum')
        self.stft_loss = MRSTFT_Loss().to(DEVICE)
        self.recon_weight = recon_weigth
        self.stft_weight = stft_weight

    def forward(self, target_audio, predicted_audio):
        mse = self.recon_mse(target_audio, predicted_audio)
        huber = self.recon_huber(target_audio, predicted_audio)
        recon = mse + huber * self.recon_weight
        stft = self.stft_loss(target_audio, predicted_audio) * self.stft_weight
        l1 = F.l1_loss(target_audio, predicted_audio)

        return recon, stft, l1




class VAE_Dataset(Dataset):
    def __init__(self, dataframe, mode):
        assert mode in ["audio", "sweep"], "Mode must be 'audio' or 'sweep'"
        self.df = dataframe
        self.mode = mode

    def __len__(self):
        return len(self.df)
    
    
    def get_unique_labels(self):
        return self.df["label"].unique()
    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.mode == "audio":
            filepath = row["audio_path"]
        elif self.mode == "sweep":
            filepath = row["sweep_path"]

        assert os.path.exists(filepath), f"File not found: {filepath}"

        audio = np.load(filepath)
        '''if self.mode == "sweep":
            # resample to 32000 Hz
            audio = librosa.resample(audio, orig_sr=48000, target_sr=32000)'''

        return {
            "audio": audio,
            "label": row["label"],
            "g": row["g"],
            "t": row["t"]
        }
    




def train(model, 
          train_loader, 
          val_loader, 
          optimizer_model,
          epochs):

    train_loss = VAE_Train_Loss().to(DEVICE)
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
            audio = batch['audio'].float().to(DEVICE).unsqueeze(1)

            optimizer_model.zero_grad()
            # easy noise augm
            noise = torch.randn(audio.size(0), 1, audio.size(2)).to(DEVICE) * 0.000001
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

        val_recon, val_stft, val_l1 = evaluate(model, val_loader)
        scheduler.step(val_l1)

        if val_l1 < best_val_l1 - 1e-6:
            best_val_l1 = val_l1
            epochs_no_improve = 0
            torch.save(model.state_dict(), OUTPUT_MODELS)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in {early_stopping_patience} epochs.")
                break

        print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(train_loader):.4f} | "
              f"Recon: {total_recon_loss / len(train_loader):.4f} | STFT: {total_stft_loss / len(train_loader):.4f} | "
              f"KL: {total_kl_loss / len(train_loader):.4f} | "
              f"Val L1: {val_l1:.4f}")

        if LOGS and (epoch + 1) % 5 == 0:
            wandb.log({
                "train/total_loss": total_loss / len(train_loader),
                "train/recon_loss": total_recon_loss / len(train_loader),
                "train/stft_loss": total_stft_loss / len(train_loader),
                "train/kl_loss": total_kl_loss / len(train_loader),
                "val/l1_loss": val_l1,
                "val/recon_loss": val_recon,
                "val/stft_loss": val_stft,
            })

        if epoch % 20 == 0:
            plot_spectrograms_to_wandb(output[0], audio[0], sr=32000)
            load_audio_to_wandb(audio[0], output[0], sr=32000)

    torch.save(model.state_dict(), OUTPUT_MODELS + "vae_model_final.pth")



def evaluate(model, val_loader):
    model.eval()
    valid_loss = VAE_Valid_Loss()
    recon, stft, l1 = 0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            audio = batch['audio'].float().to(DEVICE).unsqueeze(1)
            output, _, _, _ = model(audio)
            recon_loss, stft_loss, l1_loss = valid_loss(audio, output)
            recon += recon_loss.item()
            stft += stft_loss.item()
            l1 += l1_loss.item()

        num_batches = len(val_loader)
        avg_recon = recon / num_batches
        avg_stft = stft / num_batches
        avg_l1 = l1 / num_batches

        if LOGS:
            wandb.log({
                "val/recon": recon / num_batches,
                "val/stft_loss": stft / num_batches,
                "val/l1_loss": l1 / num_batches
            })

        return avg_recon, avg_stft, avg_l1

    
def extract_latents(dataloader, model, label_to_index, index_to_label):
    model.eval()
    all_data = []

    for batch in dataloader:
        audio = batch["audio"].float().to(DEVICE).unsqueeze(1)
        target_class = torch.tensor([label_to_index[label] for label in batch["label"]], dtype=torch.long).to(DEVICE)
        target_gain = batch["g"].clone().detach().to(torch.long).to(DEVICE)
        target_tone = batch["t"].clone().detach().to(torch.long).to(DEVICE)
        _, _, _, z = model(audio)

        for i in range(z.shape[0]):
            all_data.append({
                "label": target_class[i].item(),
                "g": target_gain[i].item(),
                "t": target_tone[i].item(),
                "latents": z[i].cpu().detach().numpy().tolist()
            })

    df = pd.DataFrame(all_data)
    df['label'] = df['label'].map(index_to_label)
    df.to_csv(OUTPUT_VAE_LATENTS_CSV, index=False)
    print("Latents saved")


def run_training_pipeline():
    dataf = pd.read_csv(METADATA_CSV)
    # filter dataframe per label pedals = ["honeybee", "zendrive", "bigfella"]
    labels = ["honeybee", "zendrive", "bigfella"]
    dataframe = dataf[dataf['label'].isin(labels)]
    full_dataset = VAE_Dataset(dataframe, mode="sweep")
    labels = full_dataset.get_unique_labels()
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    full_dataloader = DataLoader(full_dataset, 
                                 batch_size=16, 
                                 shuffle=True)
    train_df, val_df = train_test_split(dataframe, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = VAE_Dataset(train_df, mode="sweep")
    val_dataset = VAE_Dataset(val_df, mode="sweep")
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=16, 
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=16, 
                                shuffle=False)


    model = MorphDrive_VAE().to(DEVICE)
    model.apply(weights_init)

    optimizer_model = Adam(model.parameters(), 
                           lr=1e-3, 
                           weight_decay=1e-5)

    os.makedirs(OUTPUT_MODELS, exist_ok=True)

    train(model=model,
          train_loader=train_dataloader,
          val_loader=val_dataloader,
          optimizer_model=optimizer_model, 
          epochs=1000)

    extract_latents(
        dataloader=full_dataloader,
        model=model,
        label_to_index=label_to_index,
        index_to_label=index_to_label)

    tsne_on_latents("vae_output/vae_latents.csv", 
                    "vae_output/vae_latents_tsne.csv", 
                    "vae_output/vae_latents_tsne.png")

    if LOGS:
        wandb.finish()


if __name__ == "__main__":
    run_training_pipeline()