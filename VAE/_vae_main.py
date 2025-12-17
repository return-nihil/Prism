import torch
import pandas as pd

from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from vae_model import Prism_VAE, weights_init
from vae_dataset import VAE_Dataset
from vae_train import train
from vae_visualizers import tsne_on_latents

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_config import load_config
cfg = load_config("config.yaml")


DATA_PROCESSED_FOLDER = cfg["paths"]["data_processed_folder"]
VAE_OUTPUT_FOLDER = cfg["paths"]["vae_output_folder"]
os.makedirs(VAE_OUTPUT_FOLDER, exist_ok=True)
METADATA_CSV = os.path.join(DATA_PROCESSED_FOLDER, "sweeps_metadata.csv")
LABELS = cfg["data_processing"]["pedals"]
BATCH_SIZE = cfg["training"]["vae"]["batch_size"]
EPOCHS = cfg["training"]["vae"]["epochs"]
LEARNING_RATE = cfg["training"]["vae"]["learning_rate"]
LATENT_DIM = cfg["models"]["vae"]["latent_dim"]
EXPANSION_FACTOR = cfg["models"]["vae"]["expansion_factor"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_latents(dataloader, model, label_to_index, index_to_label, output_path, device):
    model.eval()
    all_data = []

    for batch in dataloader:
        sweep = batch["sweep"].float().to(device).unsqueeze(1)
        for label in batch["pedal"]:
            assert label in label_to_index, f"Label {label} not found in label_to_index mapping."
        target_class = torch.tensor([label_to_index[label] for label in batch["pedal"]], dtype=torch.long).to(device)
        target_gain = batch["gain"].clone().detach().to(torch.long).to(device)
        target_tone = batch["tone"].clone().detach().to(torch.long).to(device)
        _, _, _, z = model(sweep)

        for i in range(z.shape[0]):
            all_data.append({
                "pedal": target_class[i].item(),
                "gain": target_gain[i].item(),
                "tone": target_tone[i].item(),
                "latents": z[i].cpu().detach().numpy().tolist()
            })

    df = pd.DataFrame(all_data)
    df['pedal'] = df['pedal'].map(index_to_label)
    df.to_csv(os.path.join(output_path, "vae_latents.csv"), index=False)
    print("Latents saved to:", os.path.join(output_path, "vae_latents.csv"))


def merge_metadata_with_latents(metadata_csv, latents_csv, output_csv):
    metadata_df = pd.read_csv(metadata_csv)
    latents_df = pd.read_csv(latents_csv)

    # Print metadata_df columns
    print("Metadata DataFrame columns:", metadata_df.columns.tolist())
    # Print latents_df columns
    print("Latents DataFrame columns:", latents_df.columns.tolist())

    merged_df = pd.merge(metadata_df, latents_df, on=["pedal", "gain", "tone"])
    merged_df.to_csv(output_csv, index=False)
    print(f"Merged metadata and latents saved to: {output_csv}")


def run_training_pipeline():

    dataf = pd.read_csv(METADATA_CSV)
    dataframe = dataf[dataf['pedal'].isin(LABELS)]
    
    full_dataset = VAE_Dataset(dataframe)
    labels = full_dataset.get_unique_labels()
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    
    full_dataloader = DataLoader(full_dataset, 
                                 batch_size=16, 
                                 shuffle=False)
    
    train_df, val_df = train_test_split(dataframe, test_size=0.1, random_state=42, shuffle=True)
    train_dataset = VAE_Dataset(train_df)
    val_dataset = VAE_Dataset(val_df)
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=BATCH_SIZE, 
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=BATCH_SIZE, 
                                shuffle=False)

    model = Prism_VAE(latent_dim=LATENT_DIM, 
                      expansion_factor=EXPANSION_FACTOR).to(DEVICE)
    model.apply(weights_init)

    optimizer_model = Adam(model.parameters(), 
                           lr=LEARNING_RATE, 
                           weight_decay=1e-5)


    train(model=model,
          train_loader=train_dataloader,
          val_loader=val_dataloader,
          optimizer_model=optimizer_model, 
          epochs=EPOCHS,
          device=DEVICE,
          output_path=VAE_OUTPUT_FOLDER)

    extract_latents(
        dataloader=full_dataloader,
        model=model,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        output_path=VAE_OUTPUT_FOLDER,
        device=DEVICE)

    tsne_on_latents(os.path.join(VAE_OUTPUT_FOLDER,"vae_latents.csv"), 
                    os.path.join(VAE_OUTPUT_FOLDER,"vae_latents_tsne.csv"), 
                    os.path.join(VAE_OUTPUT_FOLDER,"vae_latents_tsne.png"))
    
    merge_metadata_with_latents(METADATA_CSV, 
                                os.path.join(VAE_OUTPUT_FOLDER,"vae_latents.csv"),
                                os.path.join(DATA_PROCESSED_FOLDER,"metadata_with_latents.csv"))

if __name__ == "__main__":
    run_training_pipeline()