import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from vae_model import Prism_VAE, weights_init
from vae_dataset import VAE_Dataset
from vae_train import train
from vae_visualizers import tsne_on_latents


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_MODELS = ""
OUTPUT_VAE_LATENTS_CSV = ""
METADATA_CSV = ""
LABELS = ["kot", "rr", "rm"] # dk jr


def extract_latents(dataloader, model, label_to_index, index_to_label, output_path, device):
    model.eval()
    all_data = []

    for batch in dataloader:
        sweep = batch["sweep"].float().to(device).unsqueeze(1)
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
    df.to_csv(output_path, index=False)
    print("Latents saved")


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
                                  batch_size=16, 
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset, 
                                batch_size=16, 
                                shuffle=False)

    model = Prism_VAE().to(DEVICE)
    model.apply(weights_init)

    optimizer_model = Adam(model.parameters(), 
                           lr=1e-3, 
                           weight_decay=1e-5)

    os.makedirs(OUTPUT_MODELS, exist_ok=True)

    train(model=model,
          train_loader=train_dataloader,
          val_loader=val_dataloader,
          optimizer_model=optimizer_model, 
          epochs=1000,
          device=DEVICE,
          output_path=OUTPUT_MODELS)

    extract_latents(
        dataloader=full_dataloader,
        model=model,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        output_path=OUTPUT_VAE_LATENTS_CSV,
        device=DEVICE)

    tsne_on_latents("vae_output/vae_latents.csv", 
                    "vae_output/vae_latents_tsne.csv", 
                    "vae_output/vae_latents_tsne.png")

if __name__ == "__main__":
    run_training_pipeline()