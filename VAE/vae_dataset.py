import numpy as np
import os
from torch.utils.data import Dataset


class VAE_Dataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def get_unique_labels(self):
        return self.df["label"].unique()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["sweep_path"]

        assert os.path.exists(filepath), f"File not found: {filepath}"

        sweep = np.load(filepath)

        return {
            "sweep": sweep,
            "pedal": row["pedal"],
            "gain": row["gain"],
            "tone": row["tone"]
        }


if __name__ == "__main__":
    pass