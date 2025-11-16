import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE


def normalize_coordinates(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def visualize_latents(reduction, X_transformed, y, image_path, label_type="pedal"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    if label_type == "pedal":
        labels = y["pedal"]
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


def tsne_on_latents(latents_csv_path, csv_path, image_path, label_type="pedal", perplexity=50, n_iter=1000):
    df = pd.read_csv(latents_csv_path)
    df["latents"] = df["latents"].apply(eval)
    X = np.array(df["latents"].to_list())
    y = df[["pedal", "gain", "tone"]]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, len(y)-1), max_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(X)
    X_tsne = normalize_coordinates(X_tsne)
    df["coords"] = X_tsne.tolist()

    df["pedal"] = df["pedal"]
    df = df[["pedal", "gain", "tone", "latents", "coords"]]
    df.to_csv(csv_path, index=False)
    
    visualize_latents("TSNE", X_tsne, y, image_path, label_type)