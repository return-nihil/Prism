# Prism

## Prepare data: 
Choose and load individual pedal folders and rename paths/data_folder in the config accordingly;

Rename the selected pedals you want to train with in data_processing/pedals. 

Prepare data running `DATA/prepare_data.py`. This should create separate folders for audio chunks and sweeps, along with metadata csv files. 


## Train the VAE
Run `VAE/_vae_main.py`. This should output model, tsne for visualization, and an additional metadata containing the latents to be used as conditioning. 


## Train the TCN
Train the TCN running `WN-TCN/_wntcn_main.py`. 

