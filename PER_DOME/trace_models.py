import os
import torch
from TCN import PrismLite

TRAINED_MODEL = ""
TRACED_MODEL = ""

device = torch.device("cpu")


model = PrismLite(
    inp_channel=1,
    out_channel=1,
    channels=64,
    kernel_size=3,
    n_blocks=9,
    cond_dim=128,
    sample_rate=48000,
    n_bands=8,
).to(device)

print(f"TRAINABLE PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


ckpt = torch.load(TRAINED_MODEL, map_location=device)

print(f"Checkpoint keys: {list(ckpt.keys())[:5]}...")

if "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
else:
    state_dict = ckpt  

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")

model.eval()

wav_x = torch.randn(1, 2048, 1).to(device)
vec_c = torch.randn(1, 8, 8).to(device)
state = torch.zeros(1, 1, 2046).to(device)

traced = torch.jit.trace(model, (wav_x, vec_c, state))
traced.save(TRACED_MODEL)
print("Traced model saved successfully.")
