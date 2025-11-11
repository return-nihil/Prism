import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import pyaudio

from TCN import PrismLite


SR = 48000
CHUNK_SIZE = 2048
INPUT_DEVICE_INDEX = 4 # QUESTA E' LA MIA M4
OUTPUT_DEVICE_INDEX = 4  
DEVICE = torch.device("cpu") 

MODEL_PATH = "/Users/ardan/Desktop/PedalinY/PRISM/model_8bands.pth"
GAIN = 1.0   
MIX = 0.5  


print("Initializing Prism model...")
model = PrismLite(
    inp_channel=1,
    out_channel=1,
    channels=64,
    kernel_size=3,
    n_blocks=9,
    cond_dim=128,
    sample_rate=SR,
    n_bands=8,
).to(DEVICE)
model.eval()


assert os.path.exists(MODEL_PATH), f"Checkpoint not found at {MODEL_PATH}"
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=False)
print("Loaded trained Prism weights.")


rf, rf_ms = model.compute_receptive_field()
pad_needed = rf - 1
print(f"Prism receptive field: {rf} samples ({rf_ms:.2f} ms) | pad_needed={pad_needed}")


state = torch.zeros(1, 1, pad_needed, device=DEVICE)
cond = torch.randn(1, model.n_bands, 8, device=DEVICE) 
print("Model ready.\n")


def audio_callback(in_data, frame_count, time_info, status):
    global state

    wav_x = np.frombuffer(in_data, dtype=np.float32)

    wav_x = np.copy(wav_x)

    x_tensor = torch.from_numpy(wav_x).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

    with torch.no_grad():
        y_pred, new_state = model(x_tensor, cond, state)

    state = new_state.detach()

    y_out = y_pred.squeeze().cpu().numpy()

    y_mix = (MIX * y_out + (1 - MIX) * wav_x) * GAIN
    y_mix = np.clip(y_mix, -1.0, 1.0).astype(np.float32)

    return (y_mix.tobytes(), pyaudio.paContinue)

def main():
    print("Entered main()")

    p = pyaudio.PyAudio()

    print("-- Available audio devices --")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        prefix = ''
        if info['maxOutputChannels'] > 0 and info['maxInputChannels'] == 0:
            prefix = 'Out '
        elif info['maxInputChannels'] > 0 and info['maxOutputChannels'] == 0:
            prefix = 'In  '
        elif info['maxInputChannels'] > 0 and info['maxOutputChannels'] > 0:
            prefix = 'I/O '
        print(f"{prefix} Device {i}: {info['name']} (in={info['maxInputChannels']} out={info['maxOutputChannels']})")

    print(f"Using SR={SR}, CHUNK_SIZE={CHUNK_SIZE}, DEVICE={DEVICE}")
    print("Streaming audio in real time... Press Ctrl+C to stop.\n")

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SR,
        input=True,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=INPUT_DEVICE_INDEX,
        output_device_index=OUTPUT_DEVICE_INDEX,
        stream_callback=audio_callback,
    )

    stream.start_stream()

    try:
        while stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping stream...")

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("Audio streaming stopped cleanly.")

if __name__ == "__main__":
    main()
