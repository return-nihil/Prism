import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal


class DC_PreEmph(torch.nn.Module): # from GreyBoxDRC

    def __init__(self, R=0.995):
        super().__init__()

        _, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=2000)
        ir = ir[0][:, 0]

        self.zPad = len(ir) - 1
        self.pars = torch.flipud(torch.tensor(ir, requires_grad=False, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
        
    def forward(self, output: torch.tensor, target: torch.tensor):
        if output.ndim == 2:
            output = output.unsqueeze(1)
            target = target.unsqueeze(1)

        output = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), output), dim=2)
        target = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), target), dim=2)

        output = torch.nn.functional.conv1d(output, self.pars.type_as(output), bias=None)
        target = torch.nn.functional.conv1d(target, self.pars.type_as(output), bias=None)

        return output.squeeze(1), target.squeeze(1)



class MRSTFT_Loss(nn.Module):
    def __init__(self, 
                 scales=[1024, 512, 256, 128, 64], 
                 overlap=0.5, 
                 emphasis=True):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        self.num_scales = len(self.scales)
        self.emphasis = emphasis
        self.pre_emph = DC_PreEmph() if emphasis else None

        self.windows = nn.ParameterList([
            nn.Parameter(torch.from_numpy(np.hanning(scale).astype(np.float32)), requires_grad=False)
            for scale in self.scales
        ])

    def _magnitude(self, stft: torch.Tensor) -> torch.Tensor:
        real, imag = stft[..., 0], stft[..., 1]
        return real**2 + imag**2

    def _stft_mag(self, x, scale, window):
        stft_complex = torch.stft(
            x,
            n_fft=scale,
            hop_length=int((1 - self.overlap) * scale),
            window=window,
            center=False,
            return_complex=True 
        )
        stft = torch.view_as_real(stft_complex)

        return self._magnitude(stft)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        lin_loss = 0.0
        log_loss = 0.0

        if self.emphasis:
            pred, target = self.pre_emph(pred, target)

        for i, scale in enumerate(self.scales):
            pred_mag = self._stft_mag(pred, scale, self.windows[i])
            target_mag = self._stft_mag(target, scale, self.windows[i])

            lin_loss += F.l1_loss(pred_mag, target_mag)
            log_loss += F.l1_loss(torch.log(pred_mag + 1e-4), torch.log(target_mag + 1e-4))

        return (lin_loss + log_loss) / self.num_scales


class KL_Loss(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class VAE_Train_Loss(nn.Module):
    def __init__(self, 
                 device,
                 recon_weigth=0.05,
                 stft_weight=1, 
                 kl_weight=0.0025):
        super().__init__()
        self.recon_mse = nn.MSELoss(reduction='sum')
        self.recon_huber = nn.HuberLoss(reduction='sum')
        self.stft_loss = MRSTFT_Loss().to(device)
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
                 device,
                 recon_weigth=1,
                 stft_weight=1):
        super().__init__()
        self.recon_mse = nn.MSELoss(reduction='sum')
        self.recon_huber = nn.HuberLoss(reduction='sum')
        self.stft_loss = MRSTFT_Loss().to(device)
        self.recon_weight = recon_weigth
        self.stft_weight = stft_weight

    def forward(self, target_audio, predicted_audio):
        mse = self.recon_mse(target_audio, predicted_audio)
        huber = self.recon_huber(target_audio, predicted_audio)
        recon = mse + huber * self.recon_weight
        stft = self.stft_loss(target_audio, predicted_audio) * self.stft_weight
        l1 = F.l1_loss(target_audio, predicted_audio)

        return recon, stft, l1
    


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion_train = VAE_Train_Loss(device)
    criterion_test = VAE_Valid_Loss(device)
    pred = torch.randn(2, 1, 64000).to(device)
    target = torch.randn(2, 1, 64000).to(device)
    mu = torch.randn(2, 8).to(device)
    logvar = torch.randn(2, 8).to(device)

    total_loss, recon_loss, stft_loss, kl_loss = criterion_train(target, pred, mu, logvar)
    print(f"TRAIN: Total Loss: {total_loss.item()}, Recon Loss: {recon_loss.item()}, STFT Loss: {stft_loss.item()}, KL Loss: {kl_loss.item()}")
    recon_loss_val, stft_loss_val, l1_loss_val = criterion_test(target, pred)
    print(f"TEST: Recon Loss: {recon_loss_val.item()}, STFT Loss: {stft_loss_val.item()}, L1 Loss: {l1_loss_val.item()}")