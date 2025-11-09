
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
    def __init__(self, scales=(1024, 512, 256, 128, 64),
                 overlap=0.75, emphasis=True):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        self.emphasis = emphasis
        self.pre_emph = DC_PreEmph() if emphasis else None

        self.windows = nn.ParameterList([
            nn.Parameter(torch.from_numpy(np.hanning(scale).astype(np.float32)),
                         requires_grad=False)
            for scale in self.scales
        ])

    def _stft_mag(self, x, scale, window):
        stft = torch.stft(
            x, n_fft=scale,
            hop_length=int((1 - self.overlap) * scale),
            window=window,
            center=False,
            return_complex=True
        )
        return torch.abs(stft) ** 2

    def forward(self, pred, target):
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])

        if self.emphasis:
            pred, target = self.pre_emph(pred, target)

        lin_loss = 0.0
        log_loss = 0.0
        for i, scale in enumerate(self.scales):
            pred_mag = self._stft_mag(pred, scale, self.windows[i])
            target_mag = self._stft_mag(target, scale, self.windows[i])

            lin_loss += F.l1_loss(pred_mag, target_mag)
            log_loss += F.l1_loss(
                torch.log(pred_mag + 1e-4),
                torch.log(target_mag + 1e-4)
            )
        return (lin_loss + log_loss) / len(self.scales)


class ESRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        mse = torch.mean((target - predict) ** 2)
        energy = torch.mean(target ** 2) + self.epsilon
        return mse / energy


class TCN_Loss(nn.Module):
    def __init__(self, 
                 recon_weigth=1.0,
                 stft_weight=1.0):
        super().__init__()
        self.recon_weigth = recon_weigth
        self.stft_weight = stft_weight
        
        self.stft_loss = MRSTFT_Loss()
        self.esr_loss = ESRLoss()

    def forward(self, predict: torch.tensor, target: torch.tensor):

        esr_loss = self.esr_loss(predict, target) * self.recon_weigth
        stft_loss = self.stft_loss(predict, target) * self.stft_weight
        
        return stft_loss, esr_loss
    

