import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)


class FiLM_Cond(nn.Module):
    def __init__(self, cond_dim, channels_per_block, shared=96):
        super().__init__()
        self.channels_per_block = list(channels_per_block)
        self.shared = nn.Sequential(
            nn.Linear(cond_dim, shared),
            nn.SiLU(inplace=True),
            nn.Linear(shared, shared),
            nn.SiLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(shared, ch * 2) for ch in self.channels_per_block])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, c):
        z = self.shared(c)
        gammas, betas = [], []
        for head, ch in zip(self.heads, self.channels_per_block):
            out = head(z).view(c.size(0), ch, 2)
            gammas.append(out[..., 0:1])
            betas.append(out[..., 1:2])
        return gammas, betas



class Band_Cond(nn.Module):
    def __init__(self, n_bands=16, cond_dim=64, latent_dim=8, band_emb_dim=8, hidden=128):
        super().__init__()
        self.n_bands = n_bands
        self.latent_dim = latent_dim
        self.band_emb = nn.Embedding(n_bands, band_emb_dim)

        self.band_mlp = nn.Sequential(
            nn.Linear(latent_dim + band_emb_dim, hidden),
            nn.PReLU(num_parameters=1),
            nn.Linear(hidden, hidden),
            nn.PReLU(num_parameters=1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden * n_bands, hidden),
            nn.PReLU(num_parameters=1),
            nn.Linear(hidden, cond_dim),
            nn.LayerNorm(cond_dim)
        )

        nn.init.normal_(self.band_emb.weight, std=0.02)

    def forward(self, c):
        B, _, _ = c.shape
        band_idx = torch.arange(self.n_bands, device=c.device)
        band_e = self.band_emb(band_idx).unsqueeze(0).expand(B, -1, -1)
        band_feat = torch.cat([c, band_e], dim=-1)
        band_feat = self.band_mlp(band_feat)
        flat = band_feat.reshape(B, -1)
        return self.fc(flat)


class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        r = max(1, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, r, 1),
            nn.PReLU(num_parameters=r),
            nn.Conv1d(r, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)


class Residual_Block(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, expansion=1.0, se_reduction=8, use_weight_norm=False, gn_groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.expanded_ch = int(round(in_channels * expansion))
        self.kernel_size = kernel_size
        self.dilation = dilation
    
        if self.expanded_ch != in_channels:
            conv_expand = nn.Conv1d(in_channels, self.expanded_ch, 1)
            self.expand = nn.utils.weight_norm(conv_expand) if use_weight_norm else conv_expand
        else:
            self.expand = None

        self.pointwise_in = nn.Conv1d(self.expanded_ch, self.expanded_ch, 1)
        self.depthwise = nn.Conv1d(self.expanded_ch, self.expanded_ch,
                                   kernel_size=kernel_size, dilation=dilation, groups=self.expanded_ch, padding=0)
        groups = min(gn_groups, self.expanded_ch)
        while self.expanded_ch % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, self.expanded_ch)

        self.pointwise_out = nn.Conv1d(self.expanded_ch, in_channels, 1)
        self.skip = nn.Conv1d(in_channels, in_channels, 1)
        self.se = SE_Block(self.expanded_ch, reduction=se_reduction) if se_reduction else None

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gamma=None, beta=None):
        residual = x
        out = x
        if self.expand is not None:
            out = self.expand(out)
        out = self.pointwise_in(out)
        pad = (self.kernel_size - 1) * self.dilation
        if pad > 0:
            out = F.pad(out, (pad, 0))

        out = self.depthwise(out)
        out = self.norm(out)
        out = F.silu(out)
        out = gamma * out + beta
        if self.se is not None:
            out = self.se(out)
        out = self.pointwise_out(out)
        res = residual + 0.5 * out
        skip = self.skip(out)
        return res, skip
    

class WN_TCN(nn.Module):
    def __init__(self,
                 inp_channel=1,
                 out_channel=1,
                 channels=64,  
                 kernel_size=3,
                 n_blocks=9, 
                 cond_dim=128,
                 sample_rate=48000,
                 n_bands=8,
                 use_weight_norm=True,
                 gn_groups=8,
                 se_reduction=8,
                 latent_dim=8,
                 band_hidden=128,
                 expansion_blocks=None):
        super().__init__()
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.base_channels = channels
        self.kernel_size = kernel_size
        self.n_blocks = n_blocks
        self.cond_dim = cond_dim
        self.sample_rate = sample_rate
        self.n_bands = n_bands
        self.latent_dim = latent_dim
        self.band_hidden = band_hidden

        mid = max(16, channels // 3)
        self.input_stem = nn.Sequential(
            nn.Conv1d(inp_channel, mid, kernel_size=kernel_size, padding=(kernel_size//2)),
            nn.PReLU(num_parameters=mid),
            nn.Conv1d(mid, channels, kernel_size=1)
        )

        expansion_blocks = expansion_blocks or {}
        default_exp = {i: 1.5 if i < 3 else (1.25 if i < 5 else 1.0) for i in range(n_blocks)}
        for k, v in default_exp.items():
            expansion_blocks.setdefault(k, v)

        self.blocks = nn.ModuleList()
        channel_schedule = []
        for i in range(n_blocks):
            exp = float(expansion_blocks.get(i, 1.0))
            block = Residual_Block(in_channels=channels,
                                       kernel_size=kernel_size,
                                       dilation=2**i,
                                       expansion=exp,
                                       se_reduction=se_reduction,
                                       use_weight_norm=use_weight_norm,
                                       gn_groups=gn_groups)
            self.blocks.append(block)
            channel_schedule.append(block.expanded_ch)

        self.film = FiLM_Cond(cond_dim=cond_dim, channels_per_block=channel_schedule, shared=96)
        self.band_cond = Band_Cond(n_bands=n_bands, cond_dim=cond_dim, latent_dim=self.latent_dim, band_emb_dim=8, hidden=self.band_hidden)

        final_conv1 = nn.Conv1d(channels, channels, 1)
        final_conv2 = nn.Conv1d(channels, out_channel, 1)
        self.final = nn.Sequential(nn.SiLU(inplace=True), final_conv1, nn.SiLU(inplace=True), final_conv2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_receptive_field(self):
        rf = 1
        for i in range(self.n_blocks):
            rf += (self.kernel_size - 1) * (2 ** i)
        rf_ms = (rf / float(self.sample_rate)) * 1000.0
        return rf, rf_ms


    def compute_num_of_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, c, state=None):

        #start = time.time()
        
        x = x.permute(0, 2, 1)
        B, C_in, _ = x.shape

        rf, _ = self.compute_receptive_field()
        pad_needed = rf - 1

        if state is None:
            state = x.new_zeros(B, C_in, pad_needed)
        full_in = torch.cat([state, x], dim=-1)
        h = self.input_stem(full_in)

        cond_vec = self.band_cond(c)
        gammas, betas = self.film(cond_vec)

        sum_skips = h.new_zeros(B, self.base_channels, h.size(-1))
        for i, block in enumerate(self.blocks):
            h, skip = block(h, gamma=gammas[i], beta=betas[i])
            sum_skips.add_(skip)

        out = self.final(sum_skips)
        out = out[:, :, pad_needed:].permute(0, 2, 1)
        new_state = full_in[:, :, -pad_needed:].contiguous()

        #end = time.time()
        #print(f"Forward: {end - start:.4f} sec for {T_block} time steps")

        return out, new_state



def print_model_param_summary(model):
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    total = count_params(model)

    tcn_params = count_params(model.input_stem) + sum(count_params(block) for block in model.blocks) + count_params(model.final)

    print("="*60)
    print(f"Total trainable parameters: {total:,}")
    print(f"TCN (InputStem + Residual Blocks + Final) params: {tcn_params:,}")
    print("="*60)
    print(f"[Input Stem]        : {count_params(model.input_stem):,}")
    print(f"[Band Conditioning] : {count_params(model.band_cond):,}")
    print(f"[FiLM Cond]         : {count_params(model.film):,}")
    print(f"[Final Conv Stack]  : {count_params(model.final):,}")
    print("\n[Residual Blocks]")
    for i, block in enumerate(model.blocks):
        blk = count_params(block)
        print(f"  Block {i:02d}: {blk:,} params  | expanded_ch={block.expanded_ch}")
        sub = {}
        if hasattr(block, 'expand') and block.expand is not None:
            sub['expand'] = count_params(block.expand)
        sub['pointwise_in'] = count_params(block.pointwise_in)
        sub['depthwise'] = count_params(block.depthwise)
        sub['norm'] = count_params(block.norm)
        sub['pointwise_out'] = count_params(block.pointwise_out)
        sub['skip'] = count_params(block.skip)
        if hasattr(block, 'se') and block.se is not None:
            sub['se'] = count_params(block.se)
        for k, v in sub.items():
            print(f"      {k:<15} : {v:,}")
    print("="*60)



if __name__ == "__main__":
    BANDS=16
    model = WN_TCN(inp_channel=1, 
                      out_channel=1, 
                      channels=64, 
                      n_blocks=9, 
                      cond_dim=128, 
                      sample_rate=48000, 
                      n_bands=BANDS, 
                      band_hidden=128)
    model.eval()
    print(f'NUMBER OF PARAMS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}') 
    input = torch.randn(2, 2048, 1) 
    cond = torch.randn(2, BANDS, 8) 
    out, new_state = model(input, cond, None)
    print_model_param_summary(model)
    rf, rf_ms = model.compute_receptive_field()
    print(f"Receptive field: {rf} samples ({rf_ms:.2f} ms)")





