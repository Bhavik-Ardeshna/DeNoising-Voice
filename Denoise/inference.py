import argparse
import os
import sys

import toml
import torch
import time
import math
import numpy as np
from fullsubnet_plus import FullSubNet_Plus
from audio_zen.utils import prepare_device


def decompress_cIRM(mask, K=10, limit=9.9):
    mask = limit * (mask >= limit) - limit * (mask <= -limit) + mask * (torch.abs(mask) < limit)
    mask = -K * torch.log((K - mask) / (K + mask))
    return mask

def stft(y, n_fft=512, hop_length=256, win_length=512):
    """
    Args:
        y: [B, F, T]
        n_fft:
        hop_length:
        win_length:
        device:

    Returns:
        [B, F, T], **complex-valued** STFT coefficients

    """
    assert y.dim() == 2
    return torch.stft(
        y,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(y.device),
        return_complex=True
    )

def istft(features, n_fft=512, hop_length=256, win_length=512, length=None, use_mag_phase=False):
    """
    Wrapper for the official torch.istft

    Args:
        features: [B, F, T, 2] (complex) or ([B, F, T], [B, F, T]) (mag and phase)
        n_fft:
        hop_length:
        win_length:
        device:
        length:
        use_mag_phase: use mag and phase as inputs of iSTFT

    Returns:
        [B, T]
    """
    if use_mag_phase:
        # (mag, phase) or [mag, phase]
        assert isinstance(features, tuple) or isinstance(features, list)
        mag, phase = features
        features = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=-1)

    return torch.istft(
        features,
        n_fft,
        hop_length,
        win_length,
        window=torch.hann_window(n_fft).to(features.device),
        length=length
    )

def mag_phase(complex_tensor):
    return torch.abs(complex_tensor), torch.angle(complex_tensor)

def mag_complex_full_band_crm_mask(noisy):
        noisy_complex = stft(noisy)
        noisy_mag, _ = mag_phase(noisy_complex)

        noisy_mag = noisy_mag.unsqueeze(1)
        noisy_real = (noisy_complex.real).unsqueeze(1)
        noisy_imag = (noisy_complex.imag).unsqueeze(1)

        device = prepare_device(torch.cuda.device_count())
        
        model = FullSubNet_Plus(sb_num_neighbors = 15,
                            fb_num_neighbors = 0,
                            num_freqs = 257,
                            look_ahead = 2,
                            sequence_model = "LSTM",
                            fb_output_activate_function = "ReLU",
                            sb_output_activate_function = False,
                            channel_attention_model = "TSSE",
                            fb_model_hidden_size = 512,
                            sb_model_hidden_size = 384,
                            weight_init = False,
                            norm_type = "offline_laplace_norm",
                            num_groups_in_drop_band = 2,
                            kersize=[3, 5, 10],
                            subband_num = 1
        )
        model.load_state_dict(torch.load('./best_model.tar'))
        model.to(device)
        model.eval()

        t1 = time.time()
        pred_crm = model(noisy_mag, noisy_real, noisy_imag)
        t2 = time.time()
        pred_crm = pred_crm.permute(0, 2, 3, 1)

        pred_crm = decompress_cIRM(pred_crm)
        enhanced_real = pred_crm[..., 0] * noisy_complex.real - pred_crm[..., 1] * noisy_complex.imag
        enhanced_imag = pred_crm[..., 1] * noisy_complex.real + pred_crm[..., 0] * noisy_complex.imag
        enhanced_complex = torch.stack((enhanced_real, enhanced_imag), dim=-1)
        enhanced = istft(enhanced_complex, length=noisy.size(-1))
        enhanced = enhanced.detach().squeeze(0).cpu().numpy()

        acoustic_sr = 16000
        rtf = (t2 - t1) / (len(enhanced) * 1.0 / acoustic_sr)
        print(f"model rtf: {rtf}")

        return enhanced



