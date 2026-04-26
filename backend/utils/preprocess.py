"""
backend/utils/preprocess.py
Shared preprocessing helpers — used by all model modules.
"""
import io
import torch
import torchaudio


TARGET_SR = 22_050


def load_audio(file_bytes: bytes) -> tuple[torch.Tensor, int]:
    """Decode audio bytes → (waveform, sample_rate)."""
    try:
        waveform, sr = torchaudio.load(io.BytesIO(file_bytes))
        return waveform, sr
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}")


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel audio to mono by averaging channels."""
    if waveform.shape[0] > 1:
        return waveform.mean(dim=0, keepdim=True)
    return waveform


def resample(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    """Resample waveform to target_sr if necessary."""
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def z_score(tensor: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalisation (mirrors sklearn StandardScaler)."""
    mean = tensor.mean()
    std  = tensor.std(unbiased=False) + 1e-8
    return (tensor - mean) / std