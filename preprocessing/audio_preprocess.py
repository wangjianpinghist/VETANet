"""Standardize a pre-localized one-second WAV and extract acoustic Mel features.

This helper does not detect events, choose an anchor, or create the one-second
candidate interval. Its input must already contain one second of audio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import resampy
import soundfile as sf
import torch
import torchaudio
import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "audio" not in config or "mel" not in config:
        raise ValueError("Configuration must define 'audio' and 'mel' sections.")
    return config


def rms_dbfs(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    mean_square = float(np.mean(np.square(samples)))
    if mean_square <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(mean_square))


def standardize_waveform(
    samples: np.ndarray,
    sample_rate: int,
    config: dict,
) -> np.ndarray:
    """Downmix, resample, validate duration, and RMS-normalize a WAV waveform."""
    audio_cfg = config["audio"]
    target_rate = int(audio_cfg["target_sample_rate"])
    expected_duration = float(audio_cfg["expected_duration_seconds"])
    target_dbfs = float(audio_cfg["rms_target_dbfs"])
    max_gain_db = float(audio_cfg.get("max_gain_db", 300.0))

    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    elif waveform.ndim != 1:
        raise ValueError("Audio must be mono or shaped as [samples, channels].")
    if waveform.size == 0 or sample_rate <= 0:
        raise ValueError("Audio must contain samples and have a positive sample rate.")

    if int(sample_rate) != target_rate:
        waveform = resampy.resample(
            waveform,
            int(sample_rate),
            target_rate,
            filter=str(audio_cfg.get("resample_filter", "kaiser_best")),
        ).astype(np.float32, copy=False)

    expected_samples = int(round(expected_duration * target_rate))
    if waveform.size != expected_samples:
        actual_duration = waveform.size / float(target_rate)
        raise ValueError(
            f"Expected a pre-localized {expected_duration:.3f} s clip "
            f"({expected_samples} samples at {target_rate} Hz); got "
            f"{actual_duration:.6f} s ({waveform.size} samples). "
            "Event localization, cropping, and padding are not part of this release."
        )

    current_dbfs = rms_dbfs(waveform)
    if np.isfinite(current_dbfs):
        gain_db = target_dbfs - current_dbfs
        if gain_db > max_gain_db:
            raise ValueError(
                f"Required gain ({gain_db:.2f} dB) exceeds the configured "
                f"maximum ({max_gain_db:.2f} dB)."
            )
        waveform = waveform * np.float32(10.0 ** (gain_db / 20.0))

    return np.asarray(waveform, dtype=np.float32)


def extract_mel_features(waveform: np.ndarray, config: dict) -> np.ndarray:
    """Return mean-centered power-Mel features shaped [1, mel, time]."""
    audio_cfg = config["audio"]
    mel_cfg = config["mel"]
    target_samples = int(
        round(float(audio_cfg["expected_duration_seconds"]) * int(audio_cfg["target_sample_rate"]))
    )
    if waveform.ndim != 1 or waveform.size != target_samples:
        raise ValueError("Pass a mono waveform standardized to the configured duration.")

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=int(audio_cfg["target_sample_rate"]),
        n_fft=int(mel_cfg["n_fft"]),
        win_length=int(mel_cfg["win_length"]),
        hop_length=int(mel_cfg["hop_length"]),
        n_mels=int(mel_cfg["n_mels"]),
        f_min=float(mel_cfg.get("f_min", 0.0)),
        f_max=mel_cfg.get("f_max"),
        power=float(mel_cfg.get("power", 2.0)),
        center=bool(mel_cfg.get("center", True)),
        pad_mode=str(mel_cfg.get("pad_mode", "reflect")),
        normalized=bool(mel_cfg.get("normalized", False)),
        norm=mel_cfg.get("norm"),
        mel_scale=str(mel_cfg.get("mel_scale", "htk")),
    )
    waveform_tensor = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
    mel = mel_transform(waveform_tensor)
    mel = mel - mel.mean(dim=2, keepdim=True)
    return mel.contiguous().cpu().numpy().astype(np.float32, copy=False)


def process_file(
    input_path: str | Path,
    output_wav: str | Path | None,
    output_mel: str | Path | None,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Process one WAV and optionally save standardized audio and Mel features."""
    samples, sample_rate = sf.read(input_path, dtype="float32", always_2d=True)
    waveform = standardize_waveform(samples, int(sample_rate), config)
    features = extract_mel_features(waveform, config)

    if output_wav is not None:
        output_path = Path(output_wav)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, int(config["audio"]["target_sample_rate"]), subtype="FLOAT")
    if output_mel is not None:
        output_path = Path(output_mel)
        if output_path.suffix.lower() != ".npy":
            raise ValueError("Mel feature output must use the .npy extension.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, features)
    return waveform, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input pre-localized one-second WAV.")
    parser.add_argument("--output-wav", help="Optional path for the standardized float WAV.")
    parser.add_argument("--output-mel", help="Optional path for the NumPy Mel feature array.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML preprocessing configuration.")
    args = parser.parse_args()
    if args.output_wav is None and args.output_mel is None:
        parser.error("Specify at least one of --output-wav or --output-mel.")

    config = load_config(args.config)
    waveform, features = process_file(args.input, args.output_wav, args.output_mel, config)
    print(
        f"sample_rate={config['audio']['target_sample_rate']} Hz; "
        f"duration={waveform.size / config['audio']['target_sample_rate']:.3f} s; "
        f"RMS={rms_dbfs(waveform):.2f} dBFS; feature_shape={tuple(features.shape)}"
    )


if __name__ == "__main__":
    main()
