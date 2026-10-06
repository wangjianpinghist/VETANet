# VETANet Supporting Code

This repository provides supporting code associated with the VETANet study, including the one-second acoustic feature frontend, performance-evaluation utilities, configuration files, and validation tests. The complete model-training implementation and selected model components are not part of this release.

## Included

- Standardization and one-second acoustic feature extraction for an already localized WAV clip.
- Accuracy, macro/weighted precision, recall and F1, per-class metrics, and confusion-matrix calculation from saved labels and predictions.
- A parameter file and synthetic unit tests.

## Not included

- Event detection, energy-anchor selection, candidate-interval cropping, or any other event-localization logic.
- The VETANet architecture and its IMTF, GATFI, and CCDC implementations.
- Training code, data splits, source recordings, trained checkpoints, or historical experiments.

The public preprocessing helper requires an already localized one-second clip. It downmixes multichannel input by channel averaging, resamples to 48 kHz with `resampy`'s `kaiser_best` filter when needed, checks that the duration is exactly one second after resampling, and applies RMS normalization to -20 dBFS. It does not crop or pad audio.

The one-second acoustic feature frontend uses a 48 kHz sample rate, a 400-point FFT, a 400-sample Hann window, a 160-sample hop, 80 HTK-scaled Mel bands, and a power Mel spectrogram. It subtracts the per-clip temporal mean. For a one-second input, the returned array has shape `[1, 80, 301]` (`batch, Mel bands, time frames`).

TorchAudio 2.5.1 may warn that some Mel filters are all zero with this configuration; these settings are retained to match the frontend used in the paper's experiments.

## Environment

Use Python 3.9 and install matching PyTorch and TorchAudio builds. The tested environment used PyTorch/TorchAudio 2.5.1 with CUDA 12.1; CPU builds with the same versions are also suitable for these utilities. Install the remaining dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## Preprocessing example

```powershell
python -m preprocessing.audio_preprocess --input path\to\localized_clip.wav --output-wav output\standardized.wav --output-mel output\mel.npy
```

The input WAV must already be a one-second candidate clip. A longer or shorter input is rejected rather than silently cropped or padded.

## Metric evaluation example

Prepare a CSV with exactly these required columns (additional columns are ignored):

```csv
true_label,predicted_label
cough,cough
feeding,normal
```

Then provide the class order explicitly. This example uses `normal`, `feeding`, `cough`, `scream`, `oestrus`:

```powershell
python -m evaluation.evaluate_predictions --input-csv predictions.csv --labels normal feeding cough scream oestrus --output-json metrics.json
```

The example rows above are illustrative only, not study data. The evaluator reports metrics for the supplied predictions; it does not run a model or regenerate predictions.

To summarize metrics saved separately for each cross-validation fold, provide at least two fold result JSON files. The utility reports the mean and population standard deviation (denominator `n`, `ddof=0`) across folds for overall and per-class metrics:

```powershell
python -m evaluation.aggregate_folds --input-json fold1.json fold2.json fold3.json fold4.json fold5.json --output-json cv-summary.json
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests use a generated sine wave and toy labels; they contain no study samples or reported performance values.

## Data and release scope

No source audio, sample identifiers, split manifests, or checkpoints are included. See [`data/README.md`](data/README.md). The complete model and reported experiments cannot be independently reproduced from this folder.

## License

This supporting code is released under the MIT License. See [`LICENSE`](LICENSE) for details.

## Attribution

The broader research workspace builds in part on AudioClassification-Pytorch by yeyupiaoling. This package contains standalone supporting utilities and does not include its training or model modules. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
