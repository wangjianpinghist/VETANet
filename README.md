# VETANet Supporting Code

This repository provides supporting code associated with the VETANet study, including the one-second acoustic feature frontend, performance-evaluation utilities, configuration files, and validation tests. The complete model-training implementation and selected model components are not part of this release.


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
