from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from evaluation.evaluate_predictions import main as evaluate_predictions_main
from evaluation.metrics import classification_metrics
from preprocessing.audio_preprocess import (
    extract_mel_features,
    load_config,
    process_file,
    rms_dbfs,
    standardize_waveform,
)


class PublicCodeTests(unittest.TestCase):
    def test_audio_frontend_matches_documented_shape_and_level(self) -> None:
        config = load_config()
        source_rate = 44100
        time = np.arange(source_rate, dtype=np.float32) / source_rate
        source = 0.08 * np.sin(2.0 * np.pi * 440.0 * time)
        standardized = standardize_waveform(source, source_rate, config)
        features = extract_mel_features(standardized, config)

        self.assertEqual(standardized.shape, (48000,))
        self.assertAlmostEqual(rms_dbfs(standardized), -20.0, places=2)
        self.assertEqual(features.shape, (1, 80, 301))
        self.assertTrue(np.isfinite(features).all())

    def test_metrics_use_fixed_class_order_and_include_absent_class(self) -> None:
        result = classification_metrics(
            ["cough", "feeding", "normal"],
            ["cough", "cough", "normal"],
            ["normal", "feeding", "cough", "scream", "oestrus"],
        )

        self.assertAlmostEqual(result["accuracy"], 2.0 / 3.0)
        self.assertEqual(result["class_order"], ["normal", "feeding", "cough", "scream", "oestrus"])
        self.assertEqual(result["per_class"][4]["support"], 0)
        self.assertEqual(len(result["confusion_matrix"]), 5)

    def test_audio_preprocessor_rejects_non_one_second_clips(self) -> None:
        config = load_config()
        with self.assertRaisesRegex(ValueError, "cropping, and padding are not part"):
            standardize_waveform(np.zeros(24000, dtype=np.float32), 48000, config)

    def test_metrics_reject_labels_outside_declared_class_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "not listed in class_labels"):
            classification_metrics(["unknown"], ["cough"], ["cough", "feeding"])

    def test_preprocessing_writes_audio_and_features(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_rate = 48000
            time = np.arange(source_rate, dtype=np.float32) / source_rate
            source = (0.05 * np.sin(2.0 * np.pi * 330.0 * time).astype(np.float32))[:, None]
            input_path = root / "input.wav"
            sf.write(input_path, source, source_rate, subtype="FLOAT")

            output_wav = root / "out" / "standardized.wav"
            output_mel = root / "out" / "features.npy"
            waveform, features = process_file(input_path, output_wav, output_mel, config)

            saved_waveform, saved_rate = sf.read(output_wav, dtype="float32")
            saved_features = np.load(output_mel)
            self.assertEqual(saved_rate, 48000)
            self.assertEqual(saved_waveform.shape, (48000,))
            self.assertEqual(saved_features.shape, (1, 80, 301))
            self.assertTrue(np.allclose(features, saved_features))
            self.assertTrue(np.allclose(waveform, saved_waveform))

    def test_evaluation_cli_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_csv = root / "predictions.csv"
            output_json = root / "metrics.json"
            with input_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["true_label", "predicted_label"])
                writer.writeheader()
                writer.writerow({"true_label": "cough", "predicted_label": "cough"})
                writer.writerow({"true_label": "feeding", "predicted_label": "cough"})

            argv = [
                "evaluate_predictions",
                "--input-csv",
                str(input_csv),
                "--labels",
                "cough",
                "feeding",
                "--output-json",
                str(output_json),
            ]
            with patch.object(sys, "argv", argv):
                evaluate_predictions_main()

            result = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertAlmostEqual(result["accuracy"], 0.5)
            self.assertEqual(result["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
