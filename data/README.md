# Data policy

This supporting-code package contains no audio recordings, labels, participant or farm metadata, or train/validation/test manifests.

`preprocessing/audio_preprocess.py` accepts only a pre-localized one-second WAV clip. It does not perform event detection, anchor selection, or event-centered cropping. Users must provide audio they are authorized to process. This release does not reconstruct the study's private data splits or reproduce the reported model results.

This package does not include split manifests or sample identifiers and does not reconstruct the study's private data splits.
