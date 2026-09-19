# VitaLight — rPPG Heart Rate Detection

A Python tool that estimates heart rate from facial video by analyzing subtle, pulse-driven color changes in the skin (remote photoplethysmography — rPPG).

## Overview

Given a video of a person's face, VitaLight extracts a physiological signal from skin color changes and estimates heart rate (BPM) without any physical sensor contact. It was developed and validated against the [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg) dataset, comparing estimated heart rate against each subject's recorded ground truth.

## How It Works

1. **Face detection** — Haar Cascade classifier (OpenCV) locates the face in each frame.
2. **ROI selection** — Three regions are extracted per frame: forehead, left cheek, right cheek.
3. **Signal quality assessment** — Each ROI is scored on brightness, contrast, and size; low-quality regions are down-weighted or discarded.
4. **Signal extraction (CHROM)** — RGB values from the ROIs are combined (quality-weighted) and transformed using the CHROM algorithm, which is more robust to illumination changes than raw RGB signals.
5. **Temporal filtering** — Detrending, moving-average smoothing, and a Butterworth bandpass filter (restricted to the 50–180 BPM frequency range) clean the signal.
6. **ICA refinement** — When multiple ROI channels are available, Independent Component Analysis (`scikit-learn`'s FastICA) is used to help separate the pulse signal from noise.
7. **Heart rate estimation** — Four independent methods (FFT peak, time-domain peak detection, autocorrelation, Welch power spectral density) each produce a BPM estimate with a confidence score; the final result is a confidence-weighted average.
8. **Ground truth comparison** — For UBFC subjects, the tool parses `ground_truth.txt` and reports absolute/relative error against the estimate.

## Tech Stack

- **OpenCV** — video I/O, Haar Cascade face detection
- **NumPy / SciPy** — signal processing (FFT, Butterworth filtering, Welch PSD, peak detection)
- **scikit-learn** — ICA (FastICA) for signal separation
- **Matplotlib** — visualization of raw/filtered signals, frequency spectrum, and per-method comparison
- **pandas / Pillow** — supporting utilities

## Setup

```bash
pip install -r requirements.txt
```

Requires a local copy of the UBFC-rPPG dataset, structured as `subjectXX/vid.avi` + `subjectXX/ground_truth.txt`.

## Usage

The `demo_rppg()` entry point currently points to a hardcoded local dataset path and is meant to be run as a script during development, not as a packaged CLI tool:

```bash
python vitalight.py
```

Update the `dataset_path` variable in `demo_rppg()` to point to your local UBFC-rPPG directory before running.

## Current Status

The project reached its original objective: extracting a usable heart-rate estimate from facial video via rPPG, with a working multi-method (FFT/peak/autocorrelation/Welch), quality-weighted, ICA-assisted pipeline, and a ground-truth comparison process against the UBFC dataset. Development proceeded iteratively (a face-detection + single-ROI baseline, then multi-ROI + CHROM + ICA), with relative error against ground truth dropping from ~195% to ~6.5% across iterations.

This is not a packaged or deployed application — it's a single-script research/experimentation tool. No web interface, real-time processing, or ML-based extensions have been built. Development has been paused at this point; no further extensions are currently planned.

For the full iteration-by-iteration development history — including intermediate results, what changed at each step, and ideas that were explored but not implemented — see [DEVELOPMENT_LOG.md](./DEVELOPMENT_LOG.md).

## Known Limitations

- No automated tests
- Hardcoded local file path in the demo entry point (not portable out of the box)
- Single-file structure — no separation between library code and the demo script
- Face detection relies on Haar Cascade, which is less robust to pose/lighting variation than landmark-based detectors
- Accuracy has only been validated against the UBFC-rPPG dataset

## Lessons Learned

- Naive single-ROI RGB extraction with FFT alone produced highly unreliable estimates; robustness improved substantially once multi-ROI signal fusion, the CHROM transform, and confidence-weighted combination of multiple estimation methods (FFT, peak, autocorrelation, Welch) were introduced together.
- Ground truth files in the UBFC dataset needed careful format handling (BVP signal, HR values, and timestamps on separate lines) — a naive single-line parser silently produced wrong values.
