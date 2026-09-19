# VitaLight — Development Log

This document preserves the full iteration-by-iteration development history of VitaLight, including intermediate results and what was learned at each stage. For a quick project overview, see [README.md](./README.md).

The project was developed using a **Spiral Model**: each cycle consists of Planning → Risk Analysis/Research → Development/Prototyping → Evaluation, with the next cycle informed by the previous one's results.

---

## Iteration 1: Initial Implementation

**Planning**
- Explore the UBFC-2 dataset format
- Implement face detection with OpenCV
- Extract signals from the forehead region
- Convert RGB values to a time series
- Estimate HR using FFT

**Results**

<img width="1536" height="802" alt="phase1_results" src="https://github.com/user-attachments/assets/ad9f018f-0ff1-40f1-974a-90e8cc58148b" />

- Estimated Heart Rate: 107.1 BPM
- Confidence: 0.015
- Average Ground Truth HR: 36.4 BPM
- Absolute Error: 70.8 BPM
- Relative Error: 194.7%

**Evaluation**
- Very large error. The single-ROI, FFT-only approach was not sufficient.
- Decided to refine preprocessing and ROI strategy, and moved development to VS Code.

---

## Iteration 1.5: Improvements

**Changes**
- Multi-ROI signal extraction (forehead + both cheeks)
- Advanced filtering (detrending, normalization)
- Multiple estimation methods (FFT, peaks, autocorrelation)
- Confidence-weighted combination of methods
- Comprehensive visualization
- More robust error handling

**Results**
- Combined Estimate: 103.3 BPM (Confidence: 0.584)
- Method breakdown:
  - FFT → 89.0 BPM (confidence 0.021, unreliable)
  - Peaks → 112.4 BPM (confidence 0.855, most stable)
  - Autocorrelation → 78.3 BPM (confidence 0.313, partially consistent)
- Ground truth BVP signal could not be reliably parsed at this stage, so no direct error comparison was available yet.

**Evaluation**
- More stable results across methods, but overall accuracy remained limited and unverifiable without proper ground truth parsing.
- Confidence-weighted fusion of multiple methods improved stability compared to any single method.
- Next step: fix ground-truth parsing and integrate a more robust signal construction algorithm (CHROM).

---

## Iteration 2: Advanced Signal Processing

**Planning**
- Integrate the CHROM algorithm
- Improve filtering and temporal stability
- Reduce error margin to an acceptable level

**Key components added**
- CHROM algorithm for robust, illumination-resistant signal construction
- Adaptive filtering: bandpass, detrending, moving average
- Multi-ROI processing with quality-weighted signal fusion
- Signal quality assessment to reject noisy data
- Temporal consistency smoothing
- ICA-based signal separation (FastICA) as an additional refinement step
- Fixed ground-truth parsing: correctly handling the UBFC 3-line format (BVP, HR, timestamps), with fallback to BVP-derived HR estimation when the HR line is missing
- Fixed a `filtfilt` "padlen" error that occurred on short signals

**Results**

<img width="1536" height="802" alt="phase_2_results" src="https://github.com/user-attachments/assets/0c278dbe-f966-4d55-9972-a8ca3bab79ca" />        
    
<img width="1602" height="895" alt="Ekran görüntüsü 2025-09-27 140028" src="https://github.com/user-attachments/assets/cac53504-3c00-4fbb-b39b-271061cd5bb5" />

- Estimated HR: 95.3 BPM vs. Ground Truth: 102.0 BPM
- Relative Error: 6.5%

**Evaluation**
- Accuracy improved substantially (from ~195% relative error in Iteration 1 to ~6.5% here), driven mainly by the CHROM transform, quality-weighted multi-ROI fusion, and fixed ground-truth handling.
- Literature review at this stage (comparing eight rPPG methods — POS, LGI, CHROM, OMIT, GREEN, ICA, PCA, PBV) suggested POS is particularly robust under motion and variable lighting.
- **Note:** based on this review, combining POS with CHROM and GREEN was considered as a next step (see Iteration 3 below), but this was not carried through to implementation — the codebase implements CHROM only, not POS or GREEN.

This iteration is where the project's core objective — extracting a usable heart-rate estimate from facial video via rPPG, validated against ground truth — was achieved.

---

## Iteration 3: Hybrid Methods & Optimization *(planned, not implemented)*

**Original planning**
- Combine multiple methods (POS, CHROM, GREEN) for robustness
- Optimize preprocessing and temporal stability
- Benchmark a hybrid pipeline against individual algorithms

**Status:** This iteration was planned following the Iteration 2 literature review, but was not carried out. The codebase does not contain POS or GREEN implementations, and no benchmarking results exist for a hybrid approach. Development was paused after Iteration 2 reached the project's original goal; this iteration remains an idea for future work rather than completed work.

---

## Iteration 4: Machine Learning Enhancement *(planned, not implemented)*

**Original planning**
- Integrate deep learning for better generalization and temporal modeling
- Ideas explored on paper: CNN for ROI selection, LSTM/GRU for temporal modeling, attention mechanisms for adaptive region weighting, training on UBFC-rPPG/PURE datasets

**Status:** Not implemented. No deep learning code exists in the repository (no TensorFlow/Keras usage). Listed here only as a documented direction that was considered but not pursued.

---

## Iteration 5: Real-time Implementation & Web App *(planned, not implemented)*

**Original planning**
- Move from offline research script to a real-time application
- Ideas explored on paper: frame-skipping for real-time performance, a Streamlit-based web interface, live visualization of heart rate history

**Status:** Not implemented. No real-time processing or web app code exists in the repository (no Streamlit usage). Listed here only as a documented direction that was considered but not pursued.

---

## Summary

- The Spiral Model suited this project's research-oriented, risk-driven nature: each iteration tested a hypothesis and fed its result into the next.
- The project's constant goal — extract a signal from facial video and reliably estimate heart rate, validated against ground truth — was reached at Iteration 2, with relative error dropping from ~195% to ~6.5% across iterations.
- Iterations 3–5 remain documented as considered future directions rather than completed work, and are not reflected in the current codebase.
