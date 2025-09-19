# Vitalight - rPPG Heart Rate Detection

A system that detects heart rate from facial video using subtle color changes in skin caused by blood circulation.

---

# Heart Rate Estimation Flowchart

```mermaid
flowchart TD
A[Video Input] --> B[Face Detection<br>(OpenCV / MediaPipe)]
B --> C[ROI Selection<br>(Forehead, Cheeks)]
C --> D[Signal Extraction<br>(RGB Time Series)]
D --> E[Signal Processing<br>(Filtering, Detrending)]
E --> F[Heart Rate Estimation<br>(FFT, Peaks, Autocorrelation)]
F --> G[Output: Heart Rate (BPM) + Visualization]
```

---

## Core Pipeline

1. **Face Detection** → Extract face region from video frames
2. **ROI Selection** → Select skin regions (forehead, cheeks)
3. **Signal Extraction** → Extract RGB time series from selected regions
4. **Signal Processing** → Apply filtering and noise reduction
5. **Heart Rate Estimation** → Use spectral analysis to find heart rate

---

## Core Libraries

- **OpenCV**: Video processing, face detection
- **MediaPipe**: Advanced face landmarks detection
- **NumPy**: Numerical computations
- **SciPy**: Signal processing (filters, FFT)
- **Scikit-learn**: ICA decomposition
- **Matplotlib**: Visualization and debugging
- **TensorFlow/Keras**: For potential deep learning models
- **Streamlit**: For web app deployment

---

## Project Phases

### Phase 1: Initial Implementation

- **Dataset Understanding**: Explored UBFC-2 dataset format
- **Face Detection**: Implemented using OpenCV
- **ROI Extraction**: Extracted signals from the forehead region
- **Basic Signal Processing**: Converted RGB values to time series
- **Heart Rate Estimation**: Used FFT to estimate heart rate

**Challenges & Notes:**

- Worked initially on Kaggle; resolved attribute errors.
- Adjusted code to handle missing `hr` in `ground_truth.txt`.
- Early results were poor:
  - Estimated Heart Rate: 107.1 BPM
  - Confidence: 0.015
  - Comparison with Ground Truth:
  - Average Ground Truth HR: 36.4 BPM
  - Estimated HR: 107.1 BPM
  - Error: 70.8 BPM
  - Relative Error: 194.7%
- Decided to switch to VS Code for development.

---

### Phase 1.5: Improvements

- ✅ Multi-ROI signal extraction
- ✅ Advanced filtering (detrending, normalization)
- ✅ Multiple estimation methods (FFT, peaks, autocorrelation)
- ✅ Confidence-weighted combination
- ✅ Better ground truth processing
- ✅ Comprehensive visualization
- ✅ Robust error handling

_Current status: actively working here._

---

### Phase 2: Advanced Signal Processing

**Duration:** 2-3 weeks

**Goals:**

- Implement CHROM algorithm
- Apply sophisticated filtering
- Significantly improve accuracy

**Key Components:**

- **CHROM Algorithm**: Advanced color-change detection
- **Adaptive Filtering**: Bandpass filters, moving average, detrending
- **Multi-ROI Processing**: Combine signals from multiple face regions
- **Signal Quality Assessment**: Detect and handle poor quality signals
- **Temporal Consistency**: Smooth heart rate estimates over time

**Expected Output:** 75–85% accuracy

---

### Phase 3: Machine Learning Enhancement

**Duration:** 2-3 weeks

**Goals:**

- Train ML models on Kaggle datasets
- Integrate deep learning components
- Achieve research-level accuracy

**Key Components:**

- **Data Collection**: UBFC-rPPG, PURE, etc.
- **Feature Engineering**: Frequency domain and statistical features
- **Deep Learning Models**:
  - CNN for ROI selection optimization
  - LSTM/GRU for temporal modeling
  - Attention mechanisms for adaptive region weighting
- **Model Training on Kaggle**: Leverage free GPU
- **Model Optimization**: Quantization, pruning for deployment

**Expected Output:** ~90%+ accuracy

---

### Phase 4: Real-time Implementation & Web App

**Duration:** 1-2 weeks

**Goals:**

- Real-time processing pipeline
- User-friendly web interface
- Deploy final application

**Key Components:**

- **Real-time Optimization**: Frame skipping, efficient processing
- **Streamlit Web App**: Clean, intuitive interface
- **Model Integration**: Seamless local model loading
- **Visualization**: Real-time plots and heart rate history
- **Error Handling**: Robust error management and user feedback

---

## VS Code Setup Instructions

1. **Create Python environment:**

````bash
conda create -n rppg python=3.11
conda activate rppg

2. **Install dependencies:**
```bash
pip install -r requirements.txt
````
