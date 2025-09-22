# VitaLight - rPPG Heart Rate Detection

# import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, filtfilt, detrend
import os
import pandas as pd
from pathlib import Path
import glob

class rPPGProcessor:

    # Initialize the rPPG processor
    def __init__(self, fps=30):
        self.fps = fps
        # Initialize face detector (haar cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Heart rate frequency range in Hz (50-180 BPM -> 0.83-3.0 Hz)
        self.hr_freq_min = 0.83  # 50 BPM
        self.hr_freq_max = 3.0   # 180 BPM
        
        # Storage for extracted signals
        self.rgb_signals = [] # color change signals from face
        self.timestamps = [] # time values for each frame
        self.face_locations = []  # Face tracking

        print("rPPGProcessor initialized successfully")

    # face detection with multiple ROIs
    def detect_face_roi(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # more sensitive face detection
            minNeighbors=8,    # more stable 
            minSize=(120, 120) # larger minimum size (don't detect tiny faces)
        )
        
        if len(faces) == 0:
            return None, None, None # no face detected
            
        # Take the largest face
        face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = face
        
        # Multiple ROIs - forehead, left cheek, right cheek
        rois = {} # dictionary to hold ROIs
        
        # Forehead ROI
        forehead_x = x + w // 4
        forehead_y = y + h // 8
        forehead_w = w // 2
        forehead_h = h // 4
        rois['forehead'] = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]
        
        # Left cheek ROI
        left_cheek_x = x + w // 8
        left_cheek_y = y + h // 2
        left_cheek_w = w // 4
        left_cheek_h = h // 6
        rois['left_cheek'] = frame[left_cheek_y:left_cheek_y+left_cheek_h, left_cheek_x:left_cheek_x+left_cheek_w]
        
        # Right cheek ROI
        right_cheek_x = x + 5*w // 8
        right_cheek_y = y + h // 2
        right_cheek_w = w // 4
        right_cheek_h = h // 6
        rois['right_cheek'] = frame[right_cheek_y:right_cheek_y+right_cheek_h, right_cheek_x:right_cheek_x+right_cheek_w]
        
        return rois, (x, y, w, h), {
            'forehead': (forehead_x, forehead_y, forehead_w, forehead_h),
            'left_cheek': (left_cheek_x, left_cheek_y, left_cheek_w, left_cheek_h),
            'right_cheek': (right_cheek_x, right_cheek_y, right_cheek_w, right_cheek_h)
        }
    
    # extract RGB signals from ROIs with quality checks
    def extract_rgb_signal(self, rois):
        if not rois or len(rois) == 0:
            return None # no ROIs
            
        signals = {} # dictionary to hold rgb values
        
        for roi_name, roi in rois.items():
            if roi is None or roi.size == 0:
                continue
                
            # Calculate mean RGB values
            mean_rgb = np.mean(roi, axis=(0, 1))
            
            # Convert BGR to RGB
            rgb_values = [mean_rgb[2], mean_rgb[1], mean_rgb[0]]  # [R, G, B]
            
            # Simple quality check - avoid too dark or too bright regions
            brightness = np.mean(mean_rgb)
            if 30 < brightness < 220:  # Good brightness range
                signals[roi_name] = rgb_values
        
        # Combine signals (weighted average, forehead gets more weight cuz it's more stable)
        if 'forehead' in signals:
            final_signal = signals['forehead']
            weight_sum = 1.0
            
            # Add cheek contributions with lower weights
            if 'left_cheek' in signals:
                final_signal = [final_signal[i]*1.0 + signals['left_cheek'][i]*0.3 for i in range(3)]
                weight_sum += 0.3
                
            if 'right_cheek' in signals:
                final_signal = [final_signal[i] + signals['right_cheek'][i]*0.3 for i in range(3)]
                weight_sum += 0.3
            
            # Normalize by total weight
            final_signal = [val/weight_sum for val in final_signal]
            return final_signal
        
        # Fallback to any available signal (if forehead missing)
        elif signals:
            return list(signals.values())[0]
        
        return None
    
    # main video processing loop
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Reset storage
        self.rgb_signals = []
        self.timestamps = []
        self.face_locations = []
        
        frame_count = 0
        successful_extractions = 0
        consecutive_failures = 0
        
        print(f"Processing video: {video_path}")
        
        # read video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_count / self.fps
            
            # Detect face and extract ROIs
            rois, face_rect, roi_coords = self.detect_face_roi(frame)
            
            if rois is not None:
                # Extract combined RGB signal
                rgb_values = self.extract_rgb_signal(rois)
                
                if rgb_values is not None:
                    self.rgb_signals.append(rgb_values)
                    self.timestamps.append(timestamp)
                    self.face_locations.append(face_rect)
                    successful_extractions += 1
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1
            
            # i'll change this when we go live with real-time processing
            # Stop if too many consecutive failures (lost face)
            if consecutive_failures > 30:  # 1 second at 30 FPS
                print(f"Warning: Lost face tracking at frame {frame_count}")
            
            frame_count += 1
            
            # Progress update
            if frame_count % 300 == 0:  # Every 10 seconds
                print(f"Processed {frame_count} frames ({frame_count/self.fps:.1f}s), extracted {successful_extractions} signals")
        
        cap.release()
        
        print(f"Video processing completed!")
        print(f"Total frames: {frame_count} ({frame_count/self.fps:.1f}s)")
        print(f"Successful extractions: {successful_extractions}")
        print(f"Success rate: {successful_extractions/frame_count*100:.1f}%")
        
        return len(self.rgb_signals) > 0
    
    # basic preprocessing: detrend and normalize
    def preprocess_signal(self, signal_data):
        print(f"DEBUG: Preprocessing signal of length {len(signal_data)}")
        if len(signal_data) < self.fps * 2:  # Need at least 2 seconds
            return None
        
        try:
            # Convert list to numpy array
            signal_array = np.array(signal_data)
            
            # Detrend to remove slow variations (for slow changes like light and brightness)
            detrended = detrend(signal_array, type='linear')
            
            # Normalize to zero mean, unit variance (this way i only keep relative changes -heartbeat)
            signal_std = np.std(detrended)
            if signal_std < 1e-8:
                print("DEBUG: Signal has zero variance, returning None")
                return None
                
            normalized = (detrended - np.mean(detrended)) / signal_std
            return normalized
        
        except Exception as e:
            print(f"DEBUG ERROR in preprocess_signal: {e}")
            traceback.print_exc()
            return None
    
    # advanced filtering with bandpass and smoothing
    def advanced_filter_signal(self, signal_data):

        # Preprocessing
        processed_signal = self.preprocess_signal(signal_data)
        if processed_signal is None:
            return None
        
        # bandpass filter design
        nyquist = 0.5 * self.fps
        low = self.hr_freq_min / nyquist
        high = self.hr_freq_max / nyquist
        
        # Ensure filter coefficients are in valid range (0,1)
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        try:
            # butterworth bandpass filter
            b, a = butter(4, [low, high], btype='band')
            filtered_signal = filtfilt(b, a, processed_signal)
            
            # Additional smoothing filter
            b_smooth, a_smooth = butter(2, 0.1, btype='low')
            smoothed = filtfilt(b_smooth, a_smooth, filtered_signal)
            
            return smoothed  # Return the smoothed version
            
        except Exception as e:
            print(f"Filtering error: {e}")
            return processed_signal
    
    # heart rate estimation using fft, peaks, autocorr
    def heart_rate_estimation(self, rgb_channel='G'):
        if len(self.rgb_signals) < self.fps * 3:  # Need at least 3 seconds
            return None, 0, {}
        
        # Convert to numpy array
        rgb_array = np.array(self.rgb_signals)
        
        # Select channel (Green is usually best)
        channel_map = {'R': 0, 'G': 1, 'B': 2}
        channel_signal = rgb_array[:, channel_map[rgb_channel]]
        
        # Apply advanced filtering
        filtered_signal = self.advanced_filter_signal(channel_signal)
        if filtered_signal is None:
            return None, 0, {}
        
        # Method 1: FFT-based estimation
        fft_hr, fft_confidence = self.fft_heart_rate_estimation(filtered_signal)
        
        # Method 2: Peak-based estimation
        peak_hr, peak_confidence = self.peak_heart_rate_estimation(filtered_signal)
        
        # Method 3: Autocorrelation-based estimation
        autocorr_hr, autocorr_confidence = self.autocorr_heart_rate_estimation(filtered_signal)
        
        # Combine estimates with confidence weighting
        methods = {
            'fft': (fft_hr, fft_confidence),
            'peaks': (peak_hr, peak_confidence),
            'autocorr': (autocorr_hr, autocorr_confidence)
        }
        
        # Filter out invalid or low confidence estimates
        valid_methods = {k: v for k, v in methods.items() if v[0] is not None and v[1] > 0.1}
        
        if not valid_methods:
            return None, 0, methods
        
        # Weighted average based on confidence
        total_weight = sum(conf for _, conf in valid_methods.values())
        weighted_hr = sum(hr * conf for hr, conf in valid_methods.values()) / total_weight # bpm
        avg_confidence = total_weight / len(valid_methods)
        
        return weighted_hr, avg_confidence, methods
    
    # fft-based heart-rate estimation
    def fft_heart_rate_estimation(self, signal_data):
        try:
            # Perform FFT
            n_samples = len(signal_data)
            fft_values = fft(signal_data)
            frequencies = fftfreq(n_samples, 1/self.fps) # her FFT katsayısının frekans karşılığı
            
            # Keep only positive frequencies in HR range
            pos_mask = (frequencies > self.hr_freq_min) & (frequencies < self.hr_freq_max)
            hr_frequencies = frequencies[pos_mask]
            hr_fft_magnitude = np.abs(fft_values[pos_mask])
            
            if len(hr_frequencies) == 0:
                return None, 0
            
            # Find peak
            peak_idx = np.argmax(hr_fft_magnitude)
            peak_frequency = hr_frequencies[peak_idx]
            heart_rate = peak_frequency * 60
            
            # Calculate confidence
            total_power = np.sum(hr_fft_magnitude)
            peak_power = hr_fft_magnitude[peak_idx]
            confidence = peak_power / total_power if total_power > 0 else 0
            
            return heart_rate, confidence
            # heart rate= bpm, confidence= 0-1
        except Exception as e:
            print(f"FFT estimation error: {e}")
            return None, 0
    
    # Peak-based heart rate estimation
    def peak_heart_rate_estimation(self, signal_data):
        
        try:
            # Find peaks in signal
            # Adaptive threshold based on signal properties 
            # çok küçük dalgalalanmaları göz ardı etmek için sinyalin standart sapmasına göre ayarlıyoruz
            threshold = np.std(signal_data) * 0.3
            min_distance = int(self.fps * 60 / 200)  # Minimum distance between peaks (200 BPM max)
            
            peaks, properties = find_peaks(signal_data, height=threshold, distance=min_distance)
            
            if len(peaks) < 3:  # Need at least 3 peaks
                return None, 0
            
            # Calculate intervals between peaks
            peak_intervals = np.diff(peaks) / self.fps  # Convert to seconds
            
            # Medyana göre çok sapmış interval değerlerini çıkar
            median_interval = np.median(peak_intervals)
            mad = np.median(np.abs(peak_intervals - median_interval))  # Median Absolute Deviation - MAD
            
            # Keep intervals within 3*MAD of median
            valid_intervals = peak_intervals[np.abs(peak_intervals - median_interval) < 3 * mad]
            
            if len(valid_intervals) < 2:
                return None, 0
            
            # Convert to heart rate
            avg_interval = np.mean(valid_intervals)
            heart_rate = 60 / avg_interval
            
            # Confidence based on interval consistency
            interval_std = np.std(valid_intervals)
            confidence = 1.0 / (1.0 + interval_std)  # Higher std = lower confidence
            
            return heart_rate, confidence
        except Exception as e:
            print(f"Peak estimation error: {e}")
            return None, 0
    
    # Autocorrelation-based heart rate estimation
    def autocorr_heart_rate_estimation(self, signal_data):
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(signal_data, signal_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]  # Keep positive lags
            
            # Convert lag to heart rate range
            min_lag = int(self.fps * 60 / self.hr_freq_max / 60)  # Max HR lag
            max_lag = int(self.fps * 60 / self.hr_freq_min / 60)  # Min HR lag
            
            if max_lag >= len(autocorr):
                return None, 0
            
            # Find peak in valid range
            search_range = autocorr[min_lag:max_lag]
            if len(search_range) == 0:
                return None, 0
            
            peak_lag = np.argmax(search_range) + min_lag
            heart_rate = 60 * self.fps / peak_lag # convert peak lag to BPM
            
            # Confidence based on peak prominence
            peak_value = autocorr[peak_lag]
            mean_value = np.mean(autocorr[min_lag:max_lag])
            confidence = (peak_value - mean_value) / (np.std(autocorr[min_lag:max_lag]) + 1e-8)
            confidence = min(1.0, max(0.0, confidence / 5.0))  # Normalize to 0-1
            
            return heart_rate, confidence
        except Exception as e:
            print(f"Autocorr estimation error: {e}")
            return None, 0
    
    # Detailed visualization with all analysis steps
    def visualize_results_detailed(self):
        if len(self.rgb_signals) == 0:
            print("No signals to visualize")
            return
        
        try:
            rgb_array = np.array(self.rgb_signals)
            timestamps = np.array(self.timestamps)
            
            # Create comprehensive plot 
            fig, axes = plt.subplots(5, 1, figsize=(15, 20))
            
            # 1. Raw RGB signals
            axes[0].plot(timestamps, rgb_array[:, 0], 'r-', label='Red', alpha=0.7)
            axes[0].plot(timestamps, rgb_array[:, 1], 'g-', label='Green', alpha=0.7)
            axes[0].plot(timestamps, rgb_array[:, 2], 'b-', label='Blue', alpha=0.7)
            axes[0].set_title('Raw RGB Signals')
            axes[0].set_xlabel('Time (seconds)')
            axes[0].set_ylabel('Intensity')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # 2. Processed Green signal
            green_signal = rgb_array[:, 1]
            processed = self.preprocess_signal(green_signal)
            if processed is not None:
                axes[1].plot(timestamps, green_signal, 'g-', alpha=0.5, label='Raw Green')
                # Adjust processed signal length to match timestamps
                if len(processed) == len(timestamps):
                    axes[1].plot(timestamps, processed, 'darkgreen', linewidth=2, label='Preprocessed')
                    print("DEBUG: Preprocessed signal plotted")
                else:
                    print(f"DEBUG: Length mismatch - processed: {len(processed)}, timestamps: {len(timestamps)}")
            else:
                print("DEBUG: Preprocessing returned None")
                
            axes[1].set_title('Signal Preprocessing')
            axes[1].set_xlabel('Time (seconds)')
            axes[1].set_ylabel('Amplitude')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            print("DEBUG: Preprocessed plot completed")
        
        # 3. Filtered signal
            filtered_signal = self.advanced_filter_signal(green_signal)
            if filtered_signal is not None and len(filtered_signal) == len(timestamps):
                axes[2].plot(timestamps, filtered_signal, 'navy', linewidth=2, label='Filtered Signal')
                
                # Add peaks if available
                try:
                    threshold = np.std(filtered_signal) * 0.3
                    min_distance = int(self.fps * 60 / 200)
                    peaks, _ = find_peaks(filtered_signal, height=threshold, distance=min_distance)
                    if len(peaks) > 0:
                        axes[2].plot(timestamps[peaks], filtered_signal[peaks], 'ro', markersize=6, label='Detected Peaks')
                        print(f"DEBUG: Found {len(peaks)} peaks for visualization")
                except Exception as e:
                    print(f"DEBUG: Peak finding error: {e}")
                    
            axes[2].set_title('Filtered Signal with Peak Detection')
            axes[2].set_xlabel('Time (seconds)')
            axes[2].set_ylabel('Amplitude')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            print("DEBUG: Filtered signal plot completed")
            
            # 4. FFT Spectrum
            print("DEBUG: Computing FFT spectrum...")
            if filtered_signal is not None:
                try:
                    n_samples = len(filtered_signal)
                    fft_values = fft(filtered_signal)
                    frequencies = fftfreq(n_samples, 1/self.fps)
                    
                    pos_mask = (frequencies > 0) & (frequencies < 5)
                    plot_frequencies = frequencies[pos_mask]
                    plot_fft_magnitude = np.abs(fft_values[pos_mask])
                    
                    axes[3].plot(plot_frequencies * 60, plot_fft_magnitude)
                    axes[3].axvspan(50, 180, alpha=0.2, color='red', label='Valid HR Range')
                    axes[3].set_title('Frequency Spectrum Analysis')
                    axes[3].set_xlabel('Heart Rate (BPM)')
                    axes[3].set_ylabel('FFT Magnitude')
                    axes[3].legend()
                    axes[3].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"DEBUG: FFT spectrum error: {e}")
                    traceback.print_exc()
            
            # 5. Heart rate estimation comparison
            print("DEBUG: Getting heart rate estimations...")
            hr_estimate, confidence, methods = self.heart_rate_estimation()
            
            if methods:
                method_names = list(methods.keys())
                hrs = [methods[m][0] if methods[m][0] is not None else 0 for m in method_names]
                confs = [methods[m][1] for m in method_names]
                
                x_pos = np.arange(len(method_names))
                bars = axes[4].bar(x_pos, hrs, alpha=0.7)
                axes[4].set_xlabel('Estimation Method')
                axes[4].set_ylabel('Heart Rate (BPM)')
                axes[4].set_title('Heart Rate Estimation Comparison')
                axes[4].set_xticks(x_pos)
                axes[4].set_xticklabels(method_names)
                
                # Color bars by confidence
                for i, (bar, conf) in enumerate(zip(bars, confs)):
                    bar.set_color(plt.cm.RdYlGn(conf))
                    axes[4].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                                f'{conf:.3f}', ha='center', va='bottom', fontsize=9)
                
                axes[4].grid(True, alpha=0.3)
                print("DEBUG: HR comparison plot completed")
            else:
                print("DEBUG: No methods available for comparison plot")
            
            print("DEBUG: Finalizing plot layout...")
            plt.tight_layout()
            
            print("DEBUG: Showing plot...")
            plt.show()
            
            # Print detailed results
            print(f"\n=== DEBUG Detailed Analysis Results ===")
            if hr_estimate is not None:
                print(f"Combined Estimate: {hr_estimate:.1f} BPM (Confidence: {confidence:.3f})")
                print(f"\nMethod Breakdown:")
                for method, (hr, conf) in methods.items():
                    print(f"  {method.capitalize()}: {'{:.1f}'.format(hr) if hr else 'N/A'} BPM (conf: {conf:.3f})")
            else:
                print("No valid heart rate estimate obtained")
                
        except Exception as e:
            print(f"DEBUG ERROR in visualize_results_detailed: {e}")
            traceback.print_exc()

# ground truth processing from BVP signal
def ground_truth_processing(gt_path, fps=30):
    try:
        # Read the BVP signal
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        
        # Parse BVP values (assuming one value per line)
        bvp_values = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                try:
                    # Handle both space-separated and single values
                    values = line.split()
                    if len(values) >= 1:
                        bvp_values.append(float(values[0]))
                except ValueError:
                    continue
        
        if len(bvp_values) == 0:
            print("No valid BVP values found in ground truth file")
            return None
        
        # convert list to numpy array
        bvp_signal = np.array(bvp_values)
        
        # Estimate HR from BVP signal using multiple methods
        # Method 1: Peak-based analysis
        hr_peak = estimate_hr_from_bvp_peaks(bvp_signal, fps)
        
        # Method 2: FFT-based analysis
        hr_fft = estimate_hr_from_bvp_fft(bvp_signal, fps)
        
        # Method 3: Sliding window analysis
        hr_window = estimate_hr_from_bvp_sliding(bvp_signal, fps)
        
        # Combine estimates
        valid_estimates = [hr for hr in [hr_peak, hr_fft, hr_window] if hr is not None and 50 <= hr <= 180]
        
        if valid_estimates:
            avg_hr = np.mean(valid_estimates)
            print(f"Ground truth HR estimates: Peak={hr_peak:.1f}, FFT={hr_fft:.1f}, Window={hr_window:.1f}")
            print(f"Average ground truth HR: {avg_hr:.1f} BPM")
            return avg_hr
        else:
            print("Could not extract reliable HR from ground truth BVP signal")
            return None
            
    except Exception as e:
        print(f"Error processing ground truth: {e}")
        return None

# Estimate HR from BVP using peak detection
def estimate_hr_from_bvp_peaks(bvp_signal, fps=30):
    try:
        # Preprocessing
        bvp_filtered = detrend(bvp_signal)
        
        # Bandpass filter for heart rate range
        nyquist = 0.5 * fps
        b, a = butter(3, [0.83/nyquist, 3.0/nyquist], btype='band')
        bvp_filtered = filtfilt(b, a, bvp_filtered)
        
        # Find peaks
        min_distance = int(fps * 60 / 200)  # Max 200 BPM
        peaks, _ = find_peaks(bvp_filtered, distance=min_distance)
        
        if len(peaks) < 3:
            return None
        
        # Calculate intervals
        intervals = np.diff(peaks) / fps
        median_interval = np.median(intervals)
        
        # Filter outliers
        mad = np.median(np.abs(intervals - median_interval))
        valid_intervals = intervals[np.abs(intervals - median_interval) < 3 * mad]
        
        if len(valid_intervals) < 2:
            return None
            
        hr = 60 / np.mean(valid_intervals)
        return hr if 50 <= hr <= 180 else None
    except Exception as e:
        print(f"HR from BVP, peaks estimation error: {e}")
        return None

# Estimate HR from BVP using FFT
def estimate_hr_from_bvp_fft(bvp_signal, fps=30):
    try:
        # Preprocessing
        bvp_processed = detrend(bvp_signal)
        bvp_processed = (bvp_processed - np.mean(bvp_processed)) / np.std(bvp_processed)
        
        # FFT
        n_samples = len(bvp_processed)
        fft_values = fft(bvp_processed)
        frequencies = fftfreq(n_samples, 1/fps)
        
        # HR frequency range
        hr_mask = (frequencies >= 0.83) & (frequencies <= 3.0)
        hr_frequencies = frequencies[hr_mask]
        hr_magnitudes = np.abs(fft_values[hr_mask])
        
        if len(hr_frequencies) == 0:
            return None
        
        # Find peak
        peak_idx = np.argmax(hr_magnitudes)
        peak_freq = hr_frequencies[peak_idx]
        hr = peak_freq * 60
        
        return hr if 50 <= hr <= 180 else None
    except Exception as e:
        print(f"HR from BVP, FFT estimation error: {e}")
        return None

# Estimate HR using sliding window approach
def estimate_hr_from_bvp_sliding(bvp_signal, fps=30, window_sec=10):
    try:
        window_size = int(window_sec * fps)
        if len(bvp_signal) < window_size:
            return None
        
        hr_estimates = []
        
        for i in range(0, len(bvp_signal) - window_size, window_size // 2):
            window = bvp_signal[i:i + window_size]
            
            # Quick FFT-based estimation for this window
            window_detrended = detrend(window)
            fft_vals = fft(window_detrended)
            freqs = fftfreq(len(window), 1/fps)
            
            hr_mask = (freqs >= 0.83) & (freqs <= 3.0)
            if np.any(hr_mask):
                hr_freqs = freqs[hr_mask]
                hr_mags = np.abs(fft_vals[hr_mask])
                peak_freq = hr_freqs[np.argmax(hr_mags)]
                hr = peak_freq * 60
                if 50 <= hr <= 180:
                    hr_estimates.append(hr) # add valid estimate to list
        
        if len(hr_estimates) >= 3:
            return np.median(hr_estimates)
        return None
    except Exception as e:
        print(f"HR from BVP, sliding window estimation error: {e}")
        return None

class UBFCDatasetLoader:
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.subjects = []
        self.load_dataset_structure()
    
    def load_dataset_structure(self):
        if not self.dataset_path.exists():
            print(f"Error: Dataset path does not exist: {self.dataset_path}")
            return
        
        subject_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name.startswith("subject")]
        
        print(f"Found {len(subject_dirs)} potential subject directories")
        
        for subject_dir in subject_dirs:
            video_file = subject_dir / "vid.avi"
            gt_file = subject_dir / "ground_truth.txt"
            
            if video_file.exists() and gt_file.exists():
                self.subjects.append({
                    'subject_id': subject_dir.name,
                    'video_path': str(video_file),
                    'ground_truth_path': str(gt_file),
                    'subject_dir': str(subject_dir)
                })
                print(f"✓ {subject_dir.name}: video and ground truth found")
            else:
                print(f"✗ {subject_dir.name}: missing files (video: {video_file.exists()}, gt: {gt_file.exists()})")
        
        print(f"Successfully loaded {len(self.subjects)} subjects")
        
    def get_subject_info(self, subject_idx):
        if subject_idx >= len(self.subjects):
            return None
        return self.subjects[subject_idx]

def demo_rppg():
    print("=== rPPG Heart Rate Detection Demo ===\n")

    dataset_path = "C:/Users/cemre/Desktop/vitalight/ubfc_2" 
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
        print("Please update the dataset_path variable with your local UBFC-2 dataset path")
        return
    
    try:
        dataset = UBFCDatasetLoader(dataset_path)
        
        if len(dataset.subjects) == 0:
            print("No subjects found in dataset")
            return
        
        # Process a subject (you can change the index)
        subject_idx = 0
        subject_info = dataset.get_subject_info(subject_idx)
        
        print(f"\nProcessing {subject_info['subject_id']}")
        print(f"Video: {subject_info['video_path']}")
        print(f"Ground Truth: {subject_info['ground_truth_path']}")
        
        # Initialize processor
        processor = rPPGProcessor(fps=30)
        
        # Process video
        print("\n--- Video Processing ---")
        success = processor.process_video(subject_info['video_path'])
        
        if success:
            # Detailed visualization
            print("\n--- Generating Detailed Analysis ---")
            processor.visualize_results_detailed()
            
            # Get heart rate estimation
            hr_estimate, confidence, methods = processor.heart_rate_estimation()
            
            # Process ground truth
            print("\n--- Processing Ground Truth ---")
            gt_hr = ground_truth_processing(subject_info['ground_truth_path'])

            # Final comparison
            print(f"\n=== FINAL RESULTS ===")
            print(f"Estimated HR: {hr_estimate:.1f} BPM (Confidence: {confidence:.3f})")
            
            if gt_hr is not None:
                error = abs(hr_estimate - gt_hr)
                relative_error = error / gt_hr * 100
                
                print(f"Ground Truth HR: {gt_hr:.1f} BPM")
                print(f"Absolute Error: {error:.1f} BPM")
                print(f"Relative Error: {relative_error:.1f}%")
                
                # Quality assessment
                if relative_error < 10:
                    print("✓ Excellent accuracy!")
                elif relative_error < 20:
                    print("✓ Good accuracy")
                elif relative_error < 30:
                    print("⚠ Moderate accuracy")
                else:
                    print("✗ Poor accuracy - needs improvement")
            else:
                print("Could not process ground truth for comparison")
                
            # Method breakdown
            print(f"\n--- Method Breakdown ---")
            for method, (hr, conf) in methods.items():
                status = "✓" if hr is not None and conf > 0.2 else "✗"
                hr_str = f"{hr:.1f}" if hr is not None else "N/A"
                print(f"{status} {method.capitalize()}: {hr_str} BPM (conf: {conf:.3f})")
        
        else:
            print("Failed to process video")
            
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
        demo_rppg()