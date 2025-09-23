# VitaLight - rPPG Heart Rate Detection

# import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, filtfilt, detrend
from sklearn.decomposition import FastICA
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
        self.roi_signals = {}  # Store individual ROI signals
        
        # Quality tracking
        self.signal_quality_scores = []

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
        
        # Forehead ROI - larger, more stable region
        forehead_x = x + w // 5
        forehead_y = y + h // 10
        forehead_w = 3 * w // 5
        forehead_h = h // 3
        rois['forehead'] = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]
        
        # Left cheek ROI
        left_cheek_x = x + w // 10
        left_cheek_y = y + 2*h // 5
        left_cheek_w = w // 3
        left_cheek_h = h // 4
        rois['left_cheek'] = frame[left_cheek_y:left_cheek_y+left_cheek_h, left_cheek_x:left_cheek_x+left_cheek_w]
        
        # Right cheek ROI
        right_cheek_x = x + 3*w // 5
        right_cheek_y = y + 2*h // 5
        right_cheek_w = w // 3
        right_cheek_h = h // 4
        rois['right_cheek'] = frame[right_cheek_y:right_cheek_y+right_cheek_h, right_cheek_x:right_cheek_x+right_cheek_w]
        
        roi_coords = {
            'forehead': (forehead_x, forehead_y, forehead_w, forehead_h),
            'left_cheek': (left_cheek_x, left_cheek_y, left_cheek_w, left_cheek_h),
            'right_cheek': (right_cheek_x, right_cheek_y, right_cheek_w, right_cheek_h)
        }
        
        return rois, (x, y, w, h), roi_coords
    
    # Quality assessment for ROI signals
    def assess_signal_quality(self, roi):
        if roi is None or roi.size == 0:
            return 0.0
            
        # Convert to grayscale for analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Multiple quality metrics
        brightness = np.mean(gray_roi)
        contrast = np.std(gray_roi)
        
        # Check for good skin tone range - avoid too dark or too bright regions
        brightness_score = 1.0 if 40 < brightness < 200 else max(0.0, 1.0 - abs(brightness - 120) / 120)
        contrast_score = min(1.0, contrast / 30.0)  # Higher contrast is better up to a point
        
        # Size quality (larger ROIs are generally better)
        size_score = min(1.0, roi.size / 10000.0)
        
        # Combined quality score (weighted average, forehead gets more weight cuz it's more stable) 
        quality = 0.5 * brightness_score + 0.3 * contrast_score + 0.2 * size_score
        return quality

    # CHROM algorithm for robust rPPG signal extraction
    def extract_chrom_signal(self, rois):
        if not rois or len(rois) == 0:
            return None, 0.0
            
        # Extract RGB signals from each ROI with quality weighting
        roi_data = {}
        total_quality = 0.0
        
        for roi_name, roi in rois.items():
            if roi is None or roi.size == 0:
                continue
                
            # Calculate mean RGB
            mean_rgb = np.mean(roi, axis=(0, 1))
            rgb_values = [mean_rgb[2], mean_rgb[1], mean_rgb[0]]  # BGR to RGB
            
            # Assess quality
            quality = self.assess_signal_quality(roi)
            
            if quality > 0.3:  # Only use decent quality signals
                roi_data[roi_name] = {
                    'rgb': rgb_values,
                    'quality': quality
                }
                total_quality += quality
        
        if not roi_data:
            return None, 0.0
        
        # CHROM algorithm implementation
        # Combine RGB signals with quality weighting
        combined_rgb = np.zeros(3)
        for roi_name, data in roi_data.items():
            weight = data['quality'] / total_quality
            combined_rgb += np.array(data['rgb']) * weight
        
        # CHROM transformation
        R, G, B = combined_rgb
        
        # Normalize by mean to reduce illumination effects
        R_norm = R / (R + G + B + 1e-8)
        G_norm = G / (R + G + B + 1e-8)
        
        # CHROM linear combination (optimized coefficients)
        X = 3 * R_norm - 2 * G_norm
        Y = 1.5 * R_norm + G_norm - 1.5 * B / (R + G + B + 1e-8)
        
        # The CHROM signal is the X component (most sensitive to pulse)
        chrom_signal = X
        
        # Store individual ROI signals for analysis
        for roi_name, data in roi_data.items():
            if roi_name not in self.roi_signals:
                self.roi_signals[roi_name] = []
            self.roi_signals[roi_name].append(data['rgb'])
        
        avg_quality = total_quality / len(roi_data)
        return chrom_signal, avg_quality
    
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
        self.roi_signals = {}
        self.signal_quality_scores = []
        
        frame_count = 0
        successful_extractions = 0
        consecutive_failures = 0
        
        # read video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_count / self.fps
            
            # Detect face and extract ROIs
            rois, face_rect, roi_coords = self.detect_face_roi(frame)
            
            if rois is not None:
                # Extract CHROM signal
                chrom_value, quality = self.extract_chrom_signal(rois)
                
                if chrom_value is not None and quality > 0.2:
                    self.rgb_signals.append(chrom_value)
                    self.timestamps.append(timestamp)
                    self.face_locations.append(face_rect)
                    self.signal_quality_scores.append(quality)
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
        print(f"Processing completed: {successful_extractions}/{frame_count} frames")
        return len(self.rgb_signals) > 0

    # Multi-stage temporal filtering
    def temporal_filtering(self, signal_data):
        if len(signal_data) < self.fps * 2:  # Need at least 2 seconds
            return None
        
        # Convert list to numpy array
        signal_array = np.array(signal_data)
        
        # 1. Detrending - remove slow variations (for slow changes like light and brightness)
        detrended = detrend(signal_array, type='linear')
        
        # 2. Moving average filter for smoothing
        window_size = max(3, self.fps // 10)  # ~0.1 second window
        smoothed = np.convolve(detrended, np.ones(window_size)/window_size, mode='same')
        
        # 3. Bandpass filter design
        nyquist = 0.5 * self.fps

        # Ensure filter coefficients are in valid range (0,1)
        low = max(0.01, self.hr_freq_min / nyquist)
        high = min(0.99, self.hr_freq_max / nyquist)
        
        try:
            # butterworth bandpass filter
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, smoothed)
            
            # 4. Normalize
            filtered_norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
            
            return filtered_norm
        except Exception as e:
            print(f"Filtering error: {e}")
            return smoothed

    def signal_quality_filtering(self):
        """Filter signals based on quality scores"""
        if len(self.signal_quality_scores) == 0:
            return self.rgb_signals
        
        # Calculate quality threshold (keep top 70% of signals)
        quality_threshold = np.percentile(self.signal_quality_scores, 30)
        
        filtered_signals = []
        filtered_timestamps = []
        
        for i, (signal, quality) in enumerate(zip(self.rgb_signals, self.signal_quality_scores)):
            if quality >= quality_threshold:
                filtered_signals.append(signal)
                filtered_timestamps.append(self.timestamps[i])
        
        return filtered_signals

    def ica_signal_separation(self, multi_roi_signals):
        """Use ICA to separate pulse signal from noise"""
        if len(multi_roi_signals) < 3 or len(self.roi_signals) < 2:
            return None
        
        try:
            # Prepare multi-channel data
            channels = []
            for roi_name, roi_data in self.roi_signals.items():
                if len(roi_data) > len(multi_roi_signals) * 0.8:  # Use ROIs with sufficient data
                    # Use green channel (index 1) as it's most sensitive to pulse
                    green_channel = [rgb[1] for rgb in roi_data[:len(multi_roi_signals)]]
                    channels.append(green_channel)
            
            if len(channels) < 2:
                return None
            
            # Apply ICA
            ica = FastICA(n_components=min(len(channels), 3), random_state=42, max_iter=1000)
            signals_matrix = np.array(channels).T
            ica_signals = ica.fit_transform(signals_matrix)
            
            # Select component with strongest heart rate signal
            best_component = None
            best_hr_power = 0
            
            for component in ica_signals.T:
                # Quick FFT to find HR power
                fft_vals = np.abs(fft(component))
                freqs = fftfreq(len(component), 1/self.fps)
                hr_mask = (freqs >= self.hr_freq_min) & (freqs <= self.hr_freq_max)
                hr_power = np.sum(fft_vals[hr_mask])
                
                if hr_power > best_hr_power:
                    best_hr_power = hr_power
                    best_component = component
            
            return best_component
        except:
            return None
        
    def heart_rate_estimation(self):
        if len(self.rgb_signals) < self.fps * 3:
            return None, 0, {}
        
        # Use quality-filtered signals
        quality_signals = self.signal_quality_filtering()
        
        if len(quality_signals) < self.fps * 3:
            return None, 0, {}
        
        # Apply temporal filtering
        filtered_signal = self.temporal_filtering(quality_signals)
        if filtered_signal is None:
            return None, 0, {}
        
        # Try ICA separation if multi-ROI data available
        ica_signal = self.ica_signal_separation(quality_signals)
        if ica_signal is not None and len(ica_signal) == len(filtered_signal):
            # Combine filtered and ICA signals
            combined_signal = 0.7 * filtered_signal + 0.3 * ica_signal
        else:
            combined_signal = filtered_signal
        
        # Multiple estimation methods
        fft_hr, fft_conf = self.fft_estimation(combined_signal)
        peak_hr, peak_conf = self.peak_estimation(combined_signal)
        autocorr_hr, autocorr_conf = self.autocorr_estimation(combined_signal)
        welch_hr, welch_conf = self.welch_estimation(combined_signal)  
        
        # Combine estimates with confidence weighting
        methods = {
            'fft': (fft_hr, fft_conf),
            'peaks': (peak_hr, peak_conf),
            'autocorr': (autocorr_hr, autocorr_conf),
            'welch': (welch_hr, welch_conf)
        }
        
        # Filter out invalid or low confidence estimates
        valid_methods = {k: v for k, v in methods.items() 
                        if v[0] is not None and v[1] > 0.1 and 50 <= v[0] <= 180}
        
        if not valid_methods:
            return None, 0, methods
        
        # Weighted average based on confidence
        total_weight = sum(conf for _, conf in valid_methods.values())
        weighted_hr = sum(hr * conf for hr, conf in valid_methods.values()) / total_weight
        avg_confidence = total_weight / len(valid_methods)
        
        return weighted_hr, avg_confidence, methods

    # fft-based heart-rate estimation
    def fft_estimation(self, signal_data):
        try:
            # Perform FFT
            n_samples = len(signal_data)
            fft_values = fft(signal_data)
            frequencies = fftfreq(n_samples, 1/self.fps) # her FFT katsayısının frekans karşılığı
            
            # Keep only positive frequencies in HR range
            hr_mask = (frequencies > self.hr_freq_min) & (frequencies < self.hr_freq_max)

            if not np.any(hr_mask):  # Check if mask has any True values
                return None, 0
        
            hr_frequencies = frequencies[hr_mask]
            hr_magnitudes = np.abs(fft_values[hr_mask])
            
            if len(hr_frequencies) == 0:
                return None, 0
            
            # Find peak
            peak_idx = np.argmax(hr_magnitudes)
            peak_freq = hr_frequencies[peak_idx]
            heart_rate = peak_freq * 60
            
            # Calculate confidence
            total_power = np.sum(hr_magnitudes)
            peak_power = hr_magnitudes[peak_idx]

            # Check for secondary peaks (harmonic validation)
            sorted_indices = np.argsort(hr_magnitudes)[::-1]
            secondary_ratio = hr_magnitudes[sorted_indices[1]] / peak_power if len(sorted_indices) > 1 else 0
            
            base_confidence = peak_power / total_power
            harmonic_bonus = 0.2 if secondary_ratio < 0.5 else 0  # Single dominant peak is better
            
            confidence = min(1.0, base_confidence + harmonic_bonus)

            return heart_rate, confidence
            # heart rate= bpm, confidence= 0-1
        except Exception as e:
            print(f"FFT estimation error: {e}")
            return None, 0
    
    # Peak-based heart rate estimation with adaptive thresholding
    def peak_estimation(self, signal_data):
        
        try:
            # Find peaks in signal
            # Adaptive threshold based on signal properties 
            # çok küçük dalgalalanmaları göz ardı etmek için sinyalin standart sapmasına göre ayarlıyoruz
            signal_std = np.std(signal_data)
            threshold = max(0.1 * signal_std, np.percentile(signal_data, 75))
            min_distance = int(self.fps * 60 / 200)  # Minimum distance between peaks (200 BPM max)
            
            peaks, properties = find_peaks(signal_data, height=threshold, distance=min_distance)
            
            if len(peaks) < 4:  # Need at least 4 peaks
                return None, 0
            
            # Calculate intervals between peaks
            intervals = np.diff(peaks) / self.fps  # Convert to seconds
            
            # Medyana göre çok sapmış interval değerlerini çıkar
            median_interval = np.median(intervals)
            mad = np.median(np.abs(intervals - median_interval))  # Median Absolute Deviation - MAD
            
            # Keep intervals within 3*MAD of median
            valid_mask = np.abs(intervals - median_interval) < 2.5 * mad
            valid_intervals = intervals[valid_mask]
            
            if len(valid_intervals) < 3:
                return None, 0
            
            # Convert to heart rate
            heart_rate = 60 / np.mean(valid_intervals)
            
            # Confidence based on interval consistency and peak count
            consistency = 1.0 / (1.0 + np.std(valid_intervals))
            peak_count_bonus = min(0.3, len(valid_intervals) / 10.0)
            confidence = consistency + peak_count_bonus  # Higher std = lower confidence
            
            return heart_rate, min(1.0, confidence)
        except Exception as e:
            print(f"Peak estimation error: {e}")
            return None, 0
    
    # Autocorrelation-based heart rate estimation
    def autocorr_estimation(self, signal_data):
        try:
            # Calculate autocorrelation
            autocorr = np.correlate(signal_data, signal_data, mode='full')
            autocorr = autocorr[autocorr.size // 2:]  # Keep positive lags
            
            # Convert lag to heart rate range
            min_lag = int(self.fps * 60 / 180)  # Max HR lag
            max_lag = int(self.fps * 60 / 50)  # Min HR lag
            
            if max_lag >= len(autocorr):
                return None, 0
            
            # Find peak in valid range
            search_autocorr = autocorr[min_lag:max_lag]

            # Find multiple peaks and select the most prominent
            peaks, _ = find_peaks(search_autocorr, height=np.percentile(search_autocorr, 70))
            
            if len(peaks) == 0:
                peak_lag = np.argmax(search_autocorr) + min_lag
            else:
                # Select highest peak
                best_peak_idx = peaks[np.argmax(search_autocorr[peaks])]
                peak_lag = best_peak_idx + min_lag

            heart_rate = 60 * self.fps / peak_lag # convert peak lag to BPM
            
            # Confidence based on peak prominence
            peak_value = autocorr[peak_lag]
            mean_value = np.mean(autocorr[min_lag:max_lag])
            std_value = np.std(autocorr[min_lag:max_lag])
            
            snr = (peak_value - mean_value) / (std_value + 1e-8)
            confidence = min(1.0, max(0.0, snr / 5.0))  # Normalize to 0-1
            
            return heart_rate, confidence
        except Exception as e:
            print(f"Autocorr estimation error: {e}")
            return None, 0
    
    def welch_estimation(self, signal_data):
        """Welch's method for spectral estimation"""
        try:
            from scipy import signal as sp_signal
            
            # Welch's method parameters
            nperseg = min(len(signal_data) // 4, self.fps * 8)  # 8 second segments
            
            freqs, psd = sp_signal.welch(signal_data, fs=self.fps, nperseg=nperseg)
            
            # Focus on heart rate range
            hr_mask = (freqs >= self.hr_freq_min) & (freqs <= self.hr_freq_max)
            hr_freqs = freqs[hr_mask]
            hr_psd = psd[hr_mask]
            
            if len(hr_freqs) == 0:
                return None, 0
            
            # Find peak frequency
            peak_idx = np.argmax(hr_psd)
            peak_freq = hr_freqs[peak_idx]
            heart_rate = peak_freq * 60
            
            # Confidence based on peak prominence in PSD
            total_power = np.sum(hr_psd)
            peak_power = hr_psd[peak_idx]
            confidence = peak_power / total_power
            
            return heart_rate, confidence
        except:
            return None, 0

    # visualization
    def visualize_results(self):
        if len(self.rgb_signals) == 0:
            print("No signals to visualize")
            return
        
        # Create comprehensive plot 
        fig, axes = plt.subplots(4, 1, figsize=(15, 16))

        timestamps = np.array(self.timestamps)
        signals = np.array(self.rgb_signals)

        # 1. Raw CHROM signals with quality overlay
        axes[0].plot(timestamps, signals, 'b-', alpha=0.7, label='CHROM Signal')
        if self.signal_quality_scores:
            quality_normalized = np.array(self.signal_quality_scores) * np.max(signals) / np.max(self.signal_quality_scores)
            axes[0].plot(timestamps, quality_normalized, 'r--', alpha=0.5, label='Signal Quality')
        axes[0].set_title('Raw CHROM Signal with Quality Assessment')
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Filtered signal
        filtered_signal = self.temporal_filtering(self.rgb_signals)
        if filtered_signal is not None:
            filtered_timestamps = timestamps[:len(filtered_signal)]
            axes[1].plot(filtered_timestamps, filtered_signal, 'navy', linewidth=2)
            
            # Add detected peaks
            try:
                threshold = np.std(filtered_signal) * 0.3
                min_distance = int(self.fps * 60 / 200)
                peaks, _ = find_peaks(filtered_signal, height=threshold, distance=min_distance)
                if len(peaks) > 0:
                    axes[1].plot(filtered_timestamps[peaks], filtered_signal[peaks], 
                               'ro', markersize=6, label=f'{len(peaks)} Peaks')
            except:
                pass
        
        axes[1].set_title('Filtered Signal with Peak Detection')
        axes[1].set_xlabel('Time (seconds)')
        axes[1].set_ylabel('Amplitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Frequency spectrum (FFT + Welch)
        if filtered_signal is not None:
            # FFT
            n_samples = len(filtered_signal)
            fft_values = np.abs(fft(filtered_signal))
            frequencies = fftfreq(n_samples, 1/self.fps)
            
            pos_mask = (frequencies > 0) & (frequencies < 5)
            axes[2].plot(frequencies[pos_mask] * 60, fft_values[pos_mask], 
                        'b-', alpha=0.7, label='FFT')
            
            # Welch PSD
            try:
                from scipy import signal as sp_signal
                freqs, psd = sp_signal.welch(filtered_signal, fs=self.fps)
                welch_mask = (freqs > 0) & (freqs < 5)
                axes[2].plot(freqs[welch_mask] * 60, psd[welch_mask], 
                           'r-', alpha=0.7, label='Welch PSD')
            except:
                pass
            
            axes[2].axvspan(50, 180, alpha=0.2, color='green', label='Valid HR Range')
        
        axes[2].set_title('Frequency Domain Analysis')
        axes[2].set_xlabel('Heart Rate (BPM)')
        axes[2].set_ylabel('Power')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
            
            # 4. Method comparison
        hr_estimate, confidence, methods = self.heart_rate_estimation()
        
        if methods:
            method_names = list(methods.keys())
            hrs = [methods[m][0] if methods[m][0] is not None else 0 for m in method_names]
            confs = [methods[m][1] for m in method_names]
            
            bars = axes[3].bar(method_names, hrs, alpha=0.7)
            for i, (bar, conf) in enumerate(zip(bars, confs)):
                color_intensity = min(1.0, max(0.0, conf))
                bar.set_color(plt.cm.RdYlGn(color_intensity))
                axes[3].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                           f'{conf:.2f}', ha='center', va='bottom', fontsize=9)
        
        axes[3].set_title('Heart Rate Estimation Methods Comparison')
        axes[3].set_xlabel('Method')
        axes[3].set_ylabel('Heart Rate (BPM)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
            
        # Print results
        print(f"\n=== Phase 2 Analysis Results ===")
        if hr_estimate is not None:
            print(f"Final Estimate: {hr_estimate:.1f} BPM (Confidence: {confidence:.3f})")
            print(f"Signal Quality: Avg = {np.mean(self.signal_quality_scores):.2f}")
            print(f"Method Breakdown:")
            for method, (hr, conf) in methods.items():
                status = "✓" if hr is not None and conf > 0.3 else "✗"
                hr_str = f"{hr:.1f}" if hr is not None else "N/A"
                print(f"  {status} {method.capitalize()}: {hr_str} BPM (conf: {conf:.3f})")
        else:
            print("No reliable heart rate estimate obtained")

# ground truth processing from BVP signal (file format: 3 lines - BVP values, HR values, timestamps)
def ground_truth_processing(gt_path, fps=30):
    try:
        # Read the BVP signal
        with open(gt_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            print("Ground truth file has insufficient data")
            return None
        
        # Try to parse as UBFC format (3 lines: BVP, HR, timestamps)
        try:
            # Second line contains HR values
            hr_line = lines[1].strip()
            hr_values = [float(x) for x in hr_line.split()]
            
            if len(hr_values) > 0:
                # Use median HR as ground truth (more stable than mean)
                gt_hr = np.median(hr_values)
                if 50 <= gt_hr <= 180:
                    print(f"Ground truth HR (from file): {gt_hr:.1f} BPM")
                    return gt_hr
        except (ValueError, IndexError):
            pass
        
        # Fallback: try to extract from BVP signal (first line)
        try:
            bvp_line = lines[0].strip()
            bvp_values = [float(x) for x in bvp_line.split()]
            
            if len(bvp_values) < fps * 5:  # Need at least 5 seconds
                print(f"BVP signal too short: {len(bvp_values)} samples")
                return None
            
            bvp_signal = np.array(bvp_values)
            print(f"Processing BVP signal: {len(bvp_signal)} samples")
            
            # Use multiple estimation methods
            hr_peak = estimate_hr_from_bvp_peaks(bvp_signal, fps)
            hr_fft = estimate_hr_from_bvp_fft(bvp_signal, fps)
            hr_window = estimate_hr_from_bvp_sliding(bvp_signal, fps)
            
            print(f"BVP HR estimates: Peak={hr_peak}, FFT={hr_fft}, Window={hr_window}")
            
            # Combine valid estimates
            valid_estimates = [hr for hr in [hr_peak, hr_fft, hr_window] 
                             if hr is not None and 50 <= hr <= 180]
            
            if valid_estimates:
                avg_hr = np.mean(valid_estimates)
                print(f"Ground truth HR (from BVP): {avg_hr:.1f} BPM")
                return avg_hr
                
        except (ValueError, IndexError):
            pass
        
        print("Could not extract reliable HR from ground truth file")
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

# Estimate HR using sliding window approach with error handling 
def estimate_hr_from_bvp_sliding(bvp_signal, fps=30, window_sec=10):
    try:
        window_size = int(window_sec * fps)
        if len(bvp_signal) < window_size:
            print(f"Sliding window: Signal too short ({len(bvp_signal)} < {window_size})")
            return None
        
        hr_estimates = []
        step_size = max(1, window_size // 4)  # 25% overlap
        
        for i in range(0, len(bvp_signal) - window_size + 1, step_size):
            window = bvp_signal[i:i + window_size]
            
            # Quick FFT-based estimation for this window
            try:
                window_detrended = detrend(window)
                
                # Check for valid signal
                if np.std(window_detrended) < 1e-8:
                    continue
                
                fft_vals = fft(window_detrended)
                freqs = fftfreq(len(window), 1/fps)
                
                hr_mask = (freqs >= 0.83) & (freqs <= 3.0)
                if not np.any(hr_mask):
                    continue
                    
                hr_freqs = freqs[hr_mask]
                hr_mags = np.abs(fft_vals[hr_mask])
                
                if len(hr_mags) == 0:
                    continue
                    
                peak_freq = hr_freqs[np.argmax(hr_mags)]
                hr = peak_freq * 60
                
                if 50 <= hr <= 180:
                    hr_estimates.append(hr) # add valid estimate to list
        
            except Exception:
                continue

        if len(hr_estimates) >= 2:
            # Use median for robustness
            return np.median(hr_estimates)
        elif len(hr_estimates) == 1:
            return hr_estimates[0]
        else:
            print("Sliding window: No valid estimates")
            return None
            
    except Exception as e:
        print(f"Sliding window error: {e}")
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
    
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
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
            # Get heart rate estimation
            hr_estimate, confidence, methods = processor.heart_rate_estimation()
            
            # Process ground truth
            print("\n--- Processing Ground Truth ---")
            gt_hr = ground_truth_processing(subject_info['ground_truth_path'])
            
            # Store result
            result = {
                'subject': subject_info['subject_id'],
                'estimated_hr': hr_estimate,
                'confidence': confidence,
                'ground_truth_hr': gt_hr,
                'methods': methods
            }
            
            # Show detailed visualization
            print("\n--- Generating Detailed Analysis ---")
            processor.visualize_results()
            
            # Final comparison
            print(f"\n=== FINAL RESULTS ===")
            print(f"Estimated HR: {hr_estimate:.1f} BPM (Confidence: {confidence:.3f})")

            if gt_hr:
                error = abs(hr_estimate - gt_hr)
                rel_error = error / gt_hr * 100
                print(f"Ground Truth HR: {gt_hr:.1f} BPM")
                print(f"Absolute Error: {error:.1f} BPM")
                print(f"Relative Error: {rel_error:.1f}%")
                
                # Quality assessment
                if rel_error < 10:
                    print("✓ Excellent accuracy!")
                elif rel_error < 20:
                    print("✓ Good accuracy")
                elif rel_error < 30:
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
            
            # Performance summary for this single subject
            if gt_hr is not None and hr_estimate is not None:
                print(f"\n=== Single Subject Performance ===")
                print(f"Subject: {subject_info['subject_id']}")
                print(f"Absolute error: {error:.1f} BPM")
                print(f"Relative error: {rel_error:.1f}%")
                print(f"Confidence: {confidence:.3f}")
                print(f"Accuracy (±10 BPM): {'✓' if error <= 10 else '✗'}")
                print(f"Accuracy (±15 BPM): {'✓' if error <= 15 else '✗'}")
                
                # Individual method performance
                print(f"\n--- Individual Method Performance ---")
                for method, (hr, conf) in methods.items():
                    if hr is not None:
                        method_error = abs(hr - gt_hr)
                        method_rel_error = method_error / gt_hr * 100
                        print(f"{method.capitalize()}: {method_error:.1f} BPM error ({method_rel_error:.1f}%), conf: {conf:.3f}")
        
        else:
            print("Failed to process video")
            
    except Exception as e:
        print(f"Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
        demo_rppg()