#!/usr/bin/env python3
"""
Audio Processor for Camera Demo
================================

Processes audio from microphone in real-time and extracts MFCC features.
"""

import numpy as np
import torch
from collections import deque
import threading
import queue
import pyaudio

# Try to import torchaudio, fall back to librosa
try:
    import torchaudio.transforms as T
    USE_TORCHAUDIO = True
except (ImportError, OSError) as e:
    USE_TORCHAUDIO = False
    try:
        import librosa
        USE_LIBROSA = True
    except ImportError:
        USE_LIBROSA = False
        print("⚠️  Warning: Neither torchaudio nor librosa available. MFCC extraction will fail.")


class AudioProcessor:
    """
    Real-time audio processor for deception detection.
    
    Captures audio from microphone, extracts MFCC features, and maintains
    a sliding window buffer for model input.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        max_length: int = 700,
        chunk_size: int = 1024,
        device: str = None
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_mfcc: Number of MFCC coefficients
            max_length: Maximum sequence length (timesteps)
            chunk_size: Audio chunk size for streaming
            device: Device for processing ('cuda' or 'cpu')
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Audio buffer (stores raw samples)
        self.audio_buffer = deque(maxlen=sample_rate * 10)  # 10 seconds max
        
        # Separate recording buffer for video export (unlimited, cleared on reset)
        self.recording_buffer = []
        self.is_recording_for_export = False
        self.recording_lock = threading.Lock()  # Thread-safe access to recording buffer
        
        # MFCC transform - use torchaudio if available, otherwise librosa
        if USE_TORCHAUDIO:
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': 1024,
                    'hop_length': 512,
                    'n_mels': 40,
                    'f_min': 0.0,
                    'f_max': 8000.0
                }
            ).to(self.device)
            self.use_torchaudio = True
        elif USE_LIBROSA:
            self.use_torchaudio = False
            self.hop_length = 512
            print("✓ Using librosa for MFCC extraction (torchaudio not available)")
        else:
            raise RuntimeError("Neither torchaudio nor librosa available. Cannot extract MFCC features.")
        
        # PyAudio setup
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Threading
        self.audio_queue = queue.Queue()
        self.processing_thread = None
    
    def start_recording(self):
        """Start recording audio from microphone."""
        if self.is_recording:
            return
        
        try:
            # Use input_device_index to avoid echo/feedback
            # Try to find a good input device
            input_device_index = None
            try:
                # Get default input device
                input_device_index = self.audio.get_default_input_device_info()['index']
            except:
                pass
            
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                start=False  # Start manually to ensure proper initialization
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print("✓ Audio recording started")
        except Exception as e:
            print(f"❌ Error starting audio recording: {e}")
            print("  Make sure microphone is connected and accessible")
            raise
    
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.audio.terminate()
        print("✓ Audio recording stopped")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # Basic echo/noise reduction: apply high-pass filter to reduce low-frequency feedback
        # Simple approach: reduce very low frequencies that might be feedback
        if len(audio_data) > 0:
            # Simple high-pass: subtract moving average (reduces DC and low freq)
            if len(audio_data) > 10:
                moving_avg = np.convolve(audio_data, np.ones(10)/10, mode='same')
                audio_data = audio_data - moving_avg * 0.3  # Reduce low frequencies
        
        # Normalize to prevent clipping
        max_val = np.abs(audio_data).max()
        if max_val > 0.95:
            audio_data = audio_data * (0.95 / max_val)
        
        # Add to sliding buffer for model inference
        self.audio_buffer.extend(audio_data)
        
        # Also add to recording buffer if recording for export (thread-safe)
        if self.is_recording_for_export:
            with self.recording_lock:
                self.recording_buffer.extend(audio_data.tolist())
        
        return (None, pyaudio.paContinue)
    
    def start_recording_for_export(self):
        """Start recording audio for video export."""
        self.is_recording_for_export = True
        self.recording_buffer = []
    
    def stop_recording_for_export(self):
        """Stop recording audio for video export."""
        self.is_recording_for_export = False
    
    def get_recorded_audio(self) -> np.ndarray:
        """Get all recorded audio samples for export (thread-safe)."""
        with self.recording_lock:
            if len(self.recording_buffer) == 0:
                return np.array([], dtype=np.float32)
            # Return a copy to avoid issues
            return np.array(self.recording_buffer, dtype=np.float32)
    
    def clear_recording_buffer(self):
        """Clear the recording buffer (thread-safe)."""
        with self.recording_lock:
            self.recording_buffer = []
    
    def get_mfcc_features(self) -> torch.Tensor:
        """
        Extract MFCC features from current audio buffer.
        
        Returns:
            MFCC features tensor of shape (1, max_length, n_mfcc)
        """
        if len(self.audio_buffer) == 0:
            # Return zeros if no audio (ensure float32)
            return torch.zeros((1, self.max_length, self.n_mfcc), dtype=torch.float32, device=self.device)
        
        # Convert buffer to tensor
        audio_samples = np.array(self.audio_buffer)
        
        # Ensure minimum length (pad if needed)
        min_samples = int(self.sample_rate * 0.5)  # At least 0.5 seconds
        if len(audio_samples) < min_samples:
            # Pad with zeros
            pad_length = min_samples - len(audio_samples)
            audio_samples = np.pad(audio_samples, (0, pad_length), mode='constant')
        
        # Extract MFCC using torchaudio or librosa
        if self.use_torchaudio:
            # Convert to tensor and add batch dimension (ensure float32)
            audio_tensor = torch.from_numpy(audio_samples.astype(np.float32)).unsqueeze(0).to(self.device)
            
            # Extract MFCC
            with torch.no_grad():
                mfcc = self.mfcc_transform(audio_tensor)  # (1, n_mfcc, T)
                mfcc = mfcc.transpose(1, 2)  # (1, T, n_mfcc)
        else:
            # Use librosa for MFCC extraction
            import librosa  # Import here to ensure it's available
            mfcc = librosa.feature.mfcc(
                y=audio_samples.astype(np.float32),
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=1024,
                n_mels=40,
                fmin=0.0,
                fmax=8000.0
            )  # (n_mfcc, T)
            mfcc = mfcc.T  # Transpose to (T, n_mfcc)
            # Convert to tensor and add batch dimension
            mfcc = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # Normalize
        with torch.no_grad():
            mfcc_mean = mfcc.mean(dim=1, keepdim=True)
            mfcc_std = mfcc.std(dim=1, keepdim=True) + 1e-8
            mfcc = (mfcc - mfcc_mean) / mfcc_std
            
            # Pad or truncate to max_length
            T = mfcc.shape[1]
            if T < self.max_length:
                # Pad with zeros (ensure float32)
                pad_length = self.max_length - T
                mfcc = torch.cat([
                    mfcc,
                    torch.zeros((1, pad_length, self.n_mfcc), dtype=torch.float32, device=self.device)
                ], dim=1)
            elif T > self.max_length:
                # Truncate (center crop)
                start = (T - self.max_length) // 2
                mfcc = mfcc[:, start:start + self.max_length, :]
        
        return mfcc
    
    def get_audio_level(self) -> float:
        """Get current audio level (RMS) for visualization."""
        if len(self.audio_buffer) == 0:
            return 0.0
        
        audio_samples = np.array(self.audio_buffer)
        rms = np.sqrt(np.mean(audio_samples ** 2))
        return float(rms)


if __name__ == "__main__":
    # Test audio processor
    import time
    
    processor = AudioProcessor()
    
    try:
        print("Starting audio test (5 seconds)...")
        processor.start_recording()
        
        for i in range(5):
            time.sleep(1)
            mfcc = processor.get_mfcc_features()
            audio_level = processor.get_audio_level()
            print(f"  Second {i+1}: MFCC shape={mfcc.shape}, Audio level={audio_level:.4f}")
        
        print("✓ Audio processor test successful!")
    finally:
        processor.stop_recording()
