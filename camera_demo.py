#!/usr/bin/env python3
"""
Camera Demo for Multimodal Deception Detection
==============================================

Real-time deception detection using camera and microphone input.

Usage:
    python camera_demo.py --checkpoint ../cache/best_model_acc_0_7876_epoch_40.pt
"""

import sys
from pathlib import Path
import argparse
import cv2
import torch
import numpy as np
import time
from typing import Optional
from datetime import datetime
import wave
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_model_from_checkpoint
from audio_processor import AudioProcessor
from video_processor import VideoProcessor


class CameraDemo:
    """Main demo application for real-time deception detection."""
    
    def __init__(
        self,
        checkpoint_path: str,
        camera_id: int = 0,
        device: str = None,
        inference_interval: float = 0.5  # Run inference every 0.5 seconds
    ):
        """
        Initialize camera demo.
        
        Args:
            checkpoint_path: Path to model checkpoint
            camera_id: Camera device ID
            device: Device for inference ('cuda' or 'cpu')
            inference_interval: Time interval between inferences (seconds)
        """
        self.checkpoint_path = checkpoint_path
        self.camera_id = camera_id
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_interval = inference_interval
        
        # Load model
        print("=" * 80)
        print("Loading Model")
        print("=" * 80)
        self.model = load_model_from_checkpoint(checkpoint_path, device=self.device)
        
        # Initialize processors
        print("\n" + "=" * 80)
        print("Initializing Processors")
        print("=" * 80)
        self.audio_processor = AudioProcessor(device=self.device)
        self.video_processor = VideoProcessor(
            camera_id=camera_id,
            device=self.device
        )
        
        # State
        self.is_running = False
        self.last_inference_time = 0
        self.current_prediction = None
        self.current_confidence = None
        self.audio_weight = None  # Audio modality contribution (0-1)
        self.expression_weight = None  # Expression modality contribution (0-1)
        
        # Recording
        self.is_recording = False
        self.video_writer = None
        self.recorded_frames = []
        self.frame_timestamps = []  # Store frame timestamps for sync
        self.recording_fps = 30.0
        self.audio_sample_rate = 16000
        self.output_dir = Path(__file__).parent / "recordings"
        self.output_dir.mkdir(exist_ok=True)
    
    def run(self):
        """Run the demo."""
        print("\n" + "=" * 80)
        print("Starting Camera Demo")
        print("=" * 80)
        print("\nControls:")
        print("  - Press 'q' to quit and save recording")
        print("  - Press 'r' to reset buffers")
        print("  - Make sure your face is visible and speak clearly")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        try:
            # Start processors
            self.audio_processor.start_recording()
            self.video_processor.start_capture()
            
            # Start recording
            self._start_recording()
            
            self.is_running = True
            
            # Main loop
            frame_count = 0
            while self.is_running:
                # Read frame
                frame = self.video_processor.read_frame()
                if frame is None:
                    continue
                
                frame_count += 1
                
                # Process frame
                bbox = self.video_processor.detect_face(frame)
                if bbox is not None:
                    self.video_processor.process_frame(frame)
                
                # Run inference periodically
                current_time = time.time()
                if current_time - self.last_inference_time >= self.inference_interval:
                    self._run_inference()
                    self.last_inference_time = current_time
                
                # Display
                display_frame = self._create_display_frame(frame, bbox)
                cv2.imshow('Deception Detection Demo', display_frame)
                
                # Record frame (audio is recorded automatically in callback)
                if self.is_recording:
                    self._record_frame(display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_buffers()
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            self.cleanup()
    
    def _run_inference(self):
        """Run model inference on current audio and video buffers."""
        try:
            # Get inputs
            audio_input = self.audio_processor.get_mfcc_features()  # (1, 700, 13)
            video_input = self.video_processor.get_frames_tensor()  # (1, 32, 3, 224, 224)
            
            # Check if we have enough data
            if len(self.video_processor.frame_buffer) < 5:
                # Not enough frames yet
                return
            
            # Run inference (ensure inputs are float32)
            with torch.no_grad():
                # Ensure float32 dtype
                audio_input = audio_input.float()
                video_input = video_input.float()
                outputs = self.model(audio_input, video_input)
                logits = outputs['logits']  # (1,) or (1, 1)
                
                # Handle different logits shapes
                if logits.dim() > 1:
                    logits = logits.squeeze()
                
                # Convert to probability
                probability = torch.sigmoid(logits).item()
                
                # Extract modality weights if available
                if 'modality_weights' in outputs and outputs['modality_weights'] is not None:
                    weights = outputs['modality_weights'][0]  # (2,)
                    self.audio_weight = weights[0].item()  # Audio contribution
                    self.expression_weight = weights[1].item()  # Expression contribution
                else:
                    # Default to equal weights if not available
                    self.audio_weight = 0.5
                    self.expression_weight = 0.5
                
                # Update state
                self.current_prediction = "DECEPTION" if probability > 0.5 else "TRUTH"
                self.current_confidence = abs(probability - 0.5) * 2  # Scale to [0, 1]
        
        except Exception as e:
            print(f"Error during inference: {e}")
    
    def _create_display_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[tuple]
    ) -> np.ndarray:
        """Create display frame with annotations."""
        display = frame.copy()
        
        # Draw face bounding box
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw prediction and modality contributions
        if self.current_prediction is not None:
            # Main prediction text
            text = f"{self.current_prediction}"
            if self.current_confidence is not None:
                text += f" ({self.current_confidence*100:.1f}%)"
            
            # Modality contribution texts
            modality_texts = []
            if self.audio_weight is not None:
                modality_texts.append(f"Audio: {self.audio_weight*100:.1f}%")
            if self.expression_weight is not None:
                modality_texts.append(f"Expression: {self.expression_weight*100:.1f}%")
            
            # Get text sizes
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            small_font_scale = 0.6
            small_thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Calculate total height needed
            line_spacing = 5
            total_height = text_height + baseline
            if modality_texts:
                (_, small_height), _ = cv2.getTextSize(
                    modality_texts[0], font, small_font_scale, small_thickness
                )
                total_height += len(modality_texts) * (small_height + line_spacing) + line_spacing
            
            # Choose color based on prediction
            if self.current_prediction == "DECEPTION":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Calculate max width for background
            max_width = text_width
            for mod_text in modality_texts:
                (mod_width, _), _ = cv2.getTextSize(mod_text, font, small_font_scale, small_thickness)
                max_width = max(max_width, mod_width)
            
            # Draw background rectangle
            cv2.rectangle(
                display,
                (10, 10),
                (25 + max_width, 15 + total_height),
                (0, 0, 0),
                -1
            )
            
            # Draw main prediction text
            y_pos = 35
            cv2.putText(
                display,
                text,
                (15, y_pos),
                font,
                font_scale,
                color,
                thickness
            )
            
            # Draw modality contributions
            y_pos += text_height + line_spacing + 5
            for mod_text in modality_texts:
                cv2.putText(
                    display,
                    mod_text,
                    (15, y_pos),
                    font,
                    small_font_scale,
                    (255, 255, 255),  # White text
                    small_thickness
                )
                y_pos += small_height + line_spacing
        
        # Draw audio level
        audio_level = self.audio_processor.get_audio_level()
        audio_bar_width = int(audio_level * 200)  # Scale to 200 pixels
        cv2.rectangle(display, (10, display.shape[0] - 30), (10 + audio_bar_width, display.shape[0] - 10), (255, 0, 0), -1)
        cv2.putText(
            display,
            "Audio Level",
            (10, display.shape[0] - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Draw frame count
        frame_count = len(self.video_processor.frame_buffer)
        cv2.putText(
            display,
            f"Frames: {frame_count}/{self.video_processor.max_frames}",
            (display.shape[1] - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        
        # Draw instructions
        cv2.putText(
            display,
            "Press 'q' to quit, 'r' to reset",
            (10, display.shape[0] - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return display
    
    def _reset_buffers(self):
        """Reset audio and video buffers."""
        self.audio_processor.audio_buffer.clear()
        self.audio_processor.clear_recording_buffer()
        self.video_processor.frame_buffer.clear()
        self.current_prediction = None
        self.current_confidence = None
        self.audio_weight = None
        self.expression_weight = None
        print("✓ Buffers reset")
    
    def _start_recording(self):
        """Start recording video and audio."""
        self.is_recording = True
        self.recorded_frames = []
        self.frame_timestamps = []
        self.recording_start_time = time.time()
        self.audio_sample_rate = self.audio_processor.sample_rate
        
        # Start audio recording for export
        self.audio_processor.start_recording_for_export()
        
        print("✓ Recording started")
    
    def _record_frame(self, frame: np.ndarray):
        """Record a video frame with timestamp."""
        if frame is not None:
            # Convert BGR to RGB for storage (we'll convert back when saving)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.recorded_frames.append(frame_rgb.copy())
            # Store timestamp for this frame
            self.frame_timestamps.append(time.time() - self.recording_start_time)
    
    def _record_audio(self):
        """Audio is recorded continuously in the callback, no action needed here."""
        # Audio recording happens automatically in audio_processor callback
        # when is_recording_for_export is True
        pass
    
    def _save_recording(self):
        """Save recorded video and audio to file."""
        # Stop audio recording
        self.audio_processor.stop_recording_for_export()
        
        # Get recorded audio
        recorded_audio = self.audio_processor.get_recorded_audio()
        
        if not self.is_recording or (len(self.recorded_frames) == 0 and len(recorded_audio) == 0):
            print("⚠️  No recording to save")
            return
        
        print("\n" + "=" * 80)
        print("Saving Recording")
        print("=" * 80)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = self.output_dir / f"demo_recording_{timestamp}.mp4"
        audio_path = self.output_dir / f"demo_audio_{timestamp}.wav"
        final_path = self.output_dir / f"demo_with_audio_{timestamp}.mp4"
        
        try:
            # Calculate actual video duration and FPS
            video_duration = 0.0
            actual_fps = self.recording_fps
            
            if len(self.frame_timestamps) > 1:
                video_duration = self.frame_timestamps[-1]
                # Calculate actual FPS from timestamps
                if video_duration > 0:
                    actual_fps = (len(self.frame_timestamps) - 1) / video_duration
                    # Clamp to reasonable range
                    actual_fps = max(10.0, min(60.0, actual_fps))
            
            # Save video
            if len(self.recorded_frames) > 0:
                print(f"Saving video ({len(self.recorded_frames)} frames, {video_duration:.2f}s, {actual_fps:.1f} fps)...")
                height, width = self.recorded_frames[0].shape[:2]
                
                # Use H.264 codec (better compatibility)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    actual_fps,
                    (width, height)
                )
                
                # Write all frames
                for frame in self.recorded_frames:
                    # Convert RGB back to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)
                
                video_writer.release()
                print(f"✓ Video saved: {video_path}")
            
            # Save audio - match duration to video
            if len(recorded_audio) > 0:
                audio_array = recorded_audio.copy()
                
                # Apply additional normalization and echo reduction
                # Normalize audio levels
                max_amp = np.abs(audio_array).max()
                if max_amp > 0:
                    # Normalize to 80% to prevent clipping
                    audio_array = audio_array * (0.8 / max_amp)
                
                # Calculate expected audio samples for video duration
                if video_duration > 0:
                    expected_audio_samples = int(video_duration * self.audio_sample_rate)
                    
                    # Trim or pad audio to match video duration exactly
                    if len(audio_array) > expected_audio_samples:
                        # Trim excess audio (keep beginning, trim end)
                        audio_array = audio_array[:expected_audio_samples]
                        print(f"Trimming audio from {len(recorded_audio)} to {expected_audio_samples} samples to match video")
                    elif len(audio_array) < expected_audio_samples:
                        # Pad with silence at the end
                        pad_length = expected_audio_samples - len(audio_array)
                        audio_array = np.pad(audio_array, (0, pad_length), mode='constant', constant_values=0.0)
                        print(f"Padding audio from {len(recorded_audio)} to {expected_audio_samples} samples to match video")
                else:
                    # No video, use all audio
                    print(f"Using all recorded audio ({len(audio_array)} samples)")
                
                print(f"Saving audio ({len(audio_array)} samples, {len(audio_array)/self.audio_sample_rate:.2f}s)...")
                
                # Normalize to [-1, 1] if needed
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    audio_array = np.clip(audio_array, -1.0, 1.0)
                
                # Convert to int16 for WAV file
                audio_int16 = (audio_array * 32767).astype(np.int16)
                
                with wave.open(str(audio_path), 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self.audio_sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                print(f"✓ Audio saved: {audio_path}")
            
            # Combine video and audio using ffmpeg
            if len(self.recorded_frames) > 0 and len(recorded_audio) > 0:
                print("Combining video and audio...")
                try:
                    # Use ffmpeg to combine
                    cmd = [
                        'ffmpeg',
                        '-y',  # Overwrite output file
                        '-i', str(video_path),
                        '-i', str(audio_path),
                        '-c:v', 'libx264',  # Re-encode video for better sync
                        '-c:a', 'aac',      # Encode audio as AAC
                        '-map', '0:v:0',    # Map video stream
                        '-map', '1:a:0',    # Map audio stream
                        '-shortest',        # Finish encoding when shortest input ends
                        '-pix_fmt', 'yuv420p',  # Ensure compatibility
                        '-async', '1',      # Audio sync method
                        '-vsync', 'cfr',    # Constant frame rate
                        str(final_path)
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"✓ Combined video saved: {final_path}")
                        # Clean up temporary files
                        video_path.unlink()
                        audio_path.unlink()
                        print(f"✓ Temporary files cleaned up")
                    else:
                        print(f"⚠️  FFmpeg warning: {result.stderr}")
                        print(f"   Video saved separately: {video_path}")
                        print(f"   Audio saved separately: {audio_path}")
                
                except FileNotFoundError:
                    print("⚠️  FFmpeg not found. Video and audio saved separately:")
                    print(f"   Video: {video_path}")
                    print(f"   Audio: {audio_path}")
                    print("   Install ffmpeg to combine them: sudo apt-get install ffmpeg")
                except subprocess.TimeoutExpired:
                    print("⚠️  FFmpeg timed out. Files saved separately.")
                except Exception as e:
                    print(f"⚠️  Error combining files: {e}")
                    print(f"   Video: {video_path}")
                    print(f"   Audio: {audio_path}")
            
            elif len(self.recorded_frames) > 0:
                # Only video
                video_path.rename(final_path)
                print(f"✓ Video saved: {final_path}")
            elif len(self.recorded_audio) > 0:
                # Only audio
                audio_path.rename(final_path.with_suffix('.wav'))
                print(f"✓ Audio saved: {final_path.with_suffix('.wav')}")
        
        except Exception as e:
            print(f"❌ Error saving recording: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n" + "=" * 80)
        print("Cleaning Up")
        print("=" * 80)
        self.is_running = False
        
        # Stop recording and save
        if self.is_recording:
            # Small delay to ensure all audio is captured
            time.sleep(0.1)
            self._save_recording()
            self.is_recording = False
        
        self.audio_processor.stop_recording()
        self.video_processor.stop_capture()
        cv2.destroyAllWindows()
        print("✓ Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Camera Demo for Multimodal Deception Detection"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../cache/best_model_acc_0_7876_epoch_40.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Inference interval in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Resolve checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Create and run demo
    demo = CameraDemo(
        checkpoint_path=str(checkpoint_path),
        camera_id=args.camera,
        device=args.device,
        inference_interval=args.interval
    )
    
    demo.run()


if __name__ == "__main__":
    main()
