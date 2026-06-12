#!/usr/bin/env python3
"""
Video Processing Script for Multimodal Deception Detection
==========================================================

Processes a video file and adds deception detection annotations.

Usage:
    python process_video.py --video path/to/video.mp4 --checkpoint ../cache/best_model_acc_0_7876_epoch_40.pt
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_loader import load_model_from_checkpoint
from audio_processor import AudioProcessor
from video_processor import VideoProcessor


class VideoProcessorDemo:
    """Process video file with deception detection."""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        inference_interval: float = 0.5
    ):
        """
        Initialize video processor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference ('cuda' or 'cpu')
            inference_interval: Time interval between inferences (seconds)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.inference_interval = inference_interval
        
        # Load model
        print("=" * 80)
        print("Loading Model")
        print("=" * 80)
        self.model = load_model_from_checkpoint(checkpoint_path, device=self.device)
        
        # State
        self.current_prediction = None
        self.current_confidence = None
        self.audio_weight = None
        self.expression_weight = None
        self.last_inference_time = 0
    
    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fps: Optional[float] = None
    ):
        """
        Process video file and save annotated output.
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (if None, auto-generates)
            fps: Output FPS (if None, uses input video FPS)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"❌ Error: Video file not found: {video_path}")
            return
        
        print("\n" + "=" * 80)
        print("Processing Video")
        print("=" * 80)
        print(f"Input video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            return
        
        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / input_fps if input_fps > 0 else 0
        
        print(f"Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {input_fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.2f}s")
        
        # Use input FPS if not specified
        if fps is None:
            fps = input_fps if input_fps > 0 else 30.0
        
        # Generate output path
        if output_path is None:
            output_dir = video_path.parent / "processed"
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{video_path.stem}_annotated_{timestamp}.mp4"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Output video: {output_path}")
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Initialize processors for frame processing
        video_processor = VideoProcessor(
            camera_id=None,  # Not using camera
            image_size=(224, 224),
            max_frames=32,
            device=self.device
        )
        
        # Extract audio from video
        print("\nExtracting audio from video...")
        audio_samples = self._extract_audio_from_video(video_path)
        audio_sample_rate = 16000  # Standard rate
        
        print(f"Extracted {len(audio_samples)} audio samples ({len(audio_samples)/audio_sample_rate:.2f}s)")
        
        # Process video frame by frame
        print("\nProcessing frames...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = frame_count / fps if fps > 0 else frame_count / 30.0
            
            # Process frame for model input
            bbox = video_processor.detect_face(frame)
            if bbox is not None:
                video_processor.process_frame(frame)
            
            # Run inference periodically
            if current_time - self.last_inference_time >= self.inference_interval:
                self._run_inference(audio_samples, current_time, audio_sample_rate, video_processor)
                self.last_inference_time = current_time
            
            # Create annotated frame
            annotated_frame = self._create_display_frame(frame, bbox)
            
            # Write frame
            out.write(annotated_frame)
            
            # Progress
            if frame_count % 30 == 0 or frame_count == total_frames:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"  Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
        
        cap.release()
        out.release()
        
        print(f"\n✓ Video processing complete!")
        print(f"  Processed {frame_count} frames")
        print(f"  Output saved to: {output_path}")
        
        # Combine with original audio if available
        if len(audio_samples) > 0:
            self._combine_video_audio(str(output_path), audio_samples, audio_sample_rate, video_path)
    
    def _extract_audio_from_video(self, video_path: Path) -> np.ndarray:
        """Extract audio from video file."""
        try:
            import librosa
            # Load audio from video
            y, sr = librosa.load(str(video_path), sr=16000, mono=True)
            return y.astype(np.float32)
        except Exception as e:
            print(f"⚠️  Warning: Could not extract audio: {e}")
            print("   Video will be saved without audio")
            return np.array([], dtype=np.float32)
    
    def _run_inference(
        self,
        audio_samples: np.ndarray,
        current_time: float,
        sample_rate: int,
        video_processor: VideoProcessor
    ):
        """Run model inference on current audio and video."""
        try:
            # Get audio segment for current time
            start_sample = int(max(0, (current_time - 2.0) * sample_rate))  # 2 seconds before
            end_sample = int(current_time * sample_rate)
            audio_segment = audio_samples[start_sample:end_sample]
            
            # Pad if needed
            min_samples = int(sample_rate * 0.5)
            if len(audio_segment) < min_samples:
                audio_segment = np.pad(audio_segment, (0, min_samples - len(audio_segment)), mode='constant')
            
            # Convert to tensor and extract MFCC
            audio_tensor = torch.from_numpy(audio_segment.astype(np.float32)).unsqueeze(0)
            
            # Get MFCC features using librosa (since we have numpy array)
            try:
                import librosa
                mfcc = librosa.feature.mfcc(
                    y=audio_segment,
                    sr=sample_rate,
                    n_mfcc=13,
                    hop_length=512,
                    n_fft=1024,
                    n_mels=40,
                    fmin=0.0,
                    fmax=8000.0
                ).T  # (T, 13)
                
                # Pad or truncate to 700
                if mfcc.shape[0] < 700:
                    mfcc = np.pad(mfcc, ((0, 700 - mfcc.shape[0]), (0, 0)), mode='constant')
                elif mfcc.shape[0] > 700:
                    start = (mfcc.shape[0] - 700) // 2
                    mfcc = mfcc[start:start + 700]
                
                # Normalize
                mfcc_mean = mfcc.mean(axis=0, keepdims=True)
                mfcc_std = mfcc.std(axis=0, keepdims=True) + 1e-8
                mfcc = (mfcc - mfcc_mean) / mfcc_std
                
                audio_input = torch.from_numpy(mfcc.astype(np.float32)).unsqueeze(0).to(self.device)
            except Exception as e:
                print(f"Error extracting MFCC: {e}")
                audio_input = torch.zeros((1, 700, 13), dtype=torch.float32, device=self.device)
            
            # Get video frames
            video_input = video_processor.get_frames_tensor()
            
            # Check if we have enough data
            if len(video_processor.frame_buffer) < 5:
                return
            
            # Run inference
            with torch.no_grad():
                audio_input = audio_input.float()
                video_input = video_input.float()
                outputs = self.model(audio_input, video_input)
                logits = outputs['logits']
                
                # Handle different logits shapes
                if logits.dim() > 1:
                    logits = logits.squeeze()
                
                # Convert to probability
                probability = torch.sigmoid(logits).item()
                
                # Extract modality weights if available
                if 'modality_weights' in outputs and outputs['modality_weights'] is not None:
                    weights = outputs['modality_weights'][0]
                    self.audio_weight = weights[0].item()
                    self.expression_weight = weights[1].item()
                else:
                    self.audio_weight = 0.5
                    self.expression_weight = 0.5
                
                # Update state
                self.current_prediction = "DECEPTION" if probability > 0.5 else "TRUTH"
                self.current_confidence = abs(probability - 0.5) * 2
        
        except Exception as e:
            print(f"Error during inference: {e}")
    
    def _create_display_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[tuple]
    ) -> np.ndarray:
        """Create annotated frame (same as camera demo)."""
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
                    (255, 255, 255),
                    small_thickness
                )
                y_pos += small_height + line_spacing
        
        return display
    
    def _combine_video_audio(
        self,
        video_path: str,
        audio_samples: np.ndarray,
        sample_rate: int,
        original_video_path: Path
    ):
        """Combine processed video with original audio."""
        import subprocess
        import wave
        from pathlib import Path
        
        video_path_obj = Path(video_path)
        audio_path = video_path_obj.parent / f"{video_path_obj.stem}_temp_audio.wav"
        final_path = video_path_obj.parent / f"{video_path_obj.stem}_with_audio.mp4"
        
        try:
            # Save audio to WAV
            print("\nCombining video with audio...")
            audio_int16 = (np.clip(audio_samples, -1.0, 1.0) * 32767).astype(np.int16)
            
            with wave.open(str(audio_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            # Use ffmpeg to combine
            cmd = [
                'ffmpeg',
                '-y',
                '-i', video_path,
                '-i', str(audio_path),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-pix_fmt', 'yuv420p',
                '-async', '1',
                '-vsync', 'cfr',
                str(final_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"✓ Combined video saved: {final_path}")
                # Remove temporary files
                Path(video_path).unlink()
                audio_path.unlink()
            else:
                print(f"⚠️  FFmpeg warning: {result.stderr}")
                print(f"   Annotated video saved: {video_path}")
        
        except FileNotFoundError:
            print("⚠️  FFmpeg not found. Video saved without audio:")
            print(f"   {video_path}")
        except Exception as e:
            print(f"⚠️  Error combining audio: {e}")
            print(f"   Annotated video saved: {video_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Video with Multimodal Deception Detection"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../cache/best_model_acc_0_7876_epoch_40.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output video (if None, auto-generates)"
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
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS (if None, uses input video FPS)"
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
    
    # Create processor and process video
    processor = VideoProcessorDemo(
        checkpoint_path=str(checkpoint_path),
        device=args.device,
        inference_interval=args.interval
    )
    
    processor.process_video(
        video_path=args.video,
        output_path=args.output,
        fps=args.fps
    )


if __name__ == "__main__":
    main()
