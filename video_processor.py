#!/usr/bin/env python3
"""
Video Processor for Camera Demo
================================

Processes video frames from camera, detects faces, and prepares frames for model input.
"""

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple
from collections import deque


class VideoProcessor:
    """
    Real-time video processor for deception detection.
    
    Captures frames from camera, detects faces, crops and resizes to model input size.
    """
    
    def __init__(
        self,
        camera_id: Optional[int] = 0,
        image_size: Tuple[int, int] = (224, 224),
        max_frames: int = 32,
        device: str = None
    ):
        """
        Initialize video processor.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera, None for video file processing)
            image_size: Target image size (height, width)
            max_frames: Maximum number of frames for model input
            device: Device for processing ('cuda' or 'cpu')
        """
        self.camera_id = camera_id
        self.image_size = image_size
        self.max_frames = max_frames
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=max_frames)
        
        # Face detector (Haar cascade - simple and fast)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Camera
        self.cap = None
        self.is_capturing = False
    
    def start_capture(self):
        """Start capturing from camera."""
        if self.is_capturing:
            return
        
        if self.camera_id is None:
            # Not using camera (for video file processing)
            return
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.camera_id}")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_capturing = True
            print(f"✓ Camera {self.camera_id} started")
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            raise
    
    def stop_capture(self):
        """Stop capturing from camera."""
        if not self.is_capturing:
            return
        
        self.is_capturing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        print("✓ Camera stopped")
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read a frame from camera.
        
        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        if not self.is_capturing or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        return frame
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            (x, y, w, h) bounding box or None if no face detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def crop_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Crop and resize face region.
        
        Args:
            frame: Input frame (BGR format)
            bbox: (x, y, w, h) bounding box
        
        Returns:
            Cropped and resized face image (RGB format)
        """
        x, y, w, h = bbox
        
        # Add padding (20% on each side)
        padding = int(min(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        # Crop
        face_crop = frame[y:y+h, x:x+w]
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        return face_rgb
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a frame: detect face, crop, and add to buffer.
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            Processed face frame (RGB format) or None if no face detected
        """
        # Detect face
        bbox = self.detect_face(frame)
        if bbox is None:
            return None
        
        # Crop face
        face_frame = self.crop_face(frame, bbox)
        
        # Add to buffer
        self.frame_buffer.append(face_frame)
        
        return face_frame
    
    def get_frames_tensor(self) -> torch.Tensor:
        """
        Get frames tensor for model input.
        
        Returns:
            Frames tensor of shape (1, max_frames, 3, H, W) normalized for DINOv3
        """
        if len(self.frame_buffer) == 0:
            # Return zeros if no frames (ensure float32)
            return torch.zeros(
                (1, self.max_frames, 3, self.image_size[0], self.image_size[1]),
                dtype=torch.float32,
                device=self.device
            )
        
        # Convert frames to numpy array (ensure float32)
        frames = np.array(list(self.frame_buffer), dtype=np.float32)  # (T, H, W, C)
        
        # Normalize to [0, 1]
        frames = frames / 255.0
        
        # Apply DINOv3 normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std
        
        # Convert to tensor and rearrange: (T, H, W, C) -> (1, T, C, H, W)
        # Ensure float32 dtype
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T, C, H, W)
        frames_tensor = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
        frames_tensor = frames_tensor.to(self.device)
        
        # Pad or truncate to max_frames
        T = frames_tensor.shape[1]
        if T < self.max_frames:
            # Pad with last frame (ensure float32)
            last_frame = frames_tensor[:, -1:, :, :, :]
            pad_length = self.max_frames - T
            frames_tensor = torch.cat([
                frames_tensor,
                last_frame.repeat(1, pad_length, 1, 1, 1)
            ], dim=1).float()
        elif T > self.max_frames:
            # Truncate (uniform sampling)
            indices = np.linspace(0, T - 1, self.max_frames, dtype=int)
            frames_tensor = frames_tensor[:, indices, :, :, :]
        
        return frames_tensor
    
    def get_display_frame(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """
        Get frame for display with annotations.
        
        Args:
            frame: Input frame (BGR format)
            bbox: Optional face bounding box to draw
        
        Returns:
            Annotated frame for display
        """
        display_frame = frame.copy()
        
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        
        return display_frame


if __name__ == "__main__":
    # Test video processor
    processor = VideoProcessor()
    
    try:
        print("Starting camera test (5 seconds)...")
        processor.start_capture()
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < 5:
            frame = processor.read_frame()
            if frame is not None:
                processed = processor.process_frame(frame)
                if processed is not None:
                    print(f"  Frame processed: {processed.shape}")
                    frames_tensor = processor.get_frames_tensor()
                    print(f"  Frames tensor: {frames_tensor.shape}")
                    break
        
        print("✓ Video processor test successful!")
    finally:
        processor.stop_capture()
