#!/usr/bin/env python3
"""
DINOv3 Feature Extraction Script
==================================

Extracts DINOv3 features from a single image.

Usage:
    python extract_dinov3_features.py --image path/to/image.jpg
    python extract_dinov3_features.py --image path/to/image.jpg --output features.npy
"""

import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


class DINOv3FeatureExtractor:
    """Extract DINOv3 features from images."""
    
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        device: str = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize DINOv3 feature extractor.
        
        Args:
            model_name: DINOv3 model name (dinov3_vits16, dinov3_vitb14, etc.)
            device: Device for processing ('cuda' or 'cpu')
            image_size: Target image size (height, width)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        
        # Load DINOv3 encoder
        print("=" * 80)
        print("Loading DINOv3 Encoder")
        print("=" * 80)
        self.encoder = self._load_dinov3_encoder()
        
        # ImageNet normalization (used by DINOv3)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
    
    def _load_dinov3_encoder(self) -> nn.Module:
        """Load DINOv3 encoder."""
        # Try to load from micro_expression_branch
        dinov3_encoder_path = Path(__file__).parent
        
        if dinov3_encoder_path.exists():
            try:
                sys.path.insert(0, str(dinov3_encoder_path))
                from encoder import DinoV3Encoder
                
                encoder = DinoV3Encoder(
                    model_name=self.model_name,
                    freeze=True,
                    use_patch_tokens=False,
                    device=self.device
                )
                print(f"✓ Loaded DINOv3 encoder: {self.model_name}")
                return encoder
            except Exception as e:
                print(f"⚠️  Could not load DINOv3 from micro_expression_branch: {e}")
        
        # Try to load from timm
        try:
            import timm
            encoder = timm.create_model(
                self.model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                global_pool=''
            )
            encoder = encoder.to(self.device)
            encoder.eval()
            print(f"✓ Loaded DINOv3 encoder from timm: {self.model_name}")
            return encoder
        except Exception as e:
            print(f"⚠️  Could not load DINOv3 from timm: {e}")
        
        # Fallback: try torch.hub
        try:
            encoder = torch.hub.load('facebookresearch/dinov2', self.model_name)
            encoder = encoder.to(self.device)
            encoder.eval()
            print(f"✓ Loaded DINOv3 encoder from torch.hub: {self.model_name}")
            return encoder
        except Exception as e:
            print(f"⚠️  Could not load DINOv3 from torch.hub: {e}")
        
        # Last resort: placeholder
        print("⚠️  Using placeholder encoder (not real DINOv3)")
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 384)
        ).to(self.device)
        return encoder
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image for DINOv3.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor (1, 3, H, W)
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to target size
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor and change from HWC to CHW
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize with ImageNet statistics
        img_tensor = (img_tensor - self.mean) / self.std
        
        return img_tensor.to(self.device)
    
    def extract_features(
        self,
        image_path: str,
        return_patch_tokens: bool = False
    ) -> dict:
        """
        Extract DINOv3 features from image.
        
        Args:
            image_path: Path to image file
            return_patch_tokens: Whether to return patch tokens (spatial features)
        
        Returns:
            Dictionary with:
                - cls_token: CLS token features (1, D)
                - patch_tokens: Patch tokens (1, N, D) if return_patch_tokens=True
                - feature_dim: Dimension of features
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        
        # Extract features
        with torch.no_grad():
            if hasattr(self.encoder, '__call__'):
                # Try to call with return_patch_tokens parameter
                try:
                    if return_patch_tokens:
                        cls_token, patch_tokens = self.encoder(img_tensor, return_patch_tokens=True)
                        features = {
                            'cls_token': cls_token.cpu().numpy(),
                            'patch_tokens': patch_tokens.cpu().numpy(),
                            'feature_dim': cls_token.shape[-1]
                        }
                    else:
                        cls_token, _ = self.encoder(img_tensor, return_patch_tokens=False)
                        features = {
                            'cls_token': cls_token.cpu().numpy(),
                            'feature_dim': cls_token.shape[-1]
                        }
                except:
                    # Fallback: just call encoder
                    output = self.encoder(img_tensor)
                    if isinstance(output, tuple):
                        cls_token = output[0]
                    else:
                        cls_token = output
                    
                    # Handle different output shapes
                    if cls_token.dim() > 2:
                        # Global average pool if needed
                        cls_token = torch.mean(cls_token.view(cls_token.shape[0], cls_token.shape[1], -1), dim=2)
                    
                    features = {
                        'cls_token': cls_token.cpu().numpy(),
                        'feature_dim': cls_token.shape[-1]
                    }
            else:
                # Standard PyTorch model
                output = self.encoder(img_tensor)
                if isinstance(output, tuple):
                    cls_token = output[0]
                else:
                    cls_token = output
                
                # Handle different output shapes
                if cls_token.dim() > 2:
                    # Global average pool if needed
                    cls_token = torch.mean(cls_token.view(cls_token.shape[0], cls_token.shape[1], -1), dim=2)
                
                features = {
                    'cls_token': cls_token.cpu().numpy(),
                    'feature_dim': cls_token.shape[-1]
                }
        
        return features
    
    def visualize_features(
        self,
        features: dict,
        output_path: Optional[str] = None
    ):
        """
        Visualize extracted features.
        
        Args:
            features: Features dictionary from extract_features
            output_path: Path to save visualization (optional)
        """
        cls_token = features['cls_token']
        
        print("\n" + "=" * 80)
        print("Feature Information")
        print("=" * 80)
        print(f"Feature shape: {cls_token.shape}")
        print(f"Feature dimension: {features['feature_dim']}")
        print(f"Feature range: [{cls_token.min():.4f}, {cls_token.max():.4f}]")
        print(f"Feature mean: {cls_token.mean():.4f}")
        print(f"Feature std: {cls_token.std():.4f}")
        
        if 'patch_tokens' in features:
            patch_tokens = features['patch_tokens']
            print(f"\nPatch tokens shape: {patch_tokens.shape}")
            print(f"Number of patches: {patch_tokens.shape[1]}")
        
        # Save visualization if requested
        if output_path:
            try:
                # Create a simple visualization
                import matplotlib.pyplot as plt
            except ImportError:
                print(f"⚠️  matplotlib not available. Skipping visualization.")
                print(f"   Install with: pip install matplotlib")
                return
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot feature values
            axes[0].plot(cls_token[0])
            axes[0].set_title('DINOv3 CLS Token Features')
            axes[0].set_xlabel('Feature Dimension')
            axes[0].set_ylabel('Feature Value')
            axes[0].grid(True)
            
            # Plot feature distribution
            axes[1].hist(cls_token[0], bins=50, edgecolor='black')
            axes[1].set_title('Feature Value Distribution')
            axes[1].set_xlabel('Feature Value')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {output_path}")
            plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 Features from Image"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save features (.npy file)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dinov3_vits16",
        help="DINOv3 model name (default: dinov3_vits16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for processing ('cuda' or 'cpu')"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (height width, default: 224 224)"
    )
    parser.add_argument(
        "--patch-tokens",
        action="store_true",
        help="Also extract patch tokens (spatial features)"
    )
    parser.add_argument(
        "--visualize",
        type=str,
        default=None,
        help="Path to save feature visualization (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    # Create extractor
    extractor = DINOv3FeatureExtractor(
        model_name=args.model,
        device=args.device,
        image_size=tuple(args.size)
    )
    
    # Extract features
    print("\n" + "=" * 80)
    print("Extracting Features")
    print("=" * 80)
    print(f"Image: {image_path}")
    
    features = extractor.extract_features(
        str(image_path),
        return_patch_tokens=args.patch_tokens
    )
    
    # Visualize
    extractor.visualize_features(features, output_path=args.visualize)
    
    # Save features
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as numpy file
        np.save(str(output_path), features)
        print(f"\n✓ Features saved to: {output_path}")
        
        # Also save as text file for inspection
        txt_path = output_path.with_suffix('.txt')
        with open(txt_path, 'w') as f:
            f.write(f"DINOv3 Features from {image_path}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Feature dimension: {features['feature_dim']}\n")
            f.write(f"\nCLS Token Features:\n")
            f.write(f"Shape: {features['cls_token'].shape}\n")
            f.write(f"Values:\n")
            np.savetxt(f, features['cls_token'][0], fmt='%.6f')
            
            if 'patch_tokens' in features:
                f.write(f"\nPatch Tokens:\n")
                f.write(f"Shape: {features['patch_tokens'].shape}\n")
        
        print(f"✓ Feature text saved to: {txt_path}")
    else:
        print("\n💡 Tip: Use --output to save features to file")


if __name__ == "__main__":
    main()
