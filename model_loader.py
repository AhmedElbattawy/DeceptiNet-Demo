#!/usr/bin/env python3
"""
Model Loader for Camera Demo
=============================

Loads the trained multimodal deception detection model from checkpoint.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config.model_config import MultimodalConfig, get_config_medium_dataset
from branches.audio_branch import LiquidAudioBranch
from branches.expression_branch import ExpressionBranch
from fusion.multimodal_fusion import MultimodalDeceptionDetector


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = None,
    config: Optional[MultimodalConfig] = None,
    dinov3_weights_path: Optional[str] = None
) -> MultimodalDeceptionDetector:
    """
    Load the multimodal deception detection model from checkpoint.

    Args:
        checkpoint_path:     Path to the .pt checkpoint file.
        device:              'cuda' or 'cpu' (auto-detected if None).
        config:              MultimodalConfig (medium-dataset defaults if None).
        dinov3_weights_path: Explicit path to dinov3_vits16_pretrain.pth.
                             Falls back to DINOV3_WEIGHTS env var, then
                             checkpoints/dinov3_vits16_pretrain.pth.

    Returns:
        Loaded model in eval mode.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if config is None:
        # Use medium dataset config (matches training)
        config = get_config_medium_dataset()
    
    # Resolve DINOv3 weights path
    # Priority: explicit argument > DINOV3_WEIGHTS env var > ./checkpoints/dinov3_vits16_pretrain.pth
    import os
    repo_root = Path(__file__).parent

    dinov3_candidates = [
        dinov3_weights_path,                                          # explicit argument
        os.environ.get("DINOV3_WEIGHTS"),                             # environment variable
        str(repo_root / "checkpoints" / "dinov3_vits16_pretrain.pth"),# local checkpoints/
        config.expression.dinov3_weights_path,                        # config default
    ]

    dinov3_path = None
    for candidate in dinov3_candidates:
        if candidate and Path(candidate).exists():
            dinov3_path = candidate
            print(f"✓ Found DINOv3 weights: {dinov3_path}")
            break

    if dinov3_path is None:
        print("⚠️  DINOv3 weights not found. Expression branch will use random init.")
        print("   Download from: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth")
        print("   Place in: checkpoints/dinov3_vits16_pretrain.pth")
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    # Build model architecture
    print("Building model architecture...")
    
    # Audio branch
    audio_branch = LiquidAudioBranch(
        input_type=config.audio.input_type,
        n_mfcc=config.audio.n_mfcc,
        embedding_dim=config.audio.embedding_dim,
        dropout=config.audio.dropout,
        use_attention_pooling=config.audio.use_attention_pooling
    )
    
    # Expression branch (override DINOv3 path with detected path)
    expression_branch = ExpressionBranch(
        dinov3_model_path=dinov3_path,
        config=config.expression,
        freeze_dinov3=config.expression.freeze_dinov3,
        embedding_dim=config.expression.embedding_dim
    )
    
    # Complete multimodal model
    model = MultimodalDeceptionDetector(
        audio_branch=audio_branch,
        expression_branch=expression_branch,
        fusion_config=config.fusion.__dict__
    )
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        # Checkpoint might be just the state dict
        state_dict = checkpoint
    
    # Load state dict with strict=False to handle encoder structure differences
    # (checkpoint may have real DINOv3 structure, but we might be using placeholder)
    try:
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys (will use random init): {len(missing_keys)} keys")
            # Filter out encoder keys if using placeholder
            encoder_missing = [k for k in missing_keys if 'encoder' in k]
            if encoder_missing:
                print(f"   ({len(encoder_missing)} encoder keys - expected if using placeholder)")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys (ignored): {len(unexpected_keys)} keys")
            # Filter out encoder keys
            encoder_unexpected = [k for k in unexpected_keys if 'encoder' in k]
            if encoder_unexpected:
                print(f"   ({len(encoder_unexpected)} encoder keys - expected if encoder structure differs)")
        print(f"✓ Model state dict loaded (non-strict mode)")
    except Exception as e:
        print(f"⚠️  Error loading state dict: {e}")
        print(f"   Attempting to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Model state dict loaded (with warnings)")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model ready for inference on {device}")
    
    return model


if __name__ == "__main__":
    # Test loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model loading")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    print("\n✓ Model loading test successful!")
