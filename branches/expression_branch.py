"""
Expression Branch with DINOv3 + SOTA 2025 Architecture
=======================================================

Adapts the SOTA 2025 micro-expression model for multimodal integration.

Architecture:
1. DINOv3 encoder (frozen, Meta AI 2024)
2. Optical flow for motion features
3. Hierarchical space-time encoder
4. Robust apex detection
5. Joint spotting + recognition
6. Output: 256-dim expression embedding

Author: Multimodal Deception Detection System
Date: 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

try:
    from config.model_config import ExpressionBranchConfig
except ImportError:
    # Fallback to simple config
    class ExpressionBranchConfig:
        dinov3_model = "dinov3_vits16"
        dinov3_dim = 384
        embedding_dim = 256
        dropout = 0.5
        device = "cuda" if torch.cuda.is_available() else "cpu"


class OpticalFlowExtractor(nn.Module):
    """
    Memory-efficient optical flow feature extractor.
    Inspired by OFVIG-Net (2025).
    """
    
    def __init__(self, in_channels: int = 3, output_dim: int = 128):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, output_dim)
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extract optical flow features from frame differences.
        
        Args:
            frames: (B, T, C, H, W)
        
        Returns:
            flow_features: (B, T, 128)
        """
        B, T, C, H, W = frames.shape
        
        if T < 2:
            return torch.zeros(B, T, 128, device=frames.device)
        
        # Compute frame differences
        flow_features_list = []
        
        # Process in chunks to save memory
        chunk_size = 8
        for chunk_start in range(0, T - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T - 1)
            chunk_flow_features = []
            
            for t in range(chunk_start, chunk_end):
                frame_diff = frames[:, t+1] - frames[:, t]  # (B, C, H, W)
                flow_feat = self.flow_net(frame_diff)  # (B, 128)
                chunk_flow_features.append(flow_feat)
            
            if chunk_flow_features:
                chunk_flow = torch.stack(chunk_flow_features, dim=1)  # (B, chunk_size, 128)
                flow_features_list.append(chunk_flow)
        
        # Concatenate chunks
        if flow_features_list:
            flow_features = torch.cat(flow_features_list, dim=1)  # (B, T-1, 128)
        else:
            flow_features = torch.zeros(B, T-1, 128, device=frames.device)
        
        # Pad to match T
        if flow_features.size(1) < T:
            last_flow = flow_features[:, -1:, :]
            flow_features = torch.cat([flow_features, last_flow], dim=1)
        
        return flow_features


class HierarchicalSpaceTimeEncoder(nn.Module):
    """
    Hierarchical Space-Time Encoder (2024 arXiv).
    Models space-time relations at multiple scales.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.3):
        super().__init__()
        
        # Spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Short-term temporal (1-3 frames)
        self.short_temporal = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=3, padding=1, groups=d_model // 4),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Long-term temporal (5-15 frames)
        self.long_temporal = nn.Sequential(
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=7, padding=3, groups=d_model // 4),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion
        self.fusion = nn.Linear(d_model // 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) frame features
        
        Returns:
            hierarchical_features: (B, T, D)
        """
        # Spatial
        spatial = self.spatial_encoder(x)  # (B, T, D/2)
        
        # Short-term temporal
        x_short = spatial.transpose(1, 2)  # (B, D/2, T)
        short_term = self.short_temporal(x_short).transpose(1, 2)  # (B, T, D/2)
        
        # Long-term temporal
        x_long = spatial.transpose(1, 2)
        long_term = self.long_temporal(x_long).transpose(1, 2)  # (B, T, D/2)
        
        # Combine
        combined = (short_term + long_term) / 2
        hierarchical = self.fusion(combined)  # (B, T, D)
        
        return hierarchical


class RobustApexDetector(nn.Module):
    """
    Robust Apex Frame Detector (2025 arXiv).
    Handles keyframe errors and noise.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # Multiple apex scoring heads (ensemble)
        self.apex_scorers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.GELU(),
                nn.Linear(d_model // 4, 1)
            )
            for _ in range(3)
        ])
        
        # Temporal smoothing
        self.temporal_smooth = nn.Conv1d(1, 1, kernel_size=5, padding=2, groups=1)
        
        # Confidence estimation
        self.confidence = nn.Linear(d_model, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D) frame features
        
        Returns:
            apex_features: (B, D)
            apex_weights: (B, T)
            apex_confidence: (B, T)
        """
        B, T, D = x.shape
        
        # Ensemble apex scores
        apex_scores_list = []
        for scorer in self.apex_scorers:
            scores = scorer(x).squeeze(-1)  # (B, T)
            apex_scores_list.append(scores)
        
        apex_scores = torch.stack(apex_scores_list, dim=0).mean(dim=0)  # (B, T)
        
        # Temporal smoothing
        apex_scores_smooth = self.temporal_smooth(apex_scores.unsqueeze(1)).squeeze(1)  # (B, T)
        
        # Confidence
        confidence = torch.sigmoid(self.confidence(x)).squeeze(-1)  # (B, T)
        
        # Weighted combination
        apex_scores_weighted = apex_scores_smooth * confidence
        apex_weights = torch.softmax(apex_scores_weighted, dim=1)  # (B, T)
        
        # Extract apex features
        apex_features = torch.sum(x * apex_weights.unsqueeze(-1), dim=1)  # (B, D)
        
        return apex_features, apex_weights, confidence


class JointSpottingRecognition(nn.Module):
    """
    Joint Spotting and Recognition (2024 arXiv).
    """
    
    def __init__(self, d_model: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        
        # Spotting branch
        self.spotting = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        # Recognition branch
        self.recognition = nn.Sequential(
            nn.Linear(d_model, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Joint fusion
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + 1, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, T, D)
        
        Returns:
            joint_features: (B, T, embedding_dim)
            spotting_scores: (B, T)
        """
        # Spotting
        spotting_scores = self.spotting(features).squeeze(-1)  # (B, T)
        
        # Recognition
        recognition_features = self.recognition(features)  # (B, T, embedding_dim)
        
        # Joint fusion
        spotting_expanded = spotting_scores.unsqueeze(-1)  # (B, T, 1)
        joint_input = torch.cat([recognition_features, spotting_expanded], dim=-1)
        joint_features = self.fusion(joint_input)  # (B, T, embedding_dim)
        
        return joint_features, spotting_scores


class ExpressionBranch(nn.Module):
    """
    Expression Branch with SOTA 2025 Architecture
    
    Combines DINOv3 encoder with state-of-the-art micro-expression detection
    techniques for multimodal deception detection.
    
    Architecture:
    1. DINOv3 encoder (frozen)
    2. Optical flow extractor
    3. Hierarchical space-time encoder
    4. Robust apex detector
    5. Joint spotting + recognition
    6. Feature fusion + temporal aggregation
    7. Output projection (256-dim embedding)
    
    Parameters: ~400K trainable (encoder frozen)
    
    Input: Video frames (B, T, C, H, W)
    Output: 256-dim expression embedding + auxiliary outputs
    """
    
    def __init__(
        self,
        dinov3_model_path: Optional[str] = None,
        config: Optional[ExpressionBranchConfig] = None,
        freeze_dinov3: bool = True,
        embedding_dim: int = 256
    ):
        super().__init__()
        
        if config is None:
            config = ExpressionBranchConfig()
        
        self.config = config
        self.embedding_dim = embedding_dim
        
        # Import DINOv3 encoder
        try:
            import sys
            sys.path.insert(0, '/workspace/dinov3_weights')
            
            from encoder import DinoV3Encoder
            
            # The encoder from micro_expression_branch doesn't have weights_path parameter
            # It auto-detects weights in its directory
            # So we don't pass weights_path, it will find it automatically
            
            self.encoder = DinoV3Encoder(
                model_name=config.dinov3_model,
                freeze=freeze_dinov3,
                use_patch_tokens=False,
                device=config.device
            )
            d_model = config.dinov3_dim
            print(f"✓ Loaded DINOv3 encoder: {config.dinov3_model}")
            
        except Exception as e:
            print(f"⚠️  Could not load DINOv3 encoder: {e}")
            print(f"    Using placeholder encoder instead")
            # Placeholder encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, config.dinov3_dim)
            )
            d_model = config.dinov3_dim
        
        # Component flags (for ablation studies)
        self.use_optical_flow = getattr(config, 'use_optical_flow', True)
        self.use_hierarchical_encoder = getattr(config, 'use_hierarchical_encoder', True)
        self.use_apex_detector = getattr(config, 'use_apex_detector', True)
        self.use_joint_spotting = getattr(config, 'use_joint_spotting', True)
        
        # Optical flow extractor
        if self.use_optical_flow:
            self.optical_flow = OpticalFlowExtractor(in_channels=3)
            self.flow_proj = nn.Linear(128, d_model // 4)
        else:
            self.optical_flow = None
            self.flow_proj = None
        
        # Hierarchical space-time encoder
        if self.use_hierarchical_encoder:
            self.hierarchical_encoder = HierarchicalSpaceTimeEncoder(d_model, config.dropout)
        else:
            self.hierarchical_encoder = None
        
        # Robust apex detector
        if self.use_apex_detector:
            self.apex_detector = RobustApexDetector(d_model)
        else:
            self.apex_detector = None
        
        # Joint spotting + recognition
        if self.use_joint_spotting:
            self.joint_model = JointSpottingRecognition(d_model, embedding_dim, config.dropout)
        else:
            self.joint_model = None
        
        # Feature fusion - calculate dimension based on enabled components
        fusion_dim = 0
        if self.use_hierarchical_encoder:
            fusion_dim += d_model
        if self.use_optical_flow:
            fusion_dim += d_model // 4
        if self.use_apex_detector:
            fusion_dim += d_model
        # If no components enabled, use base features
        if fusion_dim == 0:
            fusion_dim = d_model
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(config.dropout)
        )
        
        # Temporal aggregation (attention pooling)
        self.temporal_pool = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(embedding_dim // 4, 1)
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5)
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        return_attention: bool = False,
        return_all_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            frames: (B, T, C, H, W) video frames
            return_attention: Return attention weights
            return_all_outputs: Return all intermediate outputs
        
        Returns:
            Dict with:
                - embedding: (B, 256) expression embedding
                - apex_weights: (B, T) apex attention
                - spotting_scores: (B, T) spotting scores
                - temporal_attention: (B, T) if return_attention
        """
        B, T, C, H, W = frames.shape
        
        # 1. Extract appearance features with DINOv3
        frames_flat = frames.view(B * T, C, H, W)
        
        try:
            # Try to use proper DINOv3 encoder
            cls_tokens, _ = self.encoder(frames_flat, return_patch_tokens=False)
        except:
            # Fallback for placeholder encoder
            cls_tokens = self.encoder(frames_flat)
        
        frame_features = cls_tokens.view(B, T, -1)  # (B, T, D)
        
        # 2. Extract optical flow features
        if self.use_optical_flow:
            flow_features = self.optical_flow(frames)  # (B, T, 128)
            flow_proj = self.flow_proj(flow_features)  # (B, T, D/4)
        else:
            flow_proj = None
        
        # 3. Hierarchical space-time encoding
        if self.use_hierarchical_encoder:
            hierarchical_features = self.hierarchical_encoder(frame_features)  # (B, T, D)
        else:
            hierarchical_features = frame_features  # Use base features
        
        # 4. Robust apex detection
        if self.use_apex_detector:
            apex_features, apex_weights, apex_confidence = self.apex_detector(frame_features)
            apex_expanded = apex_features.unsqueeze(1).expand(-1, T, -1)  # (B, T, D)
        else:
            apex_expanded = None
            apex_weights = torch.ones(B, T, device=frames.device) / T  # Uniform weights
            apex_confidence = torch.ones(B, T, device=frames.device)
        
        # 5. Joint spotting + recognition
        if self.use_joint_spotting:
            joint_features, spotting_scores = self.joint_model(hierarchical_features)
        else:
            joint_features = None
            spotting_scores = torch.zeros(B, T, device=frames.device)
        
        # 6. Feature fusion
        fusion_parts = []
        if self.use_hierarchical_encoder:
            fusion_parts.append(hierarchical_features)
        if self.use_optical_flow and flow_proj is not None:
            fusion_parts.append(flow_proj)
        if self.use_apex_detector and apex_expanded is not None:
            fusion_parts.append(apex_expanded)
        
        # If no components enabled, use base features
        if len(fusion_parts) == 0:
            fusion_parts.append(frame_features)
        
        fused = torch.cat(fusion_parts, dim=-1)  # (B, T, fusion_dim)
        fused_features = self.feature_fusion(fused)  # (B, T, embedding_dim)
        
        # Combine with joint features
        if self.use_joint_spotting and joint_features is not None:
            combined_features = (fused_features + joint_features) / 2  # (B, T, embedding_dim)
        else:
            combined_features = fused_features
        
        # 7. Temporal aggregation
        attn_scores = self.temporal_pool(combined_features).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
        pooled_features = torch.sum(combined_features * attn_weights.unsqueeze(-1), dim=1)  # (B, embedding_dim)
        
        # 8. Output projection
        embedding = self.output_proj(pooled_features)  # (B, embedding_dim)
        
        outputs = {
            'embedding': embedding,
            'apex_weights': apex_weights,
            'apex_confidence': apex_confidence,
            'spotting_scores': spotting_scores
        }
        
        if return_attention:
            outputs['temporal_attention'] = attn_weights
        
        if return_all_outputs:
            outputs['frame_features'] = frame_features
            outputs['flow_features'] = flow_features
            outputs['hierarchical_features'] = hierarchical_features
            outputs['apex_features'] = apex_features
        
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'Model': 'SOTA 2025 Expression Branch',
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Frozen Parameters': total_params - trainable_params,
            'Encoder': self.config.dinov3_model,
            'Embedding Dim': self.embedding_dim,
            'Innovations': 'DINOv3 + Optical Flow + Hierarchical Space-Time + Robust Apex + Joint Spotting'
        }


if __name__ == "__main__":
    # Test the expression branch
    print("=" * 80)
    print("Testing Expression Branch")
    print("=" * 80)
    
    # Create model (without DINOv3 for testing)
    model = ExpressionBranch(
        dinov3_model_path=None,
        freeze_dinov3=True,
        embedding_dim=256
    )
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy frames
    batch_size = 2
    seq_length = 32
    frames = torch.randn(batch_size, seq_length, 3, 224, 224)
    
    print(f"\nInput shape: {frames.shape}")
    
    # Forward pass
    outputs = model(frames, return_attention=True, return_all_outputs=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n" + "=" * 80)
    print("✓ Expression Branch Test Passed!")
    print("=" * 80)

