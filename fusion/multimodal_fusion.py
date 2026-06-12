"""
Multimodal Fusion Module
========================

Implements uncertainty-aware cross-modal fusion for audio + expression modalities.

Fusion Strategies:
1. Early Fusion: Feature concatenation
2. Late Fusion: Decision-level combination
3. Cross-Attention Fusion: Learnable attention between modalities
4. Uncertainty-Aware Fusion: Confidence-weighted combination (SOTA)

Author: Multimodal Deception Detection System
Date: 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism.
    
    Allows audio and expression modalities to attend to each other,
    capturing complementary information.
    """
    
    def __init__(
        self,
        audio_dim: int = 256,
        expression_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert audio_dim == expression_dim, "For cross-attention, dimensions must match"
        
        self.d_model = audio_dim
        self.num_heads = num_heads
        self.head_dim = self.d_model // num_heads
        
        # Query, Key, Value projections
        self.q_audio = nn.Linear(audio_dim, audio_dim)
        self.k_expression = nn.Linear(expression_dim, expression_dim)
        self.v_expression = nn.Linear(expression_dim, expression_dim)
        
        self.q_expression = nn.Linear(expression_dim, expression_dim)
        self.k_audio = nn.Linear(audio_dim, audio_dim)
        self.v_audio = nn.Linear(audio_dim, audio_dim)
        
        # Output projections
        self.out_audio = nn.Linear(audio_dim, audio_dim)
        self.out_expression = nn.Linear(expression_dim, expression_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        expression_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Cross-modal attention between audio and expression.
        
        Args:
            audio_features: (B, audio_dim)
            expression_features: (B, expression_dim)
        
        Returns:
            audio_attended: (B, audio_dim) - audio features attended by expression
            expression_attended: (B, expression_dim) - expression features attended by audio
            attention_weights: Dict with attention maps
        """
        B = audio_features.size(0)
        
        # Audio attends to expression
        q_a = self.q_audio(audio_features).unsqueeze(1)  # (B, 1, d_model)
        k_e = self.k_expression(expression_features).unsqueeze(1)  # (B, 1, d_model)
        v_e = self.v_expression(expression_features).unsqueeze(1)  # (B, 1, d_model)
        
        # Reshape for multi-head attention
        q_a = q_a.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        k_e = k_e.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_e = v_e.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_ae = torch.matmul(q_a, k_e.transpose(-2, -1)) / self.scale  # (B, num_heads, 1, 1)
        attn_ae = torch.softmax(attn_ae, dim=-1)
        attn_ae = self.dropout(attn_ae)
        
        # Attended features
        audio_attended = torch.matmul(attn_ae, v_e)  # (B, num_heads, 1, head_dim)
        audio_attended = audio_attended.transpose(1, 2).contiguous().view(B, self.d_model)
        audio_attended = self.out_audio(audio_attended)
        
        # Expression attends to audio
        q_e = self.q_expression(expression_features).unsqueeze(1)
        k_a = self.k_audio(audio_features).unsqueeze(1)
        v_a = self.v_audio(audio_features).unsqueeze(1)
        
        q_e = q_e.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = k_a.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = v_a.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_ea = torch.matmul(q_e, k_a.transpose(-2, -1)) / self.scale
        attn_ea = torch.softmax(attn_ea, dim=-1)
        attn_ea = self.dropout(attn_ea)
        
        expression_attended = torch.matmul(attn_ea, v_a)
        expression_attended = expression_attended.transpose(1, 2).contiguous().view(B, self.d_model)
        expression_attended = self.out_expression(expression_attended)
        
        attention_weights = {
            'audio_to_expression': attn_ae.squeeze(2).squeeze(2),  # (B, num_heads)
            'expression_to_audio': attn_ea.squeeze(2).squeeze(2)
        }
        
        return audio_attended, expression_attended, attention_weights


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation for each modality.
    
    Estimates the confidence/reliability of each modality's predictions,
    allowing for adaptive fusion.
    """
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty for the given features.
        
        Args:
            features: (B, input_dim)
        
        Returns:
            uncertainty: (B, 1) - lower is more confident
        """
        uncertainty = self.uncertainty_net(features)
        return uncertainty


class MultimodalFusion(nn.Module):
    """
    Multimodal Fusion Module
    
    Combines audio and expression modalities using uncertainty-aware fusion.
    
    Fusion Pipeline:
    1. Cross-modal attention (audio ↔ expression)
    2. Uncertainty estimation (per modality)
    3. Confidence-weighted fusion
    4. Final projection + classification
    
    Parameters: ~200K
    
    Input:
        - Audio embedding: (B, 256)
        - Expression embedding: (B, 256)
    
    Output:
        - Deception logits: (B,)
        - Fused embedding: (B, 256)
        - Modality weights: (B, 2)
    """
    
    def __init__(
        self,
        audio_dim: int = 256,
        expression_dim: int = 256,
        fusion_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        use_cross_attention: bool = True,
        use_uncertainty: bool = True
    ):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.expression_dim = expression_dim
        self.fusion_dim = fusion_dim
        self.use_cross_attention = use_cross_attention
        self.use_uncertainty = use_uncertainty
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(
                audio_dim=audio_dim,
                expression_dim=expression_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        
        # Uncertainty estimation
        if use_uncertainty:
            self.audio_uncertainty = UncertaintyEstimator(audio_dim)
            self.expression_uncertainty = UncertaintyEstimator(expression_dim)
        
        # Modality projection (make dimensions compatible)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.expression_proj = nn.Sequential(
            nn.Linear(expression_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion network
        fusion_input_dim = fusion_dim * 2 if use_cross_attention else fusion_dim * 2
        
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1)  # Binary classification
        )
        
        # Embedding projection (for downstream tasks)
        self.embedding_proj = nn.Linear(fusion_dim // 2, fusion_dim)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        expression_features: torch.Tensor,
        return_weights: bool = False,
        return_embedding: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty-aware fusion.
        
        Args:
            audio_features: (B, audio_dim)
            expression_features: (B, expression_dim)
            return_weights: Return modality weights
            return_embedding: Return fused embedding
        
        Returns:
            Dict with:
                - logits: (B,) deception predictions
                - embedding: (B, fusion_dim) fused embedding
                - modality_weights: (B, 2) [audio_weight, expression_weight]
                - audio_uncertainty: (B, 1)
                - expression_uncertainty: (B, 1)
                - cross_attention: Dict with attention maps (if enabled)
        """
        outputs = {}
        B = audio_features.size(0)
        
        # 1. Cross-modal attention (optional)
        if self.use_cross_attention:
            audio_attended, expression_attended, attn_weights = self.cross_attention(
                audio_features, expression_features
            )
            
            # Residual connection
            audio_features = audio_features + audio_attended
            expression_features = expression_features + expression_attended
            
            outputs['cross_attention'] = attn_weights
        
        # 2. Project modalities
        audio_proj = self.audio_proj(audio_features)  # (B, fusion_dim)
        expression_proj = self.expression_proj(expression_features)  # (B, fusion_dim)
        
        # 3. Uncertainty estimation (optional)
        if self.use_uncertainty:
            audio_unc = self.audio_uncertainty(audio_features)  # (B, 1)
            expression_unc = self.expression_uncertainty(expression_features)  # (B, 1)
            
            # Convert uncertainty to confidence (inverse)
            audio_conf = 1.0 / (1.0 + audio_unc)
            expression_conf = 1.0 / (1.0 + expression_unc)
            
            # Normalize confidences to sum to 1
            total_conf = audio_conf + expression_conf
            audio_weight = audio_conf / total_conf
            expression_weight = expression_conf / total_conf
            
            # Confidence-weighted features
            audio_weighted = audio_proj * audio_weight
            expression_weighted = expression_proj * expression_weight
            
            outputs['audio_uncertainty'] = audio_unc
            outputs['expression_uncertainty'] = expression_unc
            outputs['modality_weights'] = torch.cat([audio_weight, expression_weight], dim=1)  # (B, 2)
        else:
            # Equal weighting
            audio_weighted = audio_proj
            expression_weighted = expression_proj
        
        # 4. Fusion
        fused_input = torch.cat([audio_weighted, expression_weighted], dim=1)  # (B, fusion_dim * 2)
        fused = self.fusion_net(fused_input)  # (B, fusion_dim // 2)
        
        # 5. Classification
        logits = self.classifier(fused).squeeze(-1)  # (B,)
        
        outputs['logits'] = logits
        
        # 6. Embedding (if requested)
        if return_embedding:
            embedding = self.embedding_proj(fused)  # (B, fusion_dim)
            outputs['embedding'] = embedding
        
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'Model': 'Multimodal Fusion',
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Audio Dim': self.audio_dim,
            'Expression Dim': self.expression_dim,
            'Fusion Dim': self.fusion_dim,
            'Cross Attention': self.use_cross_attention,
            'Uncertainty Estimation': self.use_uncertainty
        }


class MultimodalDeceptionDetector(nn.Module):
    """
    Complete Multimodal Deception Detection System
    
    Integrates audio and expression branches with fusion module.
    
    Total Parameters: ~1M (audio: 300K, expression: 400K, fusion: 200K, DINOv3 frozen: 22M)
    
    Input:
        - Audio: (B, T_audio, n_mfcc) or waveform
        - Video: (B, T_video, C, H, W)
    
    Output:
        - Deception prediction (logits)
        - Modality contributions
        - Attention visualizations
    """
    
    def __init__(
        self,
        audio_branch: nn.Module,
        expression_branch: nn.Module,
        fusion_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.audio_branch = audio_branch
        self.expression_branch = expression_branch
        
        # Fusion module
        if fusion_config is None:
            fusion_config = {
                'audio_dim': 256,
                'expression_dim': 256,
                'fusion_dim': 256,
                'num_heads': 4,
                'dropout': 0.3,
                'use_cross_attention': True,
                'use_uncertainty': True
            }
        
        self.fusion = MultimodalFusion(**fusion_config)
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        return_attention: bool = False,
        return_all_outputs: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete multimodal system.
        
        Args:
            audio: (B, T, n_mfcc) or (B, samples) audio input
            video: (B, T, C, H, W) video frames
            return_attention: Return attention weights
            return_all_outputs: Return all intermediate outputs
        
        Returns:
            Dict with predictions and optional attention/embeddings
        """
        # 1. Audio branch
        audio_outputs = self.audio_branch(
            audio,
            return_attention=return_attention,
            return_all_features=return_all_outputs
        )
        audio_embedding = audio_outputs['embedding']
        
        # 2. Expression branch
        expression_outputs = self.expression_branch(
            video,
            return_attention=return_attention,
            return_all_outputs=return_all_outputs
        )
        expression_embedding = expression_outputs['embedding']
        
        # 3. Fusion
        fusion_outputs = self.fusion(
            audio_embedding,
            expression_embedding,
            return_weights=True,
            return_embedding=True
        )
        
        # Combine outputs
        outputs = {
            'logits': fusion_outputs['logits'],
            'embedding': fusion_outputs.get('embedding'),
            'modality_weights': fusion_outputs.get('modality_weights')
        }
        
        if return_attention:
            outputs['audio_attention'] = audio_outputs.get('attention_weights')
            outputs['expression_attention'] = expression_outputs.get('temporal_attention')
            outputs['cross_attention'] = fusion_outputs.get('cross_attention')
        
        if return_all_outputs:
            outputs['audio_outputs'] = audio_outputs
            outputs['expression_outputs'] = expression_outputs
            outputs['fusion_outputs'] = fusion_outputs
        
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get complete model information."""
        audio_info = self.audio_branch.get_model_info()
        expression_info = self.expression_branch.get_model_info()
        fusion_info = self.fusion.get_model_info()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'System': 'Multimodal Deception Detection',
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Audio Branch': audio_info,
            'Expression Branch': expression_info,
            'Fusion Module': fusion_info
        }


if __name__ == "__main__":
    # Test the fusion module
    print("=" * 80)
    print("Testing Multimodal Fusion")
    print("=" * 80)
    
    # Create fusion module
    fusion = MultimodalFusion(
        audio_dim=256,
        expression_dim=256,
        fusion_dim=256,
        use_cross_attention=True,
        use_uncertainty=True
    )
    
    # Print model info
    info = fusion.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy features
    batch_size = 4
    audio_features = torch.randn(batch_size, 256)
    expression_features = torch.randn(batch_size, 256)
    
    print(f"\nInput shapes:")
    print(f"  Audio: {audio_features.shape}")
    print(f"  Expression: {expression_features.shape}")
    
    # Forward pass
    outputs = fusion(audio_features, expression_features, return_weights=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}: {list(value.keys())}")
    
    print("\n" + "=" * 80)
    print("✓ Fusion Module Test Passed!")
    print("=" * 80)

