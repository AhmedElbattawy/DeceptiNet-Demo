"""
Audio Branch with Liquid Neural Networks (PyTorch)
===================================================

Implements the audio processing branch using Liquid NN for adaptive
temporal modeling of speech patterns for deception detection.

Features:
- Liquid NN layers for robust temporal processing
- MFCC feature extraction
- Adaptive recurrent modeling
- Output: 256-dim audio embedding

Author: Multimodal Deception Detection System
Date: 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

# Try to import torchaudio, fall back to librosa
try:
    import torchaudio
    import torchaudio.transforms as T
    USE_TORCHAUDIO = True
except (ImportError, OSError):
    USE_TORCHAUDIO = False
    try:
        import librosa
        USE_LIBROSA = True
    except ImportError:
        USE_LIBROSA = False
        print("⚠️  Warning: Neither torchaudio nor librosa available. AudioPreprocessor will not work.")


class LiquidLayer(nn.Module):
    """
    Liquid Neural Network Layer (PyTorch Implementation)
    
    Provides adaptive temporal processing with dynamic weights
    that adjust based on input signal strength.
    
    Key Features:
    - Adaptive weight modulation
    - Noise resilience
    - Temporal consistency
    """
    
    def __init__(
        self,
        input_dim: int,
        units: int,
        activation: str = 'tanh',
        dropout: float = 0.2
    ):
        super().__init__()
        self.units = units
        self.input_dim = input_dim
        
        # Base weights (static component)
        self.W = nn.Parameter(torch.randn(input_dim, units) * 0.02)
        
        # Liquid component (adaptive)
        self.U = nn.Parameter(torch.randn(input_dim, units) * 0.02)
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(units))
        
        # Adaptive scaling factor
        self.scale = nn.Parameter(torch.ones(1))
        
        # Activation
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = lambda x: x
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive weight modulation.
        
        Args:
            x: (B, T, input_dim) input sequence
        
        Returns:
            output: (B, T, units) processed sequence
        """
        # Calculate input strength for adaptive behavior
        # input_strength: (B, T)
        input_strength = torch.mean(torch.abs(x), dim=-1, keepdim=True)  # (B, T, 1)
        
        # Adaptive scaling
        adaptive_scale = torch.tanh(self.scale * input_strength)  # (B, T, 1)
        
        # Base transformation
        output = torch.matmul(x, self.W)  # (B, T, units)
        
        # Adaptive component
        adaptive_component = torch.matmul(x, self.U) * adaptive_scale
        
        # Combine
        output = output + adaptive_component + self.bias
        
        # Apply activation and dropout
        output = self.activation(output)
        output = self.dropout(output)
        
        return output


class AudioPreprocessor(nn.Module):
    """
    Audio preprocessing: waveform -> MFCC features
    
    Input: Raw audio waveform (16kHz)
    Output: MFCC features (13 coefficients)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_mels: int = 40,
        hop_length: int = 512
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        
        if USE_TORCHAUDIO:
            # MelSpectrogram
            self.mel_spectrogram = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=1024,
                hop_length=hop_length,
                n_mels=n_mels,
                f_min=0.0,
                f_max=8000.0
            )
            
            # MFCC transform
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    'n_fft': 1024,
                    'hop_length': hop_length,
                    'n_mels': n_mels,
                    'f_min': 0.0,
                    'f_max': 8000.0
                }
            )
            self.use_torchaudio = True
        elif USE_LIBROSA:
            self.use_torchaudio = False
            print("✓ AudioPreprocessor using librosa (torchaudio not available)")
        else:
            raise RuntimeError("Neither torchaudio nor librosa available. Cannot create AudioPreprocessor.")
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.
        
        Args:
            waveform: (B, samples) audio waveform
        
        Returns:
            mfcc: (B, T, n_mfcc) MFCC features
        """
        if self.use_torchaudio:
            # Compute MFCC using torchaudio
            mfcc = self.mfcc_transform(waveform)  # (B, n_mfcc, T)
            # Transpose to (B, T, n_mfcc)
            mfcc = mfcc.transpose(1, 2)
        else:
            # Use librosa for MFCC extraction
            import librosa
            batch_size = waveform.shape[0]
            mfcc_list = []
            
            for i in range(batch_size):
                # Convert to numpy
                audio_np = waveform[i].cpu().numpy().astype(np.float32)
                # Extract MFCC
                mfcc_batch = librosa.feature.mfcc(
                    y=audio_np,
                    sr=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    hop_length=self.hop_length,
                    n_fft=1024,
                    n_mels=40,
                    fmin=0.0,
                    fmax=8000.0
                )  # (n_mfcc, T)
                mfcc_batch = mfcc_batch.T  # (T, n_mfcc)
                mfcc_list.append(mfcc_batch)
            
            # Stack and convert to tensor
            mfcc = np.stack(mfcc_list, axis=0)  # (B, T, n_mfcc)
            mfcc = torch.from_numpy(mfcc.astype(np.float32)).to(waveform.device)
        
        # Normalize
        mfcc = (mfcc - mfcc.mean(dim=1, keepdim=True)) / (mfcc.std(dim=1, keepdim=True) + 1e-8)
        
        return mfcc


class LiquidAudioBranch(nn.Module):
    """
    Audio Branch with Liquid Neural Networks
    
    Architecture:
    1. Audio Preprocessing (MFCC extraction)
    2. Liquid Layer 1 (512 units) - Coarse temporal patterns
    3. Liquid Layer 2 (256 units) - Refined patterns
    4. Liquid Layer 3 (128 units) - Fine-grained features
    5. Temporal Pooling (Attention-based)
    6. Output Projection (256-dim embedding)
    
    Parameters: ~300K trainable
    
    Input: Raw audio waveform or MFCC features
    Output: 256-dim audio embedding
    """
    
    def __init__(
        self,
        input_type: str = 'waveform',  # 'waveform' or 'mfcc'
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        use_attention_pooling: bool = True
    ):
        super().__init__()
        
        self.input_type = input_type
        self.embedding_dim = embedding_dim
        self.use_attention_pooling = use_attention_pooling
        
        # Audio preprocessing (if needed)
        if input_type == 'waveform':
            self.preprocessor = AudioPreprocessor(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc
            )
            input_dim = n_mfcc
        else:
            self.preprocessor = None
            input_dim = n_mfcc  # Assume MFCC input
        
        # Liquid layers (hierarchical temporal processing)
        self.liquid1 = LiquidLayer(input_dim, 512, activation='tanh', dropout=dropout)
        self.liquid2 = LiquidLayer(512, 256, activation='tanh', dropout=dropout * 0.8)
        self.liquid3 = LiquidLayer(256, 128, activation='tanh', dropout=dropout * 0.7)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Temporal pooling
        if use_attention_pooling:
            # Attention-based pooling (learnable)
            self.attention_pool = nn.Sequential(
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        else:
            # Global average pooling
            self.attention_pool = None
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        audio: torch.Tensor,
        return_attention: bool = False,
        return_all_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            audio: (B, samples) waveform or (B, T, n_mfcc) MFCC features
            return_attention: Return attention weights if True
            return_all_features: Return intermediate features if True
        
        Returns:
            Dict with:
                - embedding: (B, 256) audio embedding
                - temporal_features: (B, T, 128) if return_all_features
                - attention_weights: (B, T) if return_attention
        """
        outputs = {}
        
        # 1. Preprocessing (if needed)
        if self.input_type == 'waveform' and self.preprocessor is not None:
            x = self.preprocessor(audio)  # (B, T, n_mfcc)
        else:
            x = audio
        
        B, T, C = x.shape
        
        # 2. Liquid layers (adaptive temporal processing)
        # Layer 1
        x = self.liquid1(x)  # (B, T, 512)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # BN over features
        
        # Layer 2
        x = self.liquid2(x)  # (B, T, 256)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        
        # Layer 3
        x = self.liquid3(x)  # (B, T, 128)
        x = self.bn3(x.transpose(1, 2)).transpose(1, 2)
        
        temporal_features = x  # Store for optional output
        
        # 3. Temporal pooling
        if self.use_attention_pooling and self.attention_pool is not None:
            # Attention-based pooling
            attn_scores = self.attention_pool(x).squeeze(-1)  # (B, T)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T)
            
            # Weighted sum
            pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # (B, 128)
            
            if return_attention:
                outputs['attention_weights'] = attn_weights
        else:
            # Global average pooling
            pooled = torch.mean(x, dim=1)  # (B, 128)
        
        # 4. Output projection
        embedding = self.output_proj(pooled)  # (B, 256)
        
        outputs['embedding'] = embedding
        
        if return_all_features:
            outputs['temporal_features'] = temporal_features
        
        return outputs
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'Model': 'Liquid Audio Branch',
            'Total Parameters': total_params,
            'Trainable Parameters': trainable_params,
            'Input Type': self.input_type,
            'Embedding Dim': self.embedding_dim,
            'Liquid Layers': 3,
            'Attention Pooling': self.use_attention_pooling
        }


# Utility function for loading pre-extracted MFCC features
def load_mfcc_features(filepath: str) -> torch.Tensor:
    """
    Load pre-extracted MFCC features from .npy file.
    
    Args:
        filepath: Path to .npy file with MFCC features
    
    Returns:
        mfcc: (T, n_mfcc) MFCC features as torch tensor
    """
    mfcc = np.load(filepath)
    
    # Handle different formats
    if mfcc.ndim == 1:
        # Flat array, reshape to (T, n_mfcc)
        n_mfcc = 13
        T = len(mfcc) // n_mfcc
        mfcc = mfcc[:T * n_mfcc].reshape(T, n_mfcc)
    elif mfcc.ndim == 2:
        # Already (T, n_mfcc) or (n_mfcc, T)
        if mfcc.shape[0] < mfcc.shape[1]:
            mfcc = mfcc.T  # Transpose to (T, n_mfcc)
    
    return torch.from_numpy(mfcc).float()


if __name__ == "__main__":
    # Test the audio branch
    print("=" * 80)
    print("Testing Liquid Audio Branch")
    print("=" * 80)
    
    # Create model
    model = LiquidAudioBranch(
        input_type='mfcc',
        n_mfcc=13,
        embedding_dim=256,
        use_attention_pooling=True
    )
    
    # Print model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy MFCC features
    batch_size = 4
    seq_length = 700  # ~7 seconds at 15ms hop
    n_mfcc = 13
    
    dummy_mfcc = torch.randn(batch_size, seq_length, n_mfcc)
    
    print(f"\nInput shape: {dummy_mfcc.shape}")
    
    # Forward pass
    outputs = model(dummy_mfcc, return_attention=True, return_all_features=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\n" + "=" * 80)
    print("✓ Audio Branch Test Passed!")
    print("=" * 80)

