"""
Configuration Module
===================

Centralized configuration for the multimodal deception detection system.

Author: Multimodal Deception Detection System
Date: 2025
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch


@dataclass
class AudioConfig:
    """Audio branch configuration."""
    
    # Input
    input_type: str = 'mfcc'  # 'waveform' or 'mfcc'
    sample_rate: int = 16000
    n_mfcc: int = 13
    
    # Model
    embedding_dim: int = 256
    dropout: float = 0.3
    use_attention_pooling: bool = True
    
    # Training
    freeze: bool = False


@dataclass
class ExpressionConfig:
    """Expression branch configuration."""
    
    # DINOv3
    dinov3_model: str = "dinov3_vits16"  # vits16, vitb16, vitl16
    dinov3_dim: int = 384  # MUST match model (vits16=384, vitb16=768, vitl16=1024)
    dinov3_weights_path: Optional[str] = "/workspace/dinov3_weights/dinov3_vits16_pretrain.pth"  # Default path in Docker
    freeze_dinov3: bool = True
    
    # Model
    embedding_dim: int = 256
    dropout: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Video processing
    sequence_length: int = 32
    fps: int = 15
    image_size: Tuple[int, int] = (224, 224)
    
    # Component flags (for ablation studies)
    use_optical_flow: bool = True
    use_hierarchical_encoder: bool = True
    use_apex_detector: bool = True
    use_joint_spotting: bool = True


@dataclass
class FusionConfig:
    """Fusion module configuration."""
    
    audio_dim: int = 256
    expression_dim: int = 256
    fusion_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.3
    use_cross_attention: bool = True
    use_uncertainty: bool = True


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    
    # Paths
    dolos_root: str = "data/dolos"
    trial_root: str = "data/trial"
    
    # Data split
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Processing
    use_preprocessed: bool = True  # Use pre-extracted features
    cache_features: bool = True
    
    # Audio
    audio_feature_dir: str = "audio"  # Subdirectory with MFCC .npy files
    audio_max_length: int = 700  # Max sequence length (pad/truncate)
    
    # Video
    video_dir: str = "video"
    frames_dir: str = "frames"
    max_frames: int = 32
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimization
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Learning rate schedule
    use_lr_scheduler: bool = True
    lr_scheduler_type: str = "cosine"  # "cosine", "step", "plateau"
    lr_warmup_epochs: int = 5
    lr_min: float = 1e-6
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    
    # Loss weights
    loss_weight_deception: float = 1.0
    loss_weight_apex: float = 0.1  # Auxiliary loss for expression branch
    loss_weight_spotting: float = 0.05
    
    # Regularization
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1
    
    # Class balancing
    use_class_weights: bool = True
    use_focal_loss: bool = True  # Use focal loss instead of BCE (better for imbalance)
    focal_alpha: float = 0.25  # Focal loss alpha (class balance)
    focal_gamma: float = 2.0   # Focal loss gamma (focus on hard examples)
    
    # Gradient
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    
    # System
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    save_checkpoint_every: int = 5
    
    # Logging
    log_dir: str = "logs"
    log_interval: int = 10
    use_wandb: bool = False
    wandb_project: str = "multimodal-deception-detection"
    wandb_entity: Optional[str] = None
    
    # Evaluation
    eval_interval: int = 1  # Evaluate every N epochs
    compute_class_metrics: bool = True  # Precision, recall, F1


@dataclass
class MultimodalConfig:
    """Complete multimodal system configuration."""
    
    audio: AudioConfig = field(default_factory=AudioConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Validate configuration."""
        # Check dimension consistency
        assert self.audio.embedding_dim == self.fusion.audio_dim, \
            f"Audio embedding dim ({self.audio.embedding_dim}) must match fusion audio dim ({self.fusion.audio_dim})"
        
        assert self.expression.embedding_dim == self.fusion.expression_dim, \
            f"Expression embedding dim ({self.expression.embedding_dim}) must match fusion expression dim ({self.fusion.expression_dim})"
        
        # Check DINOv3 dimensions
        dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024
        }
        expected_dim = dim_map.get(self.expression.dinov3_model)
        if expected_dim and self.expression.dinov3_dim != expected_dim:
            print(f"⚠️  WARNING: {self.expression.dinov3_model} should use dinov3_dim={expected_dim}, not {self.expression.dinov3_dim}")
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'audio': self.audio.__dict__,
            'expression': self.expression.__dict__,
            'fusion': self.fusion.__dict__,
            'dataset': self.dataset.__dict__,
            'training': self.training.__dict__
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MultimodalConfig':
        """Load configuration from JSON file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct config objects
        audio_config = AudioConfig(**config_dict['audio'])
        expression_config = ExpressionConfig(**config_dict['expression'])
        fusion_config = FusionConfig(**config_dict['fusion'])
        dataset_config = DatasetConfig(**config_dict['dataset'])
        training_config = TrainingConfig(**config_dict['training'])
        
        return cls(
            audio=audio_config,
            expression=expression_config,
            fusion=fusion_config,
            dataset=dataset_config,
            training=training_config
        )


# Preset configurations

def get_config_small_dataset() -> MultimodalConfig:
    """
    Configuration for small datasets (<500 samples).
    
    Optimized for:
    - High regularization (dropout=0.5)
    - Smaller model (DINOv3-ViT-S)
    - Conservative learning rate
    """
    import os
    config = MultimodalConfig()
    config.expression.dinov3_model = "dinov3_vits16"
    config.expression.dinov3_dim = 384
    config.expression.dropout = 0.5
    config.audio.dropout = 0.4
    config.fusion.dropout = 0.4
    config.training.batch_size = 4
    config.training.learning_rate = 5e-5
    config.training.use_mixup = False
    
    # Use Docker paths if available
    if os.path.exists('/workspace/data'):
        docker_dolos = "/workspace/data/cleaned_datasets/Dolos"
        docker_trial = "/workspace/data/cleaned_datasets/court"
        if os.path.exists(docker_dolos):
            config.dataset.dolos_root = docker_dolos
        if os.path.exists(docker_trial):
            config.dataset.trial_root = docker_trial
    
    return config


def get_config_medium_dataset() -> MultimodalConfig:
    """
    Configuration for medium datasets (500-2000 samples).
    
    Balanced regularization and model capacity.
    """
    import os
    config = MultimodalConfig()
    config.expression.dinov3_model = "dinov3_vits16"
    config.expression.dinov3_dim = 384
    config.expression.dropout = 0.3
    config.audio.dropout = 0.3
    config.fusion.dropout = 0.3
    config.training.batch_size = 8
    config.training.learning_rate = 1e-4
    
    # Use Docker paths if available
    if os.path.exists('/workspace/data'):
        docker_dolos = "/workspace/data/cleaned_datasets/Dolos"
        docker_trial = "/workspace/data/cleaned_datasets/court"
        if os.path.exists(docker_dolos):
            config.dataset.dolos_root = docker_dolos
        if os.path.exists(docker_trial):
            config.dataset.trial_root = docker_trial
    
    return config


def get_config_large_dataset() -> MultimodalConfig:
    """
    Configuration for large datasets (>2000 samples).
    
    Can use larger model and lower regularization.
    """
    import os
    config = MultimodalConfig()
    config.expression.dinov3_model = "dinov3_vitb16"
    config.expression.dinov3_dim = 768
    config.expression.dropout = 0.1
    config.audio.dropout = 0.2
    config.fusion.dropout = 0.2
    config.training.batch_size = 16
    config.training.learning_rate = 3e-4
    config.training.use_mixup = True
    
    # Use Docker paths if available
    if os.path.exists('/workspace/data'):
        docker_dolos = "/workspace/data/cleaned_datasets/Dolos"
        docker_trial = "/workspace/data/cleaned_datasets/court"
        if os.path.exists(docker_dolos):
            config.dataset.dolos_root = docker_dolos
        if os.path.exists(docker_trial):
            config.dataset.trial_root = docker_trial
    
    return config


if __name__ == "__main__":
    # Test configuration
    print("=" * 80)
    print("Testing Multimodal Configuration")
    print("=" * 80)
    
    # Create default config
    config = MultimodalConfig()
    
    print("\nDefault Configuration:")
    print(f"  Audio embedding dim: {config.audio.embedding_dim}")
    print(f"  Expression embedding dim: {config.expression.embedding_dim}")
    print(f"  DINOv3 model: {config.expression.dinov3_model}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    
    # Test preset configs
    print("\n" + "-" * 80)
    print("Preset Configurations:")
    
    presets = [
        ("Small Dataset", get_config_small_dataset()),
        ("Medium Dataset", get_config_medium_dataset()),
        ("Large Dataset", get_config_large_dataset())
    ]
    
    for name, preset_config in presets:
        print(f"\n{name}:")
        print(f"  DINOv3: {preset_config.expression.dinov3_model}")
        print(f"  Dropout: {preset_config.expression.dropout}")
        print(f"  Batch size: {preset_config.training.batch_size}")
        print(f"  Learning rate: {preset_config.training.learning_rate}")
    
    # Test save/load
    print("\n" + "-" * 80)
    print("Testing Save/Load:")
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        config.save(config_path)
        loaded_config = MultimodalConfig.load(config_path)
        print(f"✓ Config saved and loaded successfully")
        print(f"  Batch size (loaded): {loaded_config.training.batch_size}")
    
    print("\n" + "=" * 80)
    print("✓ Configuration Test Passed!")
    print("=" * 80)

