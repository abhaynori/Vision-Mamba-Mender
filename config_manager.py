"""
Configuration Management for NeurIPS Framework

This script helps manage different experimental configurations and provides
easy presets for various research scenarios.
"""

import json
import os
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import argparse

from core.unified_neurips_framework import UnifiedFrameworkConfig


@dataclass
class ExperimentConfig:
    """Extended configuration for experiments"""
    
    # Base configuration
    framework_config: UnifiedFrameworkConfig
    
    # Experiment settings
    experiment_name: str = "default_experiment"
    description: str = "Default NeurIPS framework experiment"
    dataset_type: str = "synthetic"  # synthetic, cifar10, imagenet, custom
    num_epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    
    # Evaluation settings
    eval_frequency: int = 1  # Evaluate every N epochs
    save_frequency: int = 5  # Save checkpoint every N epochs
    
    # Logging
    log_frequency: int = 10  # Log every N batches
    save_plots: bool = True
    save_checkpoints: bool = True
    
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    num_workers: int = 4


class ConfigurationManager:
    """
    Manages different experimental configurations
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def create_preset_configs(self) -> Dict[str, ExperimentConfig]:
        """
        Create preset configurations for different research scenarios
        """
        
        presets = {}
        
        # 1. Baseline: Only base model (no novel components)
        baseline_config = UnifiedFrameworkConfig()
        baseline_config.enable_msnar = False
        baseline_config.enable_quantum = False
        baseline_config.enable_hyperbolic = False
        baseline_config.enable_meta_learning = False
        baseline_config.enable_adversarial = False
        
        presets['baseline'] = ExperimentConfig(
            framework_config=baseline_config,
            experiment_name="baseline_only",
            description="Baseline model without any novel components",
            num_epochs=20,
            learning_rate=0.001
        )
        
        # 2. MSNAR Only: Test neuroplasticity component
        msnar_config = UnifiedFrameworkConfig()
        msnar_config.enable_msnar = True
        msnar_config.enable_quantum = False
        msnar_config.enable_hyperbolic = False
        msnar_config.enable_meta_learning = False
        msnar_config.enable_adversarial = False
        
        presets['msnar_only'] = ExperimentConfig(
            framework_config=msnar_config,
            experiment_name="msnar_neuroplasticity",
            description="Testing Multi-Scale Neuroplasticity Adaptive Repair (MSNAR)",
            num_epochs=25,
            learning_rate=0.0005
        )
        
        # 3. Quantum Only: Test quantum optimization
        quantum_config = UnifiedFrameworkConfig()
        quantum_config.enable_msnar = False
        quantum_config.enable_quantum = True
        quantum_config.enable_hyperbolic = False
        quantum_config.enable_meta_learning = False
        quantum_config.enable_adversarial = False
        
        presets['quantum_only'] = ExperimentConfig(
            framework_config=quantum_config,
            experiment_name="quantum_optimization",
            description="Testing Quantum-Inspired State Optimization",
            num_epochs=30,
            learning_rate=0.0008
        )
        
        # 4. Hyperbolic Only: Test geometric manifolds
        hyperbolic_config = UnifiedFrameworkConfig()
        hyperbolic_config.enable_msnar = False
        hyperbolic_config.enable_quantum = False
        hyperbolic_config.enable_hyperbolic = True
        hyperbolic_config.enable_meta_learning = False
        hyperbolic_config.enable_adversarial = False
        
        presets['hyperbolic_only'] = ExperimentConfig(
            framework_config=hyperbolic_config,
            experiment_name="hyperbolic_geometry",
            description="Testing Hyperbolic Geometric Manifolds",
            num_epochs=25,
            learning_rate=0.0006
        )
        
        # 5. Pairwise Combinations
        msnar_quantum_config = UnifiedFrameworkConfig()
        msnar_quantum_config.enable_msnar = True
        msnar_quantum_config.enable_quantum = True
        msnar_quantum_config.enable_hyperbolic = False
        msnar_quantum_config.enable_meta_learning = False
        msnar_quantum_config.enable_adversarial = False
        
        presets['msnar_quantum'] = ExperimentConfig(
            framework_config=msnar_quantum_config,
            experiment_name="msnar_quantum_integration",
            description="Integration of MSNAR and Quantum Optimization",
            num_epochs=35,
            learning_rate=0.0007
        )
        
        # 6. Meta-Learning Focus
        meta_config = UnifiedFrameworkConfig()
        meta_config.enable_msnar = True
        meta_config.enable_quantum = False
        meta_config.enable_hyperbolic = False
        meta_config.enable_meta_learning = True
        meta_config.enable_adversarial = False
        
        presets['meta_learning'] = ExperimentConfig(
            framework_config=meta_config,
            experiment_name="meta_learning_adaptation",
            description="Testing Meta-Learning State Evolution with MSNAR",
            num_epochs=40,
            learning_rate=0.0005
        )
        
        # 7. Adversarial Robustness
        adversarial_config = UnifiedFrameworkConfig()
        adversarial_config.enable_msnar = True
        adversarial_config.enable_quantum = False
        adversarial_config.enable_hyperbolic = False
        adversarial_config.enable_meta_learning = False
        adversarial_config.enable_adversarial = True
        
        presets['adversarial_robust'] = ExperimentConfig(
            framework_config=adversarial_config,
            experiment_name="adversarial_robustness",
            description="Testing Adversarial Robustness Generation with MSNAR",
            num_epochs=30,
            learning_rate=0.0006
        )
        
        # 8. Triple Combination: Core Research Components
        core_triple_config = UnifiedFrameworkConfig()
        core_triple_config.enable_msnar = True
        core_triple_config.enable_quantum = True
        core_triple_config.enable_hyperbolic = True
        core_triple_config.enable_meta_learning = False
        core_triple_config.enable_adversarial = False
        
        presets['core_triple'] = ExperimentConfig(
            framework_config=core_triple_config,
            experiment_name="core_triple_integration",
            description="MSNAR + Quantum + Hyperbolic integration",
            num_epochs=45,
            learning_rate=0.0004
        )
        
        # 9. Full Framework: All Components
        full_config = UnifiedFrameworkConfig()
        full_config.enable_msnar = True
        full_config.enable_quantum = True
        full_config.enable_hyperbolic = True
        full_config.enable_meta_learning = True
        full_config.enable_adversarial = True
        
        presets['full_framework'] = ExperimentConfig(
            framework_config=full_config,
            experiment_name="full_neurips_framework",
            description="Complete NeurIPS framework with all 5 novel components",
            num_epochs=50,
            learning_rate=0.0003,
            eval_frequency=2,
            save_frequency=10
        )
        
        # 10. Quick Test: Fast configuration for debugging
        quick_config = UnifiedFrameworkConfig()
        quick_config.enable_msnar = True
        quick_config.enable_quantum = True
        quick_config.enable_hyperbolic = False
        quick_config.enable_meta_learning = False
        quick_config.enable_adversarial = False
        quick_config.batch_size = 2
        
        presets['quick_test'] = ExperimentConfig(
            framework_config=quick_config,
            experiment_name="quick_debug_test",
            description="Quick test configuration for debugging",
            num_epochs=3,
            learning_rate=0.001,
            eval_frequency=1,
            log_frequency=5
        )
        
        return presets
    
    def save_config(self, config: ExperimentConfig, name: str):
        """Save configuration to file"""
        
        config_path = os.path.join(self.config_dir, f"{name}.json")
        
        # Convert to dictionary for JSON serialization
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"💾 Configuration saved to {config_path}")
    
    def load_config(self, name: str) -> ExperimentConfig:
        """Load configuration from file"""
        
        config_path = os.path.join(self.config_dir, f"{name}.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Reconstruct UnifiedFrameworkConfig
        framework_config_dict = config_dict.pop('framework_config')
        framework_config = UnifiedFrameworkConfig()
        
        for key, value in framework_config_dict.items():
            if hasattr(framework_config, key):
                setattr(framework_config, key, value)
        
        # Create ExperimentConfig
        config = ExperimentConfig(framework_config=framework_config, **config_dict)
        
        print(f"📂 Configuration loaded from {config_path}")
        return config
    
    def list_configs(self) -> List[str]:
        """List all available configurations"""
        
        if not os.path.exists(self.config_dir):
            return []
        
        configs = []
        for file in os.listdir(self.config_dir):
            if file.endswith('.json'):
                configs.append(file[:-5])  # Remove .json extension
        
        return configs
    
    def create_all_presets(self):
        """Create and save all preset configurations"""
        
        print("🛠️ Creating preset configurations...")
        
        presets = self.create_preset_configs()
        
        for name, config in presets.items():
            self.save_config(config, name)
        
        print(f"✅ Created {len(presets)} preset configurations")
        return presets
    
    def display_config(self, config: ExperimentConfig):
        """Display configuration in a readable format"""
        
        print(f"\n📋 Experiment Configuration: {config.experiment_name}")
        print(f"Description: {config.description}")
        print(f"Dataset: {config.dataset_type}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Optimizer: {config.optimizer}")
        
        print(f"\n🧩 Enabled Components:")
        framework_config = config.framework_config
        components = [
            ("MSNAR", framework_config.enable_msnar),
            ("Quantum", framework_config.enable_quantum),
            ("Hyperbolic", framework_config.enable_hyperbolic),
            ("Meta-Learning", framework_config.enable_meta_learning),
            ("Adversarial", framework_config.enable_adversarial)
        ]
        
        enabled_count = 0
        for comp_name, enabled in components:
            status = "✅" if enabled else "❌"
            print(f"  {status} {comp_name}")
            if enabled:
                enabled_count += 1
        
        print(f"\nTotal enabled: {enabled_count}/5 components")


def main():
    """Main configuration management function"""
    
    parser = argparse.ArgumentParser(description="NeurIPS Framework Configuration Manager")
    parser.add_argument('command', choices=['create', 'list', 'show', 'run'],
                        help='Command to execute')
    parser.add_argument('--name', type=str, help='Configuration name')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Configuration directory')
    
    args = parser.parse_args()
    
    config_manager = ConfigurationManager(args.config_dir)
    
    if args.command == 'create':
        print("🛠️ Creating preset configurations...")
        presets = config_manager.create_all_presets()
        
        print(f"\n📋 Available configurations:")
        for name in presets.keys():
            print(f"  - {name}")
        
        print(f"\n💡 Usage examples:")
        print(f"  python config_manager.py show --name full_framework")
        print(f"  python train_neurips_framework.py --config full_framework")
        
    elif args.command == 'list':
        configs = config_manager.list_configs()
        if configs:
            print(f"📋 Available configurations ({len(configs)}):")
            for config in configs:
                print(f"  - {config}")
        else:
            print("❌ No configurations found. Run 'create' first.")
    
    elif args.command == 'show':
        if not args.name:
            print("❌ Please specify configuration name with --name")
            return
        
        try:
            config = config_manager.load_config(args.name)
            config_manager.display_config(config)
        except FileNotFoundError as e:
            print(f"❌ {e}")
    
    elif args.command == 'run':
        if not args.name:
            print("❌ Please specify configuration name with --name")
            return
        
        print(f"🚀 Running experiment with configuration: {args.name}")
        print(f"💡 Use: python train_neurips_framework.py --config {args.name}")
    
    print()


if __name__ == "__main__":
    main()
