# NeurIPS Framework - Quick Start Guide

## Overview

The Vision-Mamba-Mender project has been elevated to NeurIPS-level with 5 novel research components:

1. **MSNAR**: Multi-Scale Neuroplasticity Adaptive Repair
2. **Quantum-Inspired Optimization**: State space optimization using quantum principles
3. **Hyperbolic Geometric Manifolds**: Non-Euclidean geometric embedding
4. **Meta-Learning State Evolution**: Adaptive parameter optimization
5. **Adversarial Robustness Generation**: Dynamic adversarial training

## Quick Start

### 1. Create Preset Configurations

```powershell
python config_manager.py create
```

This creates 10 preset experimental configurations for different research scenarios.

### 2. List Available Configurations

```powershell
python config_manager.py list
```

### 3. View a Specific Configuration

```powershell
python config_manager.py show --name full_framework
```

### 4. Run Training Experiments

#### Option A: Use Preset Configuration
```powershell
# Full framework with all 5 components
python train_neurips_framework.py --config full_framework

# Quick test (3 epochs, 2 components)
python train_neurips_framework.py --config quick_test

# Individual components
python train_neurips_framework.py --config msnar_only
python train_neurips_framework.py --config quantum_only
python train_neurips_framework.py --config hyperbolic_only
```

#### Option B: Custom Parameters
```powershell
# Custom training with command line arguments
python train_neurips_framework.py --epochs 20 --batch-size 16 --lr 0.0005

# Smaller dataset for faster testing
python train_neurips_framework.py --train-size 100 --val-size 25 --epochs 5
```

### 5. Run Comprehensive Evaluation

```powershell
python evaluate_neurips_framework.py
```

This runs extensive evaluation including:
- Individual component testing
- Component integration analysis
- Computational efficiency benchmarks
- Robustness analysis

## Available Preset Configurations

| Configuration | Description | Components | Use Case |
|---------------|-------------|------------|----------|
| `baseline` | Original model only | None | Baseline comparison |
| `msnar_only` | MSNAR component | MSNAR | Neuroplasticity research |
| `quantum_only` | Quantum optimization | Quantum | State optimization |
| `hyperbolic_only` | Geometric manifolds | Hyperbolic | Non-Euclidean geometry |
| `msnar_quantum` | MSNAR + Quantum | 2 components | Core integration |
| `meta_learning` | MSNAR + Meta-learning | 2 components | Adaptive learning |
| `adversarial_robust` | MSNAR + Adversarial | 2 components | Robustness research |
| `core_triple` | Core 3 components | MSNAR+Quantum+Hyperbolic | Main research |
| `full_framework` | All components | All 5 | Complete system |
| `quick_test` | Fast debugging | 2 components | Development |

## Expected Results

### Training Output
```
🚀 NeurIPS Framework Training Script
==================================================

📂 Loaded configuration: full_framework
🧪 Experiment: full_neurips_framework
📝 Description: Complete NeurIPS framework with all 5 novel components

📋 Training Configuration:
  Batch size: 8
  Epochs: 50
  Learning rate: 0.0003
  Training size: 200
  Validation size: 50

🧩 Enabled Components (5/5):
  ✅ MSNAR
  ✅ Quantum
  ✅ Hyperbolic
  ✅ Meta-Learning
  ✅ Adversarial

🏋️ Starting Training...
```

### Results Directory Structure
```
experiments/
└── neurips_training/
    ├── training_curves.png
    ├── training_history.json
    ├── neurips_framework_best.pth
    └── neurips_framework_epoch_50.pth
```

### Evaluation Output
```
🔍 NeurIPS Framework Comprehensive Evaluation
==================================================

🔍 Evaluating Individual Components...
  ✅ MSNAR: 0.0234s, Loss: 2.1234
  ✅ Quantum: 0.0187s, Loss: 2.0987
  ✅ Hyperbolic: 0.0298s, Loss: 2.2156
  ✅ Meta-Learning: 0.0267s, Loss: 2.1876
  ✅ Adversarial: 0.0334s, Loss: 2.0765

🔗 Evaluating Component Integration...
  ✅ MSNAR+QUANTUM+HYPERBOLIC: Acc=0.234, Loss=1.9876, Time=0.0523s

📊 Framework is ready for research and experimentation!
```

## Troubleshooting

### Common Issues

1. **Memory Issues (CUDA)**
   ```powershell
   # Use smaller batch size
   python train_neurips_framework.py --batch-size 4 --train-size 100
   ```

2. **CPU-Only Training**
   ```powershell
   # Framework automatically detects CPU/CUDA
   python train_neurips_framework.py --device cpu
   ```

3. **Quick Testing**
   ```powershell
   # Use the quick test configuration
   python train_neurips_framework.py --config quick_test
   ```

### Debugging Steps

1. **Test Individual Components**
   ```powershell
   python test_neurips_framework.py
   ```

2. **Run Component Evaluation**
   ```powershell
   python evaluate_neurips_framework.py
   ```

3. **Check Configuration**
   ```powershell
   python config_manager.py show --name quick_test
   ```

## Research Experiments

### Ablation Study
```powershell
# Test individual components
python train_neurips_framework.py --config baseline
python train_neurips_framework.py --config msnar_only
python train_neurips_framework.py --config quantum_only
python train_neurips_framework.py --config hyperbolic_only

# Test combinations
python train_neurips_framework.py --config msnar_quantum
python train_neurips_framework.py --config core_triple
python train_neurips_framework.py --config full_framework
```

### Performance Analysis
```powershell
# Run comprehensive evaluation
python evaluate_neurips_framework.py

# Check results
ls experiments/evaluation/
```

### Custom Research
```powershell
# Create custom configuration
python config_manager.py create

# Edit configs/custom.json manually

# Run custom experiment
python train_neurips_framework.py --config custom
```

## Next Steps

1. **Analyze Results**: Check `experiments/` directories for training curves and metrics
2. **Compare Configurations**: Use evaluation results to compare different component combinations
3. **Scale Up**: Increase dataset size and training epochs for full experiments
4. **Real Data**: Replace synthetic datasets with real vision data (CIFAR-10, ImageNet, etc.)
5. **Publication**: Use results for NeurIPS submission with ablation studies and performance analysis

## File Structure

```
Vision-Mamba-Mender/
├── core/                              # Novel research components
│   ├── unified_neurips_framework.py   # Main integration framework
│   ├── neuroplasticity_state_repair.py # MSNAR component
│   ├── quantum_inspired_state_optimization.py # Quantum component
│   ├── hyperbolic_geometric_manifolds.py # Hyperbolic component
│   ├── meta_learning_state_evolution.py # Meta-learning component
│   └── adversarial_robustness_generation.py # Adversarial component
├── train_neurips_framework.py         # Main training script
├── evaluate_neurips_framework.py      # Comprehensive evaluation
├── config_manager.py                  # Configuration management
├── test_neurips_framework.py          # Component testing
├── configs/                           # Preset configurations
└── experiments/                       # Results and outputs
```

The framework is now ready for NeurIPS-level research and experimentation! 🚀
