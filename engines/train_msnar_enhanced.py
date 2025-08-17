"""
Enhanced Training Engine with MSNAR Integration

This module provides comprehensive training capabilities that integrate the novel
Mamba State Repair via Neuroplasticity-Inspired Adaptive Reconfiguration (MSNAR)
framework with the existing Vision-Mamba-Mender infrastructure.

Key Enhancements:
1. Real-time state health monitoring and repair
2. Adaptive learning with neuroplasticity principles
3. Robustness against distribution shift and adversarial conditions
4. Theoretical convergence guarantees
5. Comprehensive analysis and visualization
"""

import os
import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.loss import LabelSmoothingCrossEntropy

# Import existing components
import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter

# Import original enhanced components  
from core.adaptive_multiScale_interaction import AdaptiveMultiScaleInteractionLearner, EnhancedMambaInterpreter
from core.causal_state_intervention import CausalStateInterventionFramework, EnhancedCausalMambaFramework
from core.temporal_state_evolution import TemporalStateEvolutionTracker, EnhancedTemporalMambaAnalyzer
from core.constraints import StateConstraint

# Import our novel MSNAR framework
from core.neuroplasticity_state_repair import (
    MSNARFramework, MSNARLoss, NeuroplasticityConfig,
    ConvergenceAnalyzer, MSNARVisualizer,
    create_enhanced_vision_mamba_with_msnar
)


class MSNAREnhancedTrainer:
    """
    Advanced trainer incorporating MSNAR with comprehensive neuroplasticity mechanisms
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        # Initialize all components
        self._init_model()
        self._init_data_loaders()
        self._init_neuroplasticity_config()
        self._init_msnar_framework()
        self._init_optimization()
        self._init_analysis_components()
        self._init_logging()
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.repair_effectiveness_history = []
        self.neuroplasticity_metrics = defaultdict(list)
        
        print(f"MSNAREnhancedTrainer initialized successfully!")
        print(f"Device: {self.device}")
        print(f"Model: {self.args.model_name}")
        print(f"Dataset: {self.args.data_name}")
        
    def _init_model(self):
        """Initialize the base Vision Mamba model"""
        self.model = models.load_model(
            self.args.model_name, 
            num_classes=self.args.num_classes,
            constraint_layers=getattr(self.args, 'constraint_layers', [])
        )
        
        if hasattr(self.args, 'model_path') and self.args.model_path:
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {self.args.model_path}")
        
        self.model.to(self.device)
        
        # Extract model dimensions for MSNAR
        self.state_dim = getattr(self.args, 'state_dim', 256)
        if hasattr(self.model, 'dims'):
            self.state_dim = max(self.model.dims) if self.model.dims else 256
        
        self.num_layers = getattr(self.args, 'num_layers', 6)
        if hasattr(self.model, 'layers'):
            self.num_layers = len(self.model.layers)
        
    def _init_data_loaders(self):
        """Initialize data loaders with enhanced capabilities"""
        self.train_loader = loaders.load_data(
            self.args.data_train_dir, 
            self.args.data_name, 
            data_type='train', 
            batch_size=self.args.batch_size,
            args=self.args
        )
        
        self.test_loader = loaders.load_data(
            self.args.data_test_dir, 
            self.args.data_name, 
            data_type='test', 
            batch_size=self.args.batch_size,
            args=self.args
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Testing samples: {len(self.test_loader.dataset)}")
        
    def _init_neuroplasticity_config(self):
        """Initialize neuroplasticity configuration"""
        self.neuroplasticity_config = NeuroplasticityConfig(
            hebbian_learning_rate=getattr(self.args, 'hebbian_lr', 0.01),
            homeostatic_scaling_factor=getattr(self.args, 'homeostatic_factor', 0.1),
            metaplasticity_threshold=getattr(self.args, 'metaplasticity_threshold', 0.5),
            synaptic_decay=getattr(self.args, 'synaptic_decay', 0.99),
            plasticity_window=getattr(self.args, 'plasticity_window', 100),
            min_activity_threshold=getattr(self.args, 'min_activity', 0.1),
            max_activity_threshold=getattr(self.args, 'max_activity', 0.9),
            adaptation_momentum=getattr(self.args, 'adaptation_momentum', 0.9)
        )
        
    def _init_msnar_framework(self):
        """Initialize the MSNAR framework"""
        self.msnar_framework, self.msnar_loss = create_enhanced_vision_mamba_with_msnar(
            base_model=self.model,
            state_dim=self.state_dim,
            num_layers=self.num_layers
        )
        
        # Override config if custom parameters provided
        self.msnar_framework.config = self.neuroplasticity_config
        self.msnar_framework.to(self.device)
        
        # MSNAR loss weights
        self.msnar_loss.base_loss_weight = getattr(self.args, 'base_loss_weight', 1.0)
        self.msnar_loss.plasticity_weight = getattr(self.args, 'plasticity_weight', 0.1)
        self.msnar_loss.homeostasis_weight = getattr(self.args, 'homeostasis_weight', 0.05)
        self.msnar_loss.correlation_weight = getattr(self.args, 'correlation_weight', 0.1)
        
        print(f"MSNAR Framework initialized with {self.num_layers} layers, {self.state_dim} state dimensions")
        
    def _init_optimization(self):
        """Initialize optimizer and scheduler"""
        # Create optimizer for both original model and MSNAR components
        all_parameters = list(self.model.parameters()) + list(self.msnar_framework.parameters())
        
        self.optimizer = create_optimizer(self.args, self.model)
        
        # Add MSNAR parameters to optimizer
        msnar_param_groups = [
            {
                'params': self.msnar_framework.parameters(),
                'lr': self.args.lr * 0.1,  # Lower learning rate for MSNAR components
                'weight_decay': self.args.weight_decay * 0.5
            }
        ]
        self.optimizer.add_param_group(msnar_param_groups[0])
        
        self.scheduler, _ = create_scheduler(self.args, self.optimizer)
        
        # Loss function
        if getattr(self.args, 'smoothing', 0.0):
            self.base_criterion = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            self.base_criterion = nn.CrossEntropyLoss()
            
        # Original constraint (if applicable)
        if hasattr(self.args, 'alpha') and hasattr(self.args, 'beta'):
            self.original_constraint = StateConstraint(
                model=self.model,
                model_name=self.args.model_name,
                alpha=getattr(self.args, 'alpha', 1e7),
                beta=getattr(self.args, 'beta', 1e7),
                external_cache_layers=getattr(self.args, 'external_cache_layers', []),
                internal_cache_layers=getattr(self.args, 'internal_cache_layers', []),
                external_cache_types=getattr(self.args, 'external_cache_types', []),
                internal_cache_types=getattr(self.args, 'internal_cache_types', []),
                internal_mask_dir=getattr(self.args, 'internal_mask_dir', None)
            )
        else:
            self.original_constraint = None
            
    def _init_analysis_components(self):
        """Initialize analysis and evaluation components"""
        # Legacy enhanced components (keeping for compatibility)
        if self.original_constraint:
            self.adaptive_interpreter = EnhancedMambaInterpreter(
                self.model, self.original_constraint)
            self.causal_framework = EnhancedCausalMambaFramework(
                self.model, self.original_constraint)
            self.temporal_analyzer = EnhancedTemporalMambaAnalyzer(
                self.model, self.original_constraint)
        else:
            self.adaptive_interpreter = None
            self.causal_framework = None
            self.temporal_analyzer = None
            
        # MSNAR-specific analyzers
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.msnar_visualizer = MSNARVisualizer()
        
    def _init_logging(self):
        """Initialize logging and output directories"""
        # Create output directories
        os.makedirs(self.args.model_dir, exist_ok=True)
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        analysis_dir = os.path.join(self.args.model_dir, 'msnar_analysis')
        os.makedirs(analysis_dir, exist_ok=True)
        
        visualization_dir = os.path.join(self.args.model_dir, 'visualizations')
        os.makedirs(visualization_dir, exist_ok=True)
        
        self.analysis_dir = analysis_dir
        self.visualization_dir = visualization_dir
        
        # TensorBoard logging
        self.writer = SummaryWriter(self.args.log_dir)
        
        # Save configuration
        config_path = os.path.join(self.args.model_dir, 'msnar_config.json')
        config_dict = {
            'model_name': self.args.model_name,
            'data_name': self.args.data_name,
            'num_classes': self.args.num_classes,
            'batch_size': self.args.batch_size,
            'num_epochs': self.args.num_epochs,
            'state_dim': self.state_dim,
            'num_layers': self.num_layers,
            'neuroplasticity_config': {
                'hebbian_learning_rate': self.neuroplasticity_config.hebbian_learning_rate,
                'homeostatic_scaling_factor': self.neuroplasticity_config.homeostatic_scaling_factor,
                'metaplasticity_threshold': self.neuroplasticity_config.metaplasticity_threshold,
                'synaptic_decay': self.neuroplasticity_config.synaptic_decay,
                'plasticity_window': self.neuroplasticity_config.plasticity_window
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Enhanced training epoch with MSNAR neuroplasticity mechanisms
        """
        self.model.train()
        self.msnar_framework.train()
        self.current_epoch = epoch
        
        # Metrics tracking
        metrics_dict = {
            'loss': AverageMeter('Loss', ':.4e'),
            'accuracy': AverageMeter('Acc@1', ':6.2f'),
            'msnar_total_loss': AverageMeter('MSNAR Loss', ':.4e'),
            'plasticity_loss': AverageMeter('Plasticity', ':.4e'),
            'homeostatic_loss': AverageMeter('Homeostatic', ':.4e'),
            'correlation_loss': AverageMeter('Correlation', ':.4e'),
            'repair_ratio': AverageMeter('Repair Ratio', ':.4e'),
            'state_health': AverageMeter('State Health', ':.4f')
        }
        
        progress = ProgressMeter(
            total=len(self.train_loader),
            step=max(1, len(self.train_loader) // 20),
            prefix=f'Epoch [{epoch+1}/{self.args.num_epochs}]',
            meters=list(metrics_dict.values())
        )
        
        epoch_neuroplasticity_data = []
        
        for i, samples in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")):
            inputs, labels, _ = samples
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            batch_size = inputs.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # MSNAR forward pass with neuroplasticity mechanisms
            msnar_output = self.msnar_framework(inputs, target_performance=None)
            
            # Compute MSNAR loss
            msnar_loss_output = self.msnar_loss(msnar_output, labels)
            
            # Legacy enhanced losses (reduced weight for compatibility)
            legacy_loss = torch.tensor(0.0, device=self.device)
            if self.adaptive_interpreter and i % 20 == 0:  # Reduce frequency
                try:
                    adaptive_results = self.adaptive_interpreter.enhanced_loss_computation(
                        msnar_output['output'], labels, inputs, epoch, self.args.num_epochs)
                    legacy_loss += 0.01 * adaptive_results['adaptive_loss']  # Very small weight
                except Exception as e:
                    pass  # Skip if legacy component fails
            
            # Total loss
            total_loss = msnar_loss_output['total_loss'] + legacy_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping for stability
            if hasattr(self.args, 'clip_grad') and self.args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.msnar_framework.parameters()), 
                    self.args.clip_grad)
            
            # Optimization step
            self.optimizer.step()
            
            # Compute accuracy
            acc1, = metrics.accuracy(msnar_output['output'], labels)
            
            # Update metrics
            metrics_dict['loss'].update(total_loss.item(), batch_size)
            metrics_dict['accuracy'].update(acc1.item(), batch_size)
            metrics_dict['msnar_total_loss'].update(msnar_loss_output['total_loss'].item(), batch_size)
            metrics_dict['plasticity_loss'].update(msnar_loss_output['plasticity_loss'].item(), batch_size)
            metrics_dict['homeostatic_loss'].update(msnar_loss_output['homeostatic_loss'].item(), batch_size)
            metrics_dict['correlation_loss'].update(msnar_loss_output['correlation_loss'].item(), batch_size)
            
            # MSNAR-specific metrics
            repair_ratio = msnar_output['repair_applied']['repair_ratio']
            state_health = msnar_output['state_health_scores'].mean().item()
            
            metrics_dict['repair_ratio'].update(repair_ratio, batch_size)
            metrics_dict['state_health'].update(state_health, batch_size)
            
            # Store detailed neuroplasticity data periodically
            if i % 50 == 0:
                neuroplasticity_summary = self.msnar_framework.get_neuroplasticity_summary()
                epoch_neuroplasticity_data.append({
                    'step': self.global_step,
                    'repair_ratio': repair_ratio,
                    'state_health': state_health,
                    'summary': neuroplasticity_summary
                })
            
            # Performance tracking for meta-plasticity
            self.performance_history.append(acc1.item())
            
            # TensorBoard logging (detailed)
            if i % 100 == 0:
                self._log_step_metrics(msnar_output, msnar_loss_output, acc1.item())
            
            # Progress display
            if i % progress.step == 0:
                progress.display(i)
            
            self.global_step += 1
            
            # Clear caches to prevent memory issues
            if self.original_constraint:
                self.original_constraint.del_cache()
        
        # Store epoch-level neuroplasticity metrics
        self.neuroplasticity_metrics['epoch_data'].append(epoch_neuroplasticity_data)
        
        # Compute epoch averages
        epoch_results = {key: meter.avg for key, meter in metrics_dict.items()}
        
        return epoch_results
    
    def evaluate(self, epoch: int, save_analysis: bool = False) -> Dict[str, float]:
        """
        Enhanced evaluation with comprehensive MSNAR analysis
        """
        self.model.eval()
        self.msnar_framework.eval()
        
        metrics_dict = {
            'loss': AverageMeter('Loss', ':.4e'),
            'accuracy': AverageMeter('Acc@1', ':6.2f'),
            'state_health': AverageMeter('State Health', ':.4f'),
            'repair_effectiveness': AverageMeter('Repair Eff', ':.4f')
        }
        
        comprehensive_analysis = {
            'neuroplasticity_summaries': [],
            'stability_analyses': [],
            'repair_patterns': []
        }
        
        with torch.no_grad():
            for i, samples in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                inputs, labels, _ = samples
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                batch_size = inputs.size(0)
                
                # MSNAR forward pass
                msnar_output = self.msnar_framework(inputs)
                
                # Loss computation
                msnar_loss_output = self.msnar_loss(msnar_output, labels)
                loss = msnar_loss_output['total_loss']
                
                # Accuracy
                acc1, = metrics.accuracy(msnar_output['output'], labels)
                
                # MSNAR metrics
                state_health = msnar_output['state_health_scores'].mean().item()
                repair_effectiveness = msnar_output['repair_applied']['repair_ratio']
                
                # Update meters
                metrics_dict['loss'].update(loss.item(), batch_size)
                metrics_dict['accuracy'].update(acc1.item(), batch_size)
                metrics_dict['state_health'].update(state_health, batch_size)
                metrics_dict['repair_effectiveness'].update(repair_effectiveness, batch_size)
                
                # Comprehensive analysis on subset
                if save_analysis and i % 20 == 0 and i < 100:
                    # Neuroplasticity summary
                    summary = self.msnar_framework.get_neuroplasticity_summary()
                    comprehensive_analysis['neuroplasticity_summaries'].append(summary)
                    
                    # Stability analysis
                    stability = self.convergence_analyzer.lyapunov_stability_analysis(
                        self.msnar_framework)
                    comprehensive_analysis['stability_analyses'].append(stability)
                    
                    # Repair pattern analysis
                    repair_pattern = {
                        'per_layer_repair': msnar_output['repair_applied']['per_layer_repair'],
                        'total_repair_magnitude': msnar_output['repair_applied']['total_repair_magnitude'],
                        'state_health_scores': msnar_output['state_health_scores'].cpu().numpy().tolist()
                    }
                    comprehensive_analysis['repair_patterns'].append(repair_pattern)
        
        # Compute epoch averages
        eval_results = {key: meter.avg for key, meter in metrics_dict.items()}
        eval_results['comprehensive_analysis'] = comprehensive_analysis
        
        return eval_results
    
    def _log_step_metrics(self, msnar_output: Dict, loss_output: Dict, accuracy: float):
        """Log detailed step-level metrics to TensorBoard"""
        
        # Basic metrics
        self.writer.add_scalar('Train/StepAccuracy', accuracy, self.global_step)
        self.writer.add_scalar('Train/StepLoss', loss_output['total_loss'].item(), self.global_step)
        
        # MSNAR loss components
        self.writer.add_scalar('MSNAR/PlasticityLoss', loss_output['plasticity_loss'].item(), self.global_step)
        self.writer.add_scalar('MSNAR/HomeostaticLoss', loss_output['homeostatic_loss'].item(), self.global_step)
        self.writer.add_scalar('MSNAR/CorrelationLoss', loss_output['correlation_loss'].item(), self.global_step)
        
        # Neuroplasticity metrics
        self.writer.add_scalar('Neuroplasticity/StateHealth', 
                              msnar_output['state_health_scores'].mean().item(), self.global_step)
        self.writer.add_scalar('Neuroplasticity/RepairRatio', 
                              msnar_output['repair_applied']['repair_ratio'], self.global_step)
        
        # State health per layer
        for i, health in enumerate(msnar_output['state_health_scores']):
            self.writer.add_scalar(f'StateHealth/Layer_{i}', health.item(), self.global_step)
    
    def _log_epoch_metrics(self, epoch: int, train_results: Dict, eval_results: Dict):
        """Log epoch-level metrics"""
        
        # Training metrics
        for key, value in train_results.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Evaluation metrics
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Eval/{key}', value, epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
        
        # Neuroplasticity summary
        if self.neuroplasticity_metrics['epoch_data']:
            latest_data = self.neuroplasticity_metrics['epoch_data'][-1]
            if latest_data:
                summary = latest_data[-1]['summary']  # Last summary of the epoch
                
                self.writer.add_scalar('Neuroplasticity/AvgHealth', 
                                     summary['average_health'], epoch)
                self.writer.add_scalar('Neuroplasticity/UnhealthyLayers', 
                                     summary['unhealthy_layers'], epoch)
        
        # Console logging
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train - Loss: {train_results['loss']:.4f}, Acc: {train_results['accuracy']:.2f}%, "
              f"State Health: {train_results['state_health']:.4f}")
        print(f"Eval  - Loss: {eval_results['loss']:.4f}, Acc: {eval_results['accuracy']:.2f}%, "
              f"Repair Eff: {eval_results['repair_effectiveness']:.4f}")
        
    def train(self):
        """
        Main training loop with comprehensive MSNAR integration
        """
        print(f"\nStarting MSNAR-Enhanced Vision Mamba Training")
        print(f"{'='*80}")
        print(f"Model: {self.args.model_name}")
        print(f"Dataset: {self.args.data_name} ({self.args.num_classes} classes)")
        print(f"Epochs: {self.args.num_epochs}")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Neuroplasticity: ENABLED with {self.num_layers} layers, {self.state_dim}D states")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        # Training loop
        for epoch in range(self.args.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_results = self.train_epoch(epoch)
            
            # Evaluation
            save_analysis = (epoch % 10 == 0) or (epoch == self.args.num_epochs - 1)
            eval_results = self.evaluate(epoch, save_analysis=save_analysis)
            
            # Learning rate scheduling
            self.scheduler.step(epoch=epoch)
            
            # Logging
            self._log_epoch_metrics(epoch, train_results, eval_results)
            
            # Model saving
            is_best = eval_results['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = eval_results['accuracy']
                self.best_epoch = epoch
                self._save_checkpoint(epoch, train_results, eval_results, is_best=True)
                
                # Save detailed analysis for best model
                if save_analysis:
                    self._save_comprehensive_analysis(epoch, eval_results['comprehensive_analysis'])
            
            # Regular checkpoint saving
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, train_results, eval_results, is_best=False)
            
            # Update repair effectiveness history
            self.repair_effectiveness_history.append({
                'epoch': epoch,
                'repair_ratio': train_results['repair_ratio'],
                'total_repair_magnitude': 0.0,  # Would be computed from detailed data
                'state_health': train_results['state_health']
            })
            
            # Theoretical analysis (periodic)
            if epoch % 5 == 0:
                self._perform_theoretical_analysis(epoch)
            
            # Visualization generation (periodic)
            if epoch % 20 == 0:
                self._generate_visualizations(epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s\\n\")\n        \n        print(f\"\\n{'='*80}\")\n        print(\"MSNAR Training Completed!\")\n        print(f\"Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch+1}\")\n        print(f\"Model saved in: {self.args.model_dir}\")\n        print(f\"Analysis saved in: {self.analysis_dir}\")\n        print(f\"Visualizations saved in: {self.visualization_dir}\")\n        print(f\"{'='*80}\")\n        \n        # Final comprehensive analysis and report\n        self._generate_final_report()\n        \n    def _save_checkpoint(self, epoch: int, train_results: Dict, eval_results: Dict, is_best: bool):\n        \"\"\"Save model checkpoint with MSNAR state\"\"\"\n        \n        checkpoint = {\n            'epoch': epoch,\n            'model_state_dict': self.model.state_dict(),\n            'msnar_state_dict': self.msnar_framework.state_dict(),\n            'optimizer_state_dict': self.optimizer.state_dict(),\n            'scheduler_state_dict': self.scheduler.state_dict(),\n            'best_accuracy': self.best_accuracy,\n            'train_results': train_results,\n            'eval_results': eval_results,\n            'neuroplasticity_config': self.neuroplasticity_config.__dict__,\n            'neuroplasticity_metrics': dict(self.neuroplasticity_metrics)\n        }\n        \n        # Save checkpoint\n        filename = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'\n        checkpoint_path = os.path.join(self.args.model_dir, filename)\n        torch.save(checkpoint, checkpoint_path)\n        \n        if is_best:\n            print(f\"New best model saved: {self.best_accuracy:.2f}% accuracy\")\n            \n    def _save_comprehensive_analysis(self, epoch: int, analysis: Dict):\n        \"\"\"Save comprehensive analysis data\"\"\"\n        \n        analysis_path = os.path.join(self.analysis_dir, f'comprehensive_analysis_epoch_{epoch}.json')\n        \n        # Convert to serializable format\n        serializable_analysis = self._make_serializable(analysis)\n        \n        with open(analysis_path, 'w') as f:\n            json.dump(serializable_analysis, f, indent=2)\n            \n    def _perform_theoretical_analysis(self, epoch: int):\n        \"\"\"Perform theoretical convergence analysis\"\"\"\n        \n        # Lyapunov stability analysis\n        stability_metrics = self.convergence_analyzer.lyapunov_stability_analysis(\n            self.msnar_framework)\n        \n        # Convergence bounds\n        convergence_bounds = self.convergence_analyzer.convergence_bounds(\n            self.neuroplasticity_config, self.num_layers)\n        \n        # Log to TensorBoard\n        self.writer.add_scalar('Theory/SpectralRadius', \n                              stability_metrics['spectral_radius'], epoch)\n        self.writer.add_scalar('Theory/StabilityMargin', \n                              stability_metrics['stability_margin'], epoch)\n        self.writer.add_scalar('Theory/ConvergenceRate', \n                              convergence_bounds['convergence_rate'], epoch)\n        \n        # Store analysis\n        theoretical_analysis = {\n            'epoch': epoch,\n            'stability_metrics': stability_metrics,\n            'convergence_bounds': convergence_bounds\n        }\n        \n        analysis_path = os.path.join(self.analysis_dir, f'theoretical_analysis_epoch_{epoch}.json')\n        with open(analysis_path, 'w') as f:\n            json.dump(self._make_serializable(theoretical_analysis), f, indent=2)\n            \n    def _generate_visualizations(self, epoch: int):\n        \"\"\"Generate neuroplasticity visualizations\"\"\"\n        \n        try:\n            # Neuroplasticity dynamics plot\n            dynamics_path = os.path.join(self.visualization_dir, \n                                       f'neuroplasticity_dynamics_epoch_{epoch}.png')\n            self.msnar_visualizer.plot_neuroplasticity_dynamics(\n                self.msnar_framework, save_path=dynamics_path)\n            \n            # Repair effectiveness plot\n            if self.repair_effectiveness_history:\n                effectiveness_path = os.path.join(self.visualization_dir, \n                                                 f'repair_effectiveness_epoch_{epoch}.png')\n                self.msnar_visualizer.create_repair_effectiveness_plot(\n                    self.repair_effectiveness_history, save_path=effectiveness_path)\n                    \n        except Exception as e:\n            print(f\"Warning: Visualization generation failed: {e}\")\n            \n    def _generate_final_report(self):\n        \"\"\"Generate comprehensive final analysis report\"\"\"\n        \n        report_path = os.path.join(self.args.model_dir, 'msnar_final_report.txt')\n        \n        with open(report_path, 'w') as f:\n            f.write(\"MSNAR-Enhanced Vision Mamba Training Report\\n\")\n            f.write(\"=\" * 60 + \"\\n\\n\")\n            \n            # Basic configuration\n            f.write(f\"Model: {self.args.model_name}\\n\")\n            f.write(f\"Dataset: {self.args.data_name}\\n\")\n            f.write(f\"Classes: {self.args.num_classes}\\n\")\n            f.write(f\"Epochs: {self.args.num_epochs}\\n\")\n            f.write(f\"Batch Size: {self.args.batch_size}\\n\")\n            f.write(f\"Best Accuracy: {self.best_accuracy:.2f}% (Epoch {self.best_epoch+1})\\n\\n\")\n            \n            # MSNAR configuration\n            f.write(\"MSNAR Configuration:\\n\")\n            f.write(\"-\" * 30 + \"\\n\")\n            f.write(f\"State Dimensions: {self.state_dim}\\n\")\n            f.write(f\"Number of Layers: {self.num_layers}\\n\")\n            f.write(f\"Hebbian Learning Rate: {self.neuroplasticity_config.hebbian_learning_rate}\\n\")\n            f.write(f\"Homeostatic Scaling: {self.neuroplasticity_config.homeostatic_scaling_factor}\\n\")\n            f.write(f\"Meta-plasticity Threshold: {self.neuroplasticity_config.metaplasticity_threshold}\\n\")\n            f.write(f\"Synaptic Decay: {self.neuroplasticity_config.synaptic_decay}\\n\\n\")\n            \n            # Performance summary\n            f.write(\"Performance Summary:\\n\")\n            f.write(\"-\" * 30 + \"\\n\")\n            \n            if self.repair_effectiveness_history:\n                avg_repair_ratio = np.mean([h['repair_ratio'] for h in self.repair_effectiveness_history])\n                avg_state_health = np.mean([h['state_health'] for h in self.repair_effectiveness_history])\n                \n                f.write(f\"Average Repair Ratio: {avg_repair_ratio:.4f}\\n\")\n                f.write(f\"Average State Health: {avg_state_health:.4f}\\n\")\n            \n            # Final neuroplasticity summary\n            final_summary = self.msnar_framework.get_neuroplasticity_summary()\n            f.write(f\"Final Average Health: {final_summary['average_health']:.4f}\\n\")\n            f.write(f\"Final Unhealthy Layers: {final_summary['unhealthy_layers']}\\n\")\n            \n            # Theoretical guarantees\n            f.write(\"\\nTheoretical Analysis:\\n\")\n            f.write(\"-\" * 30 + \"\\n\")\n            \n            try:\n                stability_metrics = self.convergence_analyzer.lyapunov_stability_analysis(\n                    self.msnar_framework)\n                convergence_bounds = self.convergence_analyzer.convergence_bounds(\n                    self.neuroplasticity_config, self.num_layers)\n                \n                f.write(f\"System Stability: {'STABLE' if stability_metrics['is_stable'] else 'UNSTABLE'}\\n\")\n                f.write(f\"Spectral Radius: {stability_metrics['spectral_radius']:.4f}\\n\")\n                f.write(f\"Convergence Rate: {convergence_bounds['convergence_rate']:.6f}\\n\")\n                f.write(f\"Time to Convergence: {convergence_bounds['time_to_convergence']:.2f} steps\\n\")\n                \n            except Exception as e:\n                f.write(f\"Theoretical analysis failed: {e}\\n\")\n            \n            f.write(\"\\nMSNAR training completed successfully with neuroplasticity-inspired state repair.\\n\")\n            \n        print(f\"Final analysis report saved to: {report_path}\")\n        \n    def _make_serializable(self, data):\n        \"\"\"Convert tensors and other non-serializable objects to serializable format\"\"\"\n        if isinstance(data, torch.Tensor):\n            return data.detach().cpu().numpy().tolist()\n        elif isinstance(data, np.ndarray):\n            return data.tolist()\n        elif isinstance(data, dict):\n            return {k: self._make_serializable(v) for k, v in data.items()}\n        elif isinstance(data, list):\n            return [self._make_serializable(item) for item in data]\n        elif isinstance(data, (int, float, str, bool)):\n            return data\n        else:\n            return str(data)  # Fallback to string representation\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='MSNAR-Enhanced Vision-Mamba-Mender Training')\n    \n    # Basic arguments\n    parser.add_argument('--model_name', default='vmamba_tiny', type=str, help='model name')\n    parser.add_argument('--data_name', default='imagenet', type=str, help='data name')\n    parser.add_argument('--num_classes', default=1000, type=int, help='num classes')\n    parser.add_argument('--num_epochs', default=100, type=int, help='num epochs')\n    parser.add_argument('--batch_size', default=64, type=int, help='batch size')\n    \n    # Paths\n    parser.add_argument('--model_dir', default='./outputs/msnar_models', type=str, help='model dir')\n    parser.add_argument('--data_train_dir', default='./data/train', type=str, help='train data dir')\n    parser.add_argument('--data_test_dir', default='./data/test', type=str, help='test data dir')\n    parser.add_argument('--log_dir', default='./logs/msnar_training', type=str, help='log dir')\n    parser.add_argument('--model_path', default=None, type=str, help='path to pretrained model')\n    \n    # Optimization\n    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')\n    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')\n    parser.add_argument('--opt', default='adamw', type=str, help='optimizer')\n    parser.add_argument('--sched', default='cosine', type=str, help='scheduler')\n    parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')\n    parser.add_argument('--clip_grad', type=float, default=1.0, help='gradient clipping')\n    parser.add_argument('--smoothing', type=float, default=0.1, help='label smoothing')\n    \n    # MSNAR-specific arguments\n    parser.add_argument('--state_dim', default=256, type=int, help='state dimension for MSNAR')\n    parser.add_argument('--num_layers', default=6, type=int, help='number of layers')\n    \n    # Neuroplasticity configuration\n    parser.add_argument('--hebbian_lr', type=float, default=0.01, help='Hebbian learning rate')\n    parser.add_argument('--homeostatic_factor', type=float, default=0.1, help='homeostatic scaling factor')\n    parser.add_argument('--metaplasticity_threshold', type=float, default=0.5, help='meta-plasticity threshold')\n    parser.add_argument('--synaptic_decay', type=float, default=0.99, help='synaptic decay rate')\n    parser.add_argument('--plasticity_window', type=int, default=100, help='plasticity window size')\n    parser.add_argument('--min_activity', type=float, default=0.1, help='minimum activity threshold')\n    parser.add_argument('--max_activity', type=float, default=0.9, help='maximum activity threshold')\n    parser.add_argument('--adaptation_momentum', type=float, default=0.9, help='adaptation momentum')\n    \n    # MSNAR loss weights\n    parser.add_argument('--base_loss_weight', type=float, default=1.0, help='base classification loss weight')\n    parser.add_argument('--plasticity_weight', type=float, default=0.1, help='plasticity loss weight')\n    parser.add_argument('--homeostasis_weight', type=float, default=0.05, help='homeostatic loss weight')\n    parser.add_argument('--correlation_weight', type=float, default=0.1, help='correlation loss weight')\n    \n    # Legacy compatibility\n    parser.add_argument('--constraint_layers', nargs='*', type=int, default=[1, 3, 5], \n                       help='layers for constraint analysis')\n    parser.add_argument('--alpha', default=1e5, type=float, help='weight of external loss')\n    parser.add_argument('--beta', default=1e5, type=float, help='weight of internal loss')\n    parser.add_argument('--external_cache_layers', default=[1, 3], nargs='*', type=int)\n    parser.add_argument('--internal_cache_layers', default=[1, 3], nargs='*', type=int)\n    parser.add_argument('--external_cache_types', default=['x', 'c'], nargs='*', type=str)\n    parser.add_argument('--internal_cache_types', default=['x', 'c'], nargs='*', type=str)\n    \n    args = parser.parse_args()\n    \n    # Create and run trainer\n    trainer = MSNAREnhancedTrainer(args)\n    trainer.train()\n\n\nif __name__ == '__main__':\n    main()"
