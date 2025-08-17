"""
MSNAR Evaluation Engine

This module provides comprehensive evaluation capabilities for the MSNAR
(Mamba State Repair via Neuroplasticity-Inspired Adaptive Reconfiguration) framework.

Features:
1. Robustness evaluation under various conditions
2. State repair effectiveness analysis
3. Neuroplasticity mechanism assessment
4. Theoretical validation of convergence properties
5. Comprehensive benchmarking and comparison
"""

import os
import argparse
import time
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import existing components
import loaders
import models
import metrics

# Import MSNAR framework
from core.neuroplasticity_state_repair import (
    MSNARFramework, MSNARLoss, NeuroplasticityConfig,
    ConvergenceAnalyzer, MSNARVisualizer,
    create_enhanced_vision_mamba_with_msnar
)

# Import for adversarial testing
try:
    import torchattacks
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False
    print("Warning: torchattacks not available. Adversarial robustness tests will be skipped.")


class MSNARRobustnessEvaluator:
    """
    Comprehensive robustness evaluation for MSNAR framework
    """
    
    def __init__(self, msnar_framework: MSNARFramework, device: torch.device):
        self.msnar_framework = msnar_framework
        self.device = device
        self.results = defaultdict(list)
        
    def evaluate_noise_robustness(self, 
                                data_loader: DataLoader, 
                                noise_levels: List[float]) -> Dict[str, List[float]]:
        """
        Evaluate robustness to various levels of input noise
        """
        print("Evaluating noise robustness...")
        
        noise_results = {
            'noise_levels': noise_levels,
            'clean_accuracy': 0.0,
            'noisy_accuracies': [],
            'repair_ratios': [],
            'state_health_scores': []
        }
        
        self.msnar_framework.eval()
        
        # First, evaluate clean performance
        clean_acc = self._evaluate_clean_accuracy(data_loader)
        noise_results['clean_accuracy'] = clean_acc
        
        # Then evaluate with noise
        for noise_level in tqdm(noise_levels, desc="Noise levels"):
            total_correct = 0
            total_samples = 0
            repair_ratios = []
            state_healths = []
            
            with torch.no_grad():
                for i, (inputs, labels, _) in enumerate(data_loader):
                    if i >= 50:  # Limit evaluation for efficiency
                        break
                        
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Add noise
                    noise = torch.randn_like(inputs) * noise_level
                    noisy_inputs = torch.clamp(inputs + noise, 0, 1)
                    
                    # MSNAR forward pass
                    msnar_output = self.msnar_framework(noisy_inputs)
                    
                    # Compute accuracy
                    _, predicted = torch.max(msnar_output['output'], 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    # Collect MSNAR metrics
                    repair_ratios.append(msnar_output['repair_applied']['repair_ratio'])
                    state_healths.append(msnar_output['state_health_scores'].mean().item())
            
            accuracy = 100.0 * total_correct / total_samples
            avg_repair_ratio = np.mean(repair_ratios)
            avg_state_health = np.mean(state_healths)
            
            noise_results['noisy_accuracies'].append(accuracy)
            noise_results['repair_ratios'].append(avg_repair_ratio)
            noise_results['state_health_scores'].append(avg_state_health)
        
        return noise_results
    
    def evaluate_adversarial_robustness(self, 
                                      data_loader: DataLoader,
                                      attack_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate robustness against adversarial attacks
        """
        if not ADVERSARIAL_AVAILABLE:
            return {'error': 'torchattacks not available'}
            
        print("Evaluating adversarial robustness...")
        
        adversarial_results = {
            'attacks': [],
            'clean_accuracy': 0.0,
            'adversarial_accuracies': {},
            'repair_effectiveness': {}
        }
        
        self.msnar_framework.eval()
        
        # Clean accuracy
        clean_acc = self._evaluate_clean_accuracy(data_loader)
        adversarial_results['clean_accuracy'] = clean_acc
        
        # Create attacks
        attacks = {}
        if 'fgsm' in attack_params:
            attacks['FGSM'] = torchattacks.FGSM(self.msnar_framework, eps=attack_params['fgsm']['eps'])
        if 'pgd' in attack_params:
            attacks['PGD'] = torchattacks.PGD(self.msnar_framework, 
                                            eps=attack_params['pgd']['eps'],
                                            alpha=attack_params['pgd']['alpha'],
                                            steps=attack_params['pgd']['steps'])
        
        # Evaluate each attack
        for attack_name, attack in attacks.items():
            print(f"Evaluating {attack_name} attack...")
            
            total_correct = 0
            total_samples = 0
            repair_ratios = []
            
            for i, (inputs, labels, _) in enumerate(tqdm(data_loader, desc=f"{attack_name}")):
                if i >= 30:  # Limit for efficiency
                    break
                    
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Generate adversarial examples
                adv_inputs = attack(inputs, labels)
                
                # MSNAR forward pass
                with torch.no_grad():
                    msnar_output = self.msnar_framework(adv_inputs)
                
                # Compute accuracy
                _, predicted = torch.max(msnar_output['output'], 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                
                # Collect repair metrics
                repair_ratios.append(msnar_output['repair_applied']['repair_ratio'])
            
            accuracy = 100.0 * total_correct / total_samples
            avg_repair_ratio = np.mean(repair_ratios)
            
            adversarial_results['adversarial_accuracies'][attack_name] = accuracy
            adversarial_results['repair_effectiveness'][attack_name] = avg_repair_ratio
            adversarial_results['attacks'].append(attack_name)
        
        return adversarial_results
    
    def evaluate_distribution_shift(self, 
                                  clean_loader: DataLoader,
                                  shifted_loaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """
        Evaluate robustness under distribution shift
        """
        print("Evaluating distribution shift robustness...")
        
        shift_results = {
            'clean_accuracy': 0.0,
            'shifted_accuracies': {},
            'adaptation_metrics': {}
        }
        
        self.msnar_framework.eval()
        
        # Clean performance
        clean_acc = self._evaluate_clean_accuracy(clean_loader)
        shift_results['clean_accuracy'] = clean_acc
        
        # Evaluate on shifted distributions
        for shift_name, shifted_loader in shifted_loaders.items():
            print(f"Evaluating {shift_name} distribution...")
            
            total_correct = 0
            total_samples = 0
            adaptation_scores = []
            
            with torch.no_grad():
                for i, (inputs, labels, _) in enumerate(tqdm(shifted_loader, desc=shift_name)):
                    if i >= 50:  # Limit evaluation
                        break
                        
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # MSNAR forward pass
                    msnar_output = self.msnar_framework(inputs)
                    
                    # Compute accuracy
                    _, predicted = torch.max(msnar_output['output'], 1)
                    total_correct += (predicted == labels).sum().item()
                    total_samples += labels.size(0)
                    
                    # Measure adaptation (repair effectiveness)
                    adaptation_score = msnar_output['repair_applied']['repair_ratio']
                    adaptation_scores.append(adaptation_score)
            
            accuracy = 100.0 * total_correct / total_samples
            avg_adaptation = np.mean(adaptation_scores)
            
            shift_results['shifted_accuracies'][shift_name] = accuracy
            shift_results['adaptation_metrics'][shift_name] = avg_adaptation
        
        return shift_results
    
    def _evaluate_clean_accuracy(self, data_loader: DataLoader) -> float:
        """Evaluate clean accuracy"""
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for i, (inputs, labels, _) in enumerate(data_loader):
                if i >= 50:  # Limit evaluation
                    break
                    
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                msnar_output = self.msnar_framework(inputs)
                _, predicted = torch.max(msnar_output['output'], 1)
                
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
        
        return 100.0 * total_correct / total_samples


class MSNARComprehensiveEvaluator:
    """
    Main evaluation framework for MSNAR
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._init_model_and_msnar()
        self._init_data_loaders()
        self._init_evaluators()
        self._init_output_dirs()
        
        print(f"MSNAR Comprehensive Evaluator initialized!")
        print(f"Device: {self.device}")
        print(f"Model: {self.args.model_name}")
        
    def _init_model_and_msnar(self):
        """Initialize model and MSNAR framework"""
        # Load base model
        self.model = models.load_model(
            self.args.model_name, 
            num_classes=self.args.num_classes
        )
        
        # Load checkpoint if provided
        if self.args.model_path:
            checkpoint = torch.load(self.args.model_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"Loaded model from {self.args.model_path}")
        
        self.model.to(self.device)
        
        # Extract dimensions
        self.state_dim = getattr(self.args, 'state_dim', 256)
        self.num_layers = getattr(self.args, 'num_layers', 6)
        
        # Create MSNAR framework
        self.msnar_framework, self.msnar_loss = create_enhanced_vision_mamba_with_msnar(
            base_model=self.model,
            state_dim=self.state_dim,
            num_layers=self.num_layers
        )
        
        # Load MSNAR state if available
        if self.args.model_path and 'msnar_state_dict' in checkpoint:
            self.msnar_framework.load_state_dict(checkpoint['msnar_state_dict'])
            print("Loaded MSNAR state from checkpoint")
        
        self.msnar_framework.to(self.device)
        
    def _init_data_loaders(self):
        """Initialize data loaders"""
        self.test_loader = loaders.load_data(
            self.args.data_test_dir, 
            self.args.data_name, 
            data_type='test', 
            batch_size=self.args.batch_size,
            args=self.args
        )
        
        # For distribution shift evaluation, we might need multiple test sets
        self.shifted_loaders = {}
        
        # Add corrupted versions if requested
        if getattr(self.args, 'evaluate_corrupted', False):
            # This would be implemented based on available corrupted datasets
            pass
            
    def _init_evaluators(self):
        """Initialize specialized evaluators"""
        self.robustness_evaluator = MSNARRobustnessEvaluator(
            self.msnar_framework, self.device)
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.msnar_visualizer = MSNARVisualizer()
        
    def _init_output_dirs(self):
        """Initialize output directories"""
        self.output_dir = self.args.output_dir
        self.analysis_dir = os.path.join(self.output_dir, 'analysis')
        self.visualization_dir = os.path.join(self.output_dir, 'visualizations')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
    def evaluate_basic_performance(self) -> Dict[str, float]:
        """
        Evaluate basic classification performance
        """
        print("Evaluating basic performance...")
        
        self.msnar_framework.eval()
        
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        
        neuroplasticity_metrics = {
            'repair_ratios': [],
            'state_health_scores': [],
            'plasticity_magnitudes': []
        }
        
        with torch.no_grad():
            for inputs, labels, _ in tqdm(self.test_loader, desc="Basic evaluation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # MSNAR forward pass
                msnar_output = self.msnar_framework(inputs)
                
                # Loss computation
                loss_output = self.msnar_loss(msnar_output, labels)
                
                # Accuracy
                _, predicted = torch.max(msnar_output['output'], 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss_output['total_loss'].item()
                
                # Neuroplasticity metrics
                neuroplasticity_metrics['repair_ratios'].append(
                    msnar_output['repair_applied']['repair_ratio'])
                neuroplasticity_metrics['state_health_scores'].append(
                    msnar_output['state_health_scores'].mean().item())
        
        accuracy = 100.0 * total_correct / total_samples
        avg_loss = total_loss / len(self.test_loader)
        
        # Compute average neuroplasticity metrics
        avg_repair_ratio = np.mean(neuroplasticity_metrics['repair_ratios'])
        avg_state_health = np.mean(neuroplasticity_metrics['state_health_scores'])
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'repair_ratio': avg_repair_ratio,
            'state_health': avg_state_health,
            'total_samples': total_samples
        }
    
    def evaluate_neuroplasticity_mechanisms(self) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of neuroplasticity mechanisms
        """
        print("Evaluating neuroplasticity mechanisms...")
        
        # Test with and without neuroplasticity to show effectiveness
        results = {
            'with_neuroplasticity': {},
            'without_neuroplasticity': {},
            'improvement_metrics': {}
        }
        
        # Evaluate with neuroplasticity (normal MSNAR)
        results['with_neuroplasticity'] = self.evaluate_basic_performance()
        
        # Temporarily disable neuroplasticity mechanisms for comparison
        original_config = self.msnar_framework.config
        disabled_config = copy.deepcopy(original_config)
        disabled_config.hebbian_learning_rate = 0.0
        disabled_config.homeostatic_scaling_factor = 0.0
        
        self.msnar_framework.config = disabled_config
        results['without_neuroplasticity'] = self.evaluate_basic_performance()
        
        # Restore original configuration
        self.msnar_framework.config = original_config
        
        # Compute improvement metrics
        for metric in ['accuracy', 'state_health', 'repair_ratio']:\n            improvement = (results['with_neuroplasticity'][metric] - \n                          results['without_neuroplasticity'][metric])\n            relative_improvement = improvement / (results['without_neuroplasticity'][metric] + 1e-8)\n            \n            results['improvement_metrics'][f'{metric}_improvement'] = improvement\n            results['improvement_metrics'][f'{metric}_relative_improvement'] = relative_improvement\n        \n        return results\n    \n    def evaluate_theoretical_properties(self) -> Dict[str, Any]:\n        \"\"\"Evaluate theoretical convergence and stability properties\"\"\"\n        print(\"Evaluating theoretical properties...\")\n        \n        # Lyapunov stability analysis\n        stability_metrics = self.convergence_analyzer.lyapunov_stability_analysis(\n            self.msnar_framework)\n        \n        # Convergence bounds\n        convergence_bounds = self.convergence_analyzer.convergence_bounds(\n            self.msnar_framework.config, self.num_layers)\n        \n        return {\n            'stability_analysis': stability_metrics,\n            'convergence_bounds': convergence_bounds\n        }\n    \n    def run_comprehensive_evaluation(self) -> Dict[str, Any]:\n        \"\"\"Run complete comprehensive evaluation\"\"\"\n        \n        print(f\"\\nStarting Comprehensive MSNAR Evaluation\")\n        print(f\"{'='*60}\")\n        print(f\"Model: {self.args.model_name}\")\n        print(f\"Dataset: {self.args.data_name}\")\n        print(f\"State Dim: {self.state_dim}, Layers: {self.num_layers}\")\n        print(f\"{'='*60}\\n\")\n        \n        comprehensive_results = {}\n        \n        # 1. Basic Performance\n        print(\"\\n\" + \"=\"*40)\n        print(\"BASIC PERFORMANCE EVALUATION\")\n        print(\"=\"*40)\n        \n        basic_results = self.evaluate_basic_performance()\n        comprehensive_results['basic_performance'] = basic_results\n        \n        print(f\"Accuracy: {basic_results['accuracy']:.2f}%\")\n        print(f\"Average State Health: {basic_results['state_health']:.4f}\")\n        print(f\"Average Repair Ratio: {basic_results['repair_ratio']:.4f}\")\n        \n        # 2. Neuroplasticity Effectiveness\n        print(\"\\n\" + \"=\"*40)\n        print(\"NEUROPLASTICITY MECHANISMS\")\n        print(\"=\"*40)\n        \n        neuroplasticity_results = self.evaluate_neuroplasticity_mechanisms()\n        comprehensive_results['neuroplasticity_effectiveness'] = neuroplasticity_results\n        \n        for metric, improvement in neuroplasticity_results['improvement_metrics'].items():\n            if 'relative' in metric:\n                print(f\"{metric}: {improvement*100:.2f}%\")\n        \n        # 3. Robustness Evaluation\n        print(\"\\n\" + \"=\"*40)\n        print(\"ROBUSTNESS EVALUATION\")\n        print(\"=\"*40)\n        \n        # Noise robustness\n        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]\n        noise_results = self.robustness_evaluator.evaluate_noise_robustness(\n            self.test_loader, noise_levels)\n        comprehensive_results['noise_robustness'] = noise_results\n        \n        print(\"Noise Robustness Results:\")\n        for i, (noise, acc) in enumerate(zip(noise_levels, [noise_results['clean_accuracy']] + noise_results['noisy_accuracies'])):\n            print(f\"  Noise {noise:.1f}: {acc:.2f}% (Repair: {noise_results['repair_ratios'][i] if i < len(noise_results['repair_ratios']) else 0:.4f})\")\n        \n        # Adversarial robustness (if available)\n        if ADVERSARIAL_AVAILABLE and getattr(self.args, 'evaluate_adversarial', False):\n            attack_params = {\n                'fgsm': {'eps': 0.031},\n                'pgd': {'eps': 0.031, 'alpha': 0.007, 'steps': 10}\n            }\n            adversarial_results = self.robustness_evaluator.evaluate_adversarial_robustness(\n                self.test_loader, attack_params)\n            comprehensive_results['adversarial_robustness'] = adversarial_results\n            \n            print(\"\\nAdversarial Robustness Results:\")\n            for attack in adversarial_results['attacks']:\n                acc = adversarial_results['adversarial_accuracies'][attack]\n                repair = adversarial_results['repair_effectiveness'][attack]\n                print(f\"  {attack}: {acc:.2f}% (Repair: {repair:.4f})\")\n        \n        # 4. Theoretical Analysis\n        print(\"\\n\" + \"=\"*40)\n        print(\"THEORETICAL ANALYSIS\")\n        print(\"=\"*40)\n        \n        theoretical_results = self.evaluate_theoretical_properties()\n        comprehensive_results['theoretical_analysis'] = theoretical_results\n        \n        stability = theoretical_results['stability_analysis']\n        convergence = theoretical_results['convergence_bounds']\n        \n        print(f\"System Stability: {'STABLE' if stability['is_stable'] else 'UNSTABLE'}\")\n        print(f\"Spectral Radius: {stability['spectral_radius']:.4f}\")\n        print(f\"Convergence Rate: {convergence['convergence_rate']:.6f}\")\n        print(f\"Time to Convergence: {convergence['time_to_convergence']:.1f} steps\")\n        \n        # 5. Generate Visualizations\n        print(\"\\n\" + \"=\"*40)\n        print(\"GENERATING VISUALIZATIONS\")\n        print(\"=\"*40)\n        \n        self._generate_evaluation_visualizations(comprehensive_results)\n        \n        # 6. Save Results\n        self._save_evaluation_results(comprehensive_results)\n        \n        print(f\"\\n{'='*60}\")\n        print(\"COMPREHENSIVE EVALUATION COMPLETED\")\n        print(f\"Results saved to: {self.output_dir}\")\n        print(f\"Visualizations saved to: {self.visualization_dir}\")\n        print(f\"{'='*60}\")\n        \n        return comprehensive_results\n    \n    def _generate_evaluation_visualizations(self, results: Dict[str, Any]):\n        \"\"\"Generate evaluation visualizations\"\"\"\n        \n        try:\n            # Neuroplasticity dynamics\n            dynamics_path = os.path.join(self.visualization_dir, 'neuroplasticity_dynamics.png')\n            self.msnar_visualizer.plot_neuroplasticity_dynamics(\n                self.msnar_framework, save_path=dynamics_path)\n            \n            # Noise robustness plot\n            if 'noise_robustness' in results:\n                noise_results = results['noise_robustness']\n                \n                import matplotlib.pyplot as plt\n                \n                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))\n                \n                # Accuracy vs noise\n                noise_levels = noise_results['noise_levels']\n                accuracies = [noise_results['clean_accuracy']] + noise_results['noisy_accuracies']\n                \n                ax1.plot([0] + noise_levels, accuracies, 'b-o', linewidth=2, markersize=6)\n                ax1.set_xlabel('Noise Level')\n                ax1.set_ylabel('Accuracy (%)')\n                ax1.set_title('Noise Robustness')\n                ax1.grid(True, alpha=0.3)\n                \n                # Repair ratio vs noise\n                ax2.plot(noise_levels, noise_results['repair_ratios'], 'r-s', linewidth=2, markersize=6)\n                ax2.set_xlabel('Noise Level')\n                ax2.set_ylabel('Repair Ratio')\n                ax2.set_title('Repair Effectiveness vs Noise')\n                ax2.grid(True, alpha=0.3)\n                \n                plt.tight_layout()\n                noise_plot_path = os.path.join(self.visualization_dir, 'noise_robustness.png')\n                plt.savefig(noise_plot_path, dpi=300, bbox_inches='tight')\n                plt.close()\n                \n            print(\"Visualizations generated successfully\")\n            \n        except Exception as e:\n            print(f\"Warning: Visualization generation failed: {e}\")\n    \n    def _save_evaluation_results(self, results: Dict[str, Any]):\n        \"\"\"Save evaluation results to files\"\"\"\n        \n        # Convert to serializable format\n        serializable_results = self._make_serializable(results)\n        \n        # Save comprehensive results\n        results_path = os.path.join(self.analysis_dir, 'comprehensive_evaluation_results.json')\n        with open(results_path, 'w') as f:\n            json.dump(serializable_results, f, indent=2)\n        \n        # Save summary report\n        self._generate_summary_report(serializable_results)\n        \n    def _generate_summary_report(self, results: Dict[str, Any]):\n        \"\"\"Generate human-readable summary report\"\"\"\n        \n        report_path = os.path.join(self.output_dir, 'evaluation_summary_report.txt')\n        \n        with open(report_path, 'w') as f:\n            f.write(\"MSNAR Comprehensive Evaluation Summary Report\\n\")\n            f.write(\"=\" * 60 + \"\\n\\n\")\n            \n            # Configuration\n            f.write(f\"Model: {self.args.model_name}\\n\")\n            f.write(f\"Dataset: {self.args.data_name}\\n\")\n            f.write(f\"State Dimensions: {self.state_dim}\\n\")\n            f.write(f\"Number of Layers: {self.num_layers}\\n\\n\")\n            \n            # Basic Performance\n            if 'basic_performance' in results:\n                basic = results['basic_performance']\n                f.write(\"BASIC PERFORMANCE:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                f.write(f\"Accuracy: {basic['accuracy']:.2f}%\\n\")\n                f.write(f\"Average State Health: {basic['state_health']:.4f}\\n\")\n                f.write(f\"Average Repair Ratio: {basic['repair_ratio']:.4f}\\n\")\n                f.write(f\"Total Samples: {basic['total_samples']}\\n\\n\")\n            \n            # Neuroplasticity Effectiveness\n            if 'neuroplasticity_effectiveness' in results:\n                neuro = results['neuroplasticity_effectiveness']\n                f.write(\"NEUROPLASTICITY EFFECTIVENESS:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                \n                with_neuro = neuro['with_neuroplasticity']\n                without_neuro = neuro['without_neuroplasticity']\n                improvements = neuro['improvement_metrics']\n                \n                f.write(f\"With Neuroplasticity - Accuracy: {with_neuro['accuracy']:.2f}%\\n\")\n                f.write(f\"Without Neuroplasticity - Accuracy: {without_neuro['accuracy']:.2f}%\\n\")\n                f.write(f\"Accuracy Improvement: {improvements['accuracy_relative_improvement']*100:.2f}%\\n\")\n                f.write(f\"State Health Improvement: {improvements['state_health_relative_improvement']*100:.2f}%\\n\\n\")\n            \n            # Robustness\n            if 'noise_robustness' in results:\n                noise = results['noise_robustness']\n                f.write(\"NOISE ROBUSTNESS:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                f.write(f\"Clean Accuracy: {noise['clean_accuracy']:.2f}%\\n\")\n                \n                for i, (level, acc) in enumerate(zip(noise['noise_levels'], noise['noisy_accuracies'])):\n                    repair = noise['repair_ratios'][i] if i < len(noise['repair_ratios']) else 0\n                    f.write(f\"Noise {level:.1f} - Accuracy: {acc:.2f}%, Repair: {repair:.4f}\\n\")\n                f.write(\"\\n\")\n            \n            if 'adversarial_robustness' in results:\n                adv = results['adversarial_robustness']\n                f.write(\"ADVERSARIAL ROBUSTNESS:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                for attack in adv['attacks']:\n                    acc = adv['adversarial_accuracies'][attack]\n                    repair = adv['repair_effectiveness'][attack]\n                    f.write(f\"{attack} - Accuracy: {acc:.2f}%, Repair: {repair:.4f}\\n\")\n                f.write(\"\\n\")\n            \n            # Theoretical Analysis\n            if 'theoretical_analysis' in results:\n                theory = results['theoretical_analysis']\n                f.write(\"THEORETICAL ANALYSIS:\\n\")\n                f.write(\"-\" * 30 + \"\\n\")\n                \n                stability = theory['stability_analysis']\n                convergence = theory['convergence_bounds']\n                \n                f.write(f\"System Stability: {'STABLE' if stability['is_stable'] else 'UNSTABLE'}\\n\")\n                f.write(f\"Spectral Radius: {stability['spectral_radius']:.4f}\\n\")\n                f.write(f\"Stability Margin: {stability['stability_margin']:.4f}\\n\")\n                f.write(f\"Convergence Rate: {convergence['convergence_rate']:.6f}\\n\")\n                f.write(f\"Time to Convergence: {convergence['time_to_convergence']:.1f} steps\\n\")\n                f.write(f\"Theoretical Error Bound: {convergence['theoretical_error_bound']:.6f}\\n\\n\")\n            \n            f.write(\"MSNAR evaluation completed successfully.\\n\")\n            f.write(f\"This represents a novel neuroplasticity-inspired approach to Vision Mamba state repair.\\n\")\n        \n        print(f\"Summary report saved to: {report_path}\")\n    \n    def _make_serializable(self, data):\n        \"\"\"Convert tensors and other non-serializable objects to serializable format\"\"\"\n        if isinstance(data, torch.Tensor):\n            return data.detach().cpu().numpy().tolist()\n        elif isinstance(data, np.ndarray):\n            return data.tolist()\n        elif isinstance(data, dict):\n            return {k: self._make_serializable(v) for k, v in data.items()}\n        elif isinstance(data, list):\n            return [self._make_serializable(item) for item in data]\n        elif isinstance(data, (int, float, str, bool)):\n            return data\n        else:\n            return str(data)\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='MSNAR Comprehensive Evaluation')\n    \n    # Basic arguments\n    parser.add_argument('--model_name', default='vmamba_tiny', type=str, help='model name')\n    parser.add_argument('--data_name', default='imagenet', type=str, help='data name')\n    parser.add_argument('--num_classes', default=1000, type=int, help='num classes')\n    parser.add_argument('--batch_size', default=32, type=int, help='batch size for evaluation')\n    \n    # Paths\n    parser.add_argument('--model_path', required=True, type=str, help='path to trained model')\n    parser.add_argument('--data_test_dir', default='./data/test', type=str, help='test data dir')\n    parser.add_argument('--output_dir', default='./evaluation_results', type=str, help='output dir')\n    \n    # MSNAR configuration\n    parser.add_argument('--state_dim', default=256, type=int, help='state dimension')\n    parser.add_argument('--num_layers', default=6, type=int, help='number of layers')\n    \n    # Evaluation options\n    parser.add_argument('--evaluate_adversarial', action='store_true', help='evaluate adversarial robustness')\n    parser.add_argument('--evaluate_corrupted', action='store_true', help='evaluate on corrupted datasets')\n    \n    args = parser.parse_args()\n    \n    # Create evaluator and run comprehensive evaluation\n    evaluator = MSNARComprehensiveEvaluator(args)\n    results = evaluator.run_comprehensive_evaluation()\n    \n    print(\"\\nEvaluation completed successfully!\")\n    print(f\"Results available in: {args.output_dir}\")\n\n\nif __name__ == '__main__':\n    main()"
