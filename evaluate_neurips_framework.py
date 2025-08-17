"""
Comprehensive Evaluation Script for NeurIPS Framework

This script runs various experiments to validate the framework's capabilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Dict, List, Any
import json

from core.unified_neurips_framework import UnifiedNeurIPSFramework, UnifiedFrameworkConfig
from loaders.image_dataset import create_synthetic_dataset
from metrics.accuracy import compute_accuracy


class NeurIPSFrameworkEvaluator:
    """
    Comprehensive evaluator for the NeurIPS framework
    """
    
    def __init__(self, config: UnifiedFrameworkConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.framework = UnifiedNeurIPSFramework(config).to(self.device)
        
        self.results = {
            'component_performance': {},
            'integration_analysis': {},
            'computational_efficiency': {},
            'robustness_analysis': {}
        }
    
    def evaluate_individual_components(self) -> Dict[str, Any]:
        """
        Evaluate each component individually
        """
        
        print("🔍 Evaluating Individual Components...")
        
        component_results = {}
        test_inputs = torch.randn(4, 3, 32, 32).to(self.device)
        test_targets = torch.randint(0, 10, (4,)).to(self.device)
        
        # Test each component by enabling only one at a time
        components = ['msnar', 'quantum', 'hyperbolic', 'meta_learning', 'adversarial']
        
        for component in components:
            print(f"  Testing {component.upper()}...")
            
            # Create config with only this component enabled
            test_config = UnifiedFrameworkConfig()
            setattr(test_config, f'enable_{component}', True)
            for other_comp in components:
                if other_comp != component:
                    setattr(test_config, f'enable_{other_comp}', False)
            
            try:
                # Create framework with single component
                single_framework = UnifiedNeurIPSFramework(test_config).to(self.device)
                single_framework.eval()
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = single_framework(test_inputs, targets=test_targets)
                
                inference_time = time.time() - start_time
                
                # Extract metrics
                component_results[component] = {
                    'inference_time': inference_time,
                    'output_shape': tuple(outputs['final_output'].shape),
                    'loss_value': outputs['unified_loss'].item(),
                    'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                    'success': True
                }
                
                print(f"    ✅ {component}: {inference_time:.4f}s, Loss: {outputs['unified_loss'].item():.4f}")
                
            except Exception as e:
                component_results[component] = {
                    'error': str(e),
                    'success': False
                }
                print(f"    ❌ {component}: {e}")
        
        self.results['component_performance'] = component_results
        return component_results
    
    def evaluate_component_integration(self) -> Dict[str, Any]:
        """
        Evaluate how components work together
        """
        
        print("\n🔗 Evaluating Component Integration...")
        
        integration_results = {}
        test_inputs = torch.randn(4, 3, 32, 32).to(self.device)
        test_targets = torch.randint(0, 10, (4,)).to(self.device)
        
        # Test different combinations of components
        combinations = [
            ['msnar'],
            ['quantum'],
            ['hyperbolic'], 
            ['msnar', 'quantum'],
            ['msnar', 'hyperbolic'],
            ['quantum', 'hyperbolic'],
            ['msnar', 'quantum', 'hyperbolic'],
            ['msnar', 'quantum', 'hyperbolic', 'adversarial'],
            ['msnar', 'quantum', 'hyperbolic', 'meta_learning', 'adversarial']  # All
        ]
        
        for i, combo in enumerate(combinations):
            combo_name = '+'.join([c.upper() for c in combo])
            print(f"  Testing combination: {combo_name}")
            
            try:
                # Create config for this combination
                test_config = UnifiedFrameworkConfig()
                all_components = ['msnar', 'quantum', 'hyperbolic', 'meta_learning', 'adversarial']
                
                for comp in all_components:
                    setattr(test_config, f'enable_{comp}', comp in combo)
                
                # Create framework
                combo_framework = UnifiedNeurIPSFramework(test_config).to(self.device)
                combo_framework.eval()
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = combo_framework(test_inputs, targets=test_targets)
                
                inference_time = time.time() - start_time
                
                # Compute accuracy
                accuracy = compute_accuracy(outputs['final_output'], test_targets)
                
                integration_results[combo_name] = {
                    'components': combo,
                    'inference_time': inference_time,
                    'accuracy': accuracy,
                    'loss': outputs['unified_loss'].item(),
                    'output_shape': tuple(outputs['final_output'].shape),
                    'active_components': outputs.get('active_components', []),
                    'success': True
                }
                
                print(f"    ✅ {combo_name}: Acc={accuracy:.3f}, Loss={outputs['unified_loss'].item():.4f}, Time={inference_time:.4f}s")
                
            except Exception as e:
                integration_results[combo_name] = {
                    'error': str(e),
                    'success': False
                }
                print(f"    ❌ {combo_name}: {e}")
        
        self.results['integration_analysis'] = integration_results
        return integration_results
    
    def evaluate_computational_efficiency(self) -> Dict[str, Any]:
        """
        Evaluate computational efficiency and scaling
        """
        
        print("\n⚡ Evaluating Computational Efficiency...")
        
        efficiency_results = {}
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16] if torch.cuda.is_available() else [1, 2, 4]
        image_sizes = [32, 64] if torch.cuda.is_available() else [32]
        
        for batch_size in batch_sizes:
            for image_size in image_sizes:
                test_name = f"batch_{batch_size}_size_{image_size}"
                print(f"  Testing {test_name}...")
                
                try:
                    test_inputs = torch.randn(batch_size, 3, image_size, image_size).to(self.device)
                    test_targets = torch.randint(0, 10, (batch_size,)).to(self.device)
                    
                    self.framework.eval()
                    
                    # Warmup
                    with torch.no_grad():
                        _ = self.framework(test_inputs, targets=test_targets)
                    
                    # Actual timing
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    
                    with torch.no_grad():
                        outputs = self.framework(test_inputs, targets=test_targets)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    inference_time = time.time() - start_time
                    
                    # Memory usage
                    if torch.cuda.is_available():
                        memory_usage = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                    else:
                        memory_usage = peak_memory = 0
                    
                    efficiency_results[test_name] = {
                        'batch_size': batch_size,
                        'image_size': image_size,
                        'inference_time': inference_time,
                        'time_per_sample': inference_time / batch_size,
                        'memory_usage_mb': memory_usage,
                        'peak_memory_mb': peak_memory,
                        'throughput_samples_per_sec': batch_size / inference_time,
                        'success': True
                    }
                    
                    print(f"    ✅ {test_name}: {inference_time:.4f}s ({batch_size/inference_time:.2f} samples/s)")
                    
                except Exception as e:
                    efficiency_results[test_name] = {
                        'error': str(e),
                        'success': False
                    }
                    print(f"    ❌ {test_name}: {e}")
        
        self.results['computational_efficiency'] = efficiency_results
        return efficiency_results
    
    def evaluate_robustness(self) -> Dict[str, Any]:
        """
        Evaluate robustness to different inputs and conditions
        """
        
        print("\n🛡️ Evaluating Robustness...")
        
        robustness_results = {}
        
        # Test different input conditions
        test_conditions = {
            'normal': lambda: torch.randn(4, 3, 32, 32),
            'noisy': lambda: torch.randn(4, 3, 32, 32) + 0.5 * torch.randn(4, 3, 32, 32),
            'extreme_values': lambda: torch.randn(4, 3, 32, 32) * 10,
            'zeros': lambda: torch.zeros(4, 3, 32, 32),
            'ones': lambda: torch.ones(4, 3, 32, 32),
            'random_extreme': lambda: torch.randn(4, 3, 32, 32) * torch.randint(1, 100, (1,)).float()
        }
        
        for condition_name, input_gen in test_conditions.items():
            print(f"  Testing {condition_name} inputs...")
            
            try:
                test_inputs = input_gen().to(self.device)
                test_targets = torch.randint(0, 10, (4,)).to(self.device)
                
                self.framework.eval()
                
                with torch.no_grad():
                    outputs = self.framework(test_inputs, targets=test_targets)
                
                # Check for NaN/Inf in outputs
                has_nan = torch.isnan(outputs['final_output']).any().item()
                has_inf = torch.isinf(outputs['final_output']).any().item()
                loss_value = outputs['unified_loss'].item()
                
                robustness_results[condition_name] = {
                    'success': True,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'loss_finite': np.isfinite(loss_value),
                    'loss_value': loss_value,
                    'output_range': [
                        outputs['final_output'].min().item(),
                        outputs['final_output'].max().item()
                    ],
                    'stable': not (has_nan or has_inf or not np.isfinite(loss_value))
                }
                
                stability = "✅ Stable" if robustness_results[condition_name]['stable'] else "❌ Unstable"
                print(f"    {stability} - Loss: {loss_value:.4f}")
                
            except Exception as e:
                robustness_results[condition_name] = {
                    'error': str(e),
                    'success': False,
                    'stable': False
                }
                print(f"    ❌ {condition_name}: {e}")
        
        self.results['robustness_analysis'] = robustness_results
        return robustness_results
    
    def run_comprehensive_evaluation(self, save_dir: str = "experiments/evaluation") -> Dict[str, Any]:
        """
        Run all evaluation experiments
        """
        
        print("🚀 Starting Comprehensive NeurIPS Framework Evaluation")
        print("=" * 60)
        
        os.makedirs(save_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Run all evaluations
        try:
            self.evaluate_individual_components()
            self.evaluate_component_integration()
            self.evaluate_computational_efficiency()
            self.evaluate_robustness()
            
            total_time = time.time() - start_time
            
            # Add summary statistics
            self.results['evaluation_summary'] = {
                'total_evaluation_time': total_time,
                'device_used': str(self.device),
                'framework_config': self.config.__dict__,
                'cuda_available': torch.cuda.is_available(),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save results
            results_file = os.path.join(save_dir, 'comprehensive_evaluation.json')
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Generate report
            self.generate_evaluation_report(save_dir)
            
            print(f"\n🎉 Evaluation completed in {total_time:.2f}s!")
            print(f"📊 Results saved to {save_dir}")
            
            return self.results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_evaluation_report(self, save_dir: str):
        """
        Generate a comprehensive evaluation report
        """
        
        report_file = os.path.join(save_dir, 'evaluation_report.md')
        
        with open(report_file, 'w') as f:
            f.write("# NeurIPS Framework Comprehensive Evaluation Report\n\n")
            f.write(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**Device:** {self.device}\\n")
            f.write(f"**CUDA Available:** {torch.cuda.is_available()}\\n\\n")
            
            # Component Performance
            f.write("## Individual Component Performance\\n\\n")
            comp_perf = self.results.get('component_performance', {})
            for comp, metrics in comp_perf.items():
                if metrics.get('success', False):
                    f.write(f"- **{comp.upper()}**: ✅ Success\\n")
                    f.write(f"  - Inference Time: {metrics.get('inference_time', 0):.4f}s\\n")
                    f.write(f"  - Loss: {metrics.get('loss_value', 0):.4f}\\n")
                else:
                    f.write(f"- **{comp.upper()}**: ❌ Failed - {metrics.get('error', 'Unknown error')}\\n")
            
            # Integration Analysis
            f.write("\\n## Component Integration Analysis\\n\\n")
            integration = self.results.get('integration_analysis', {})
            for combo, metrics in integration.items():
                if metrics.get('success', False):
                    f.write(f"- **{combo}**: ✅ Success\\n")
                    f.write(f"  - Accuracy: {metrics.get('accuracy', 0):.3f}\\n")
                    f.write(f"  - Inference Time: {metrics.get('inference_time', 0):.4f}s\\n")
                else:
                    f.write(f"- **{combo}**: ❌ Failed\\n")
            
            # Efficiency
            f.write("\\n## Computational Efficiency\\n\\n")
            efficiency = self.results.get('computational_efficiency', {})
            if efficiency:
                f.write("| Test | Batch Size | Image Size | Time (s) | Throughput (samples/s) |\\n")
                f.write("|------|------------|------------|----------|------------------------|\\n")
                for test, metrics in efficiency.items():
                    if metrics.get('success', False):
                        f.write(f"| {test} | {metrics.get('batch_size', 0)} | {metrics.get('image_size', 0)} | ")
                        f.write(f"{metrics.get('inference_time', 0):.4f} | {metrics.get('throughput_samples_per_sec', 0):.2f} |\\n")
            
            # Robustness
            f.write("\\n## Robustness Analysis\\n\\n")
            robustness = self.results.get('robustness_analysis', {})
            stable_count = sum(1 for metrics in robustness.values() if metrics.get('stable', False))
            total_tests = len(robustness)
            f.write(f"**Stability Score:** {stable_count}/{total_tests} tests passed\\n\\n")
            
            for condition, metrics in robustness.items():
                status = "✅ Stable" if metrics.get('stable', False) else "❌ Unstable"
                f.write(f"- **{condition}**: {status}\\n")
        
        print(f"📋 Evaluation report saved to {report_file}")


def main():
    """Main evaluation function"""
    
    print("🔍 NeurIPS Framework Comprehensive Evaluation")
    print("=" * 50)
    
    # Create configuration
    config = UnifiedFrameworkConfig()
    config.batch_size = 4
    
    # Enable all components for full evaluation
    config.enable_msnar = True
    config.enable_quantum = True
    config.enable_hyperbolic = True
    config.enable_meta_learning = True
    config.enable_adversarial = True
    
    print(f"📋 Configuration:")
    print(f"  All components enabled: 5/5")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    # Create evaluator and run evaluation
    evaluator = NeurIPSFrameworkEvaluator(config)
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\\n📊 Evaluation Summary:")
        
        # Component success rate
        comp_perf = results.get('component_performance', {})
        successful_comps = sum(1 for m in comp_perf.values() if m.get('success', False))
        print(f"  Individual Components: {successful_comps}/{len(comp_perf)} working")
        
        # Integration success rate
        integration = results.get('integration_analysis', {})
        successful_integrations = sum(1 for m in integration.values() if m.get('success', False))
        print(f"  Integration Tests: {successful_integrations}/{len(integration)} working")
        
        # Robustness
        robustness = results.get('robustness_analysis', {})
        stable_tests = sum(1 for m in robustness.values() if m.get('stable', False))
        print(f"  Robustness Tests: {stable_tests}/{len(robustness)} stable")
        
        print("\\n🎯 Framework is ready for research and experimentation!")
    else:
        print("❌ Evaluation failed. Check the error logs.")


if __name__ == "__main__":
    main()
