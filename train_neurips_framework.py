"""
Training Script for NeurIPS-Level Vision-Mamba-Mender Framework

This script demonstrates how to train the complete framework with all novel components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from typing import Dict, List, Tuple
import json

# Import our framework
from core.unified_neurips_framework import UnifiedNeurIPSFramework, UnifiedFrameworkConfig
from loaders.image_dataset import create_synthetic_dataset
from metrics.accuracy import compute_accuracy


class NeurIPSFrameworkTrainer:
    """
    Comprehensive trainer for the NeurIPS framework
    """
    
    def __init__(self, 
                 config: UnifiedFrameworkConfig,
                 save_dir: str = "experiments"):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize framework
        print("🚀 Initializing NeurIPS Framework...")
        self.framework = UnifiedNeurIPSFramework(config).to(self.device)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.component_metrics = {
            'msnar': [],
            'quantum': [],
            'hyperbolic': [],
            'meta_learning': [],
            'adversarial': []
        }
        
    def create_datasets(self, 
                       train_size: int = 1000,
                       val_size: int = 200,
                       image_size: int = 32,
                       num_classes: int = 10) -> Tuple[DataLoader, DataLoader]:
        """Create synthetic datasets for training and validation"""
        
        print(f"📊 Creating datasets: train={train_size}, val={val_size}")
        
        # Create training data
        train_images = torch.randn(train_size, 3, image_size, image_size)
        train_labels = torch.randint(0, num_classes, (train_size,))
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Create validation data
        val_images = torch.randn(val_size, 3, image_size, image_size)
        val_labels = torch.randint(0, num_classes, (val_size,))
        val_dataset = TensorDataset(val_images, val_labels)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with proper learning rates for different components"""
        
        # Group parameters by component for different learning rates
        base_params = []
        novel_params = []
        
        for name, param in self.framework.named_parameters():
            if any(component in name for component in ['msnar', 'quantum', 'hyperbolic', 'meta', 'adversarial']):
                novel_params.append(param)
            else:
                base_params.append(param)
        
        # Use different learning rates for base model vs novel components
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': self.config.learning_rate},
            {'params': novel_params, 'lr': self.config.learning_rate * 0.5}  # Lower LR for novel components
        ], weight_decay=1e-4)
        
        return optimizer
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.framework.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        component_losses = {comp: 0.0 for comp in self.component_metrics.keys()}
        
        print(f"\n🔄 Epoch {epoch+1} Training...")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            try:
                outputs = self.framework(images, targets=labels)
                
                # Get predictions and compute accuracy
                predictions = outputs['final_output']
                loss = outputs['unified_loss']
                
                # Compute accuracy
                accuracy = compute_accuracy(predictions, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.framework.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
                
                # Track component-specific metrics if available
                if 'component_losses' in outputs:
                    for comp, comp_loss in outputs['component_losses'].items():
                        if comp in component_losses:
                            component_losses[comp] += comp_loss
                
                # Print progress
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}: "
                          f"Loss={loss.item():.4f}, Acc={accuracy:.3f}")
                    
            except Exception as e:
                print(f"⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        # Average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        for comp in component_losses:
            component_losses[comp] /= max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'component_losses': component_losses
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        
        self.framework.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        print(f"🔍 Epoch {epoch+1} Validation...")
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                try:
                    outputs = self.framework(images, targets=labels)
                    
                    predictions = outputs['final_output']
                    loss = outputs['unified_loss']
                    
                    accuracy = compute_accuracy(predictions, labels)
                    
                    total_loss += loss.item()
                    total_accuracy += accuracy
                    num_batches += 1
                    
                except Exception as e:
                    print(f"⚠️ Validation error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_accuracy = total_accuracy / max(num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy
        }
    
    def train(self, 
              num_epochs: int = 10,
              train_size: int = 1000,
              val_size: int = 200) -> Dict[str, List[float]]:
        """Main training loop"""
        
        print("🎯 Starting NeurIPS Framework Training!")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Training samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print("-" * 50)
        
        # Create datasets
        train_loader, val_loader = self.create_datasets(train_size, val_size)
        
        # Setup optimizer
        optimizer = self.setup_optimizer()
        
        # Setup learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Track metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Track component metrics
            for comp, loss in train_metrics.get('component_losses', {}).items():
                self.component_metrics[comp].append(loss)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.3f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val Acc:   {val_metrics['accuracy']:.3f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            # Component breakdown
            if train_metrics.get('component_losses'):
                print("  Component Losses:")
                for comp, loss in train_metrics['component_losses'].items():
                    if loss > 0:
                        print(f"    {comp}: {loss:.4f}")
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.2f}s!")
        
        # Save final metrics
        self.save_training_history()
        self.plot_training_curves()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
    
    def save_checkpoint(self, epoch: int, best: bool = False):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.framework.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'config': self.config.__dict__
        }
        
        filename = f"neurips_framework_{'best' if best else f'epoch_{epoch+1}'}.pth"
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
        if best:
            print(f"💾 Saved best model to {path}")
    
    def save_training_history(self):
        """Save training metrics to JSON"""
        
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'component_metrics': self.component_metrics
        }
        
        path = os.path.join(self.save_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"📊 Saved training history to {path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Acc')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Component losses
        ax3.set_title('Component Loss Evolution')
        for comp, losses in self.component_metrics.items():
            if losses and any(l > 0 for l in losses):
                ax3.plot(epochs[:len(losses)], losses, label=comp.upper())
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Component Loss')
        ax3.legend()
        ax3.grid(True)
        
        # Learning curve comparison
        ax4.plot(epochs, self.train_losses, 'b-', alpha=0.7, label='Train Loss')
        ax4.plot(epochs, self.val_losses, 'r-', alpha=0.7, label='Val Loss')
        ax4.fill_between(epochs, self.train_losses, alpha=0.3, color='blue')
        ax4.fill_between(epochs, self.val_losses, alpha=0.3, color='red')
        ax4.set_title('Learning Curves with Fill')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Saved training curves to {plot_path}")


def main():
    """Main training function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="NeurIPS Framework Training")
    parser.add_argument('--config', type=str, default=None,
                        help='Configuration name to load (from config_manager.py)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--train-size', type=int, default=200,
                        help='Training dataset size')
    parser.add_argument('--val-size', type=int, default=50,
                        help='Validation dataset size')
    
    args = parser.parse_args()
    
    print("🚀 NeurIPS Framework Training Script")
    print("=" * 50)
    
    # Load configuration if specified
    if args.config:
        try:
            from config_manager import ConfigurationManager
            config_manager = ConfigurationManager()
            experiment_config = config_manager.load_config(args.config)
            
            print(f"📂 Loaded configuration: {args.config}")
            print(f"🧪 Experiment: {experiment_config.experiment_name}")
            print(f"📝 Description: {experiment_config.description}")
            
            # Use configuration
            config = experiment_config.framework_config
            num_epochs = experiment_config.num_epochs
            learning_rate = experiment_config.learning_rate
            
            # Override with command line arguments if provided
            if hasattr(args, 'epochs') and args.epochs != 10:
                num_epochs = args.epochs
            if hasattr(args, 'lr') and args.lr != 0.0001:
                learning_rate = args.lr
                config.learning_rate = learning_rate
            if hasattr(args, 'batch_size') and args.batch_size != 8:
                config.batch_size = args.batch_size
            
        except Exception as e:
            print(f"❌ Failed to load configuration '{args.config}': {e}")
            print("🔄 Using default configuration...")
            config = UnifiedFrameworkConfig()
            config.batch_size = args.batch_size
            config.learning_rate = args.lr
            num_epochs = args.epochs
            
            # Enable all components for demonstration
            config.enable_msnar = True
            config.enable_quantum = True
            config.enable_hyperbolic = True
            config.enable_meta_learning = True
            config.enable_adversarial = True
    else:
        # Create default configuration
        config = UnifiedFrameworkConfig()
        config.batch_size = args.batch_size
        config.learning_rate = args.lr
        num_epochs = args.epochs
        
        # Enable all components for demonstration
        config.enable_msnar = True
        config.enable_quantum = True
        config.enable_hyperbolic = True
        config.enable_meta_learning = True
        config.enable_adversarial = True
    
    # Display current configuration
    print(f"\n📋 Training Configuration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  State dimension: {config.state_dim}")
    print(f"  Number of layers: {config.num_layers}")
    print(f"  Training size: {args.train_size}")
    print(f"  Validation size: {args.val_size}")
    
    # Show enabled components
    components = [
        ("MSNAR", config.enable_msnar),
        ("Quantum", config.enable_quantum), 
        ("Hyperbolic", config.enable_hyperbolic),
        ("Meta-Learning", config.enable_meta_learning),
        ("Adversarial", config.enable_adversarial)
    ]
    
    enabled_count = sum(1 for _, enabled in components if enabled)
    print(f"\n🧩 Enabled Components ({enabled_count}/5):")
    for comp_name, enabled in components:
        status = "✅" if enabled else "❌"
        print(f"  {status} {comp_name}")
    
    # Create trainer
    trainer = NeurIPSFrameworkTrainer(config, save_dir="experiments/neurips_training")
    
    # Start training
    try:
        results = trainer.train(
            num_epochs=num_epochs,
            train_size=args.train_size,
            val_size=args.val_size
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"📊 Final Results:")
        print(f"  Train Loss: {results['train_losses'][-1]:.4f}")
        print(f"  Val Loss: {results['val_losses'][-1]:.4f}")
        print(f"  Train Accuracy: {results['train_accuracies'][-1]:.3f}")
        print(f"  Val Accuracy: {results['val_accuracies'][-1]:.3f}")
        print(f"  Results saved to: experiments/neurips_training")
        
        return {
            'best_accuracy': max(results['val_accuracies']),
            'final_loss': results['val_losses'][-1],
            'training_time': 0,  # Will be computed in trainer
            'save_dir': 'experiments/neurips_training'
        }
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
