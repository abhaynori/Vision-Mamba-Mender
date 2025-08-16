import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class AdaptiveMultiScaleInteractionLearner(nn.Module):
    """
    Novel Adaptive Multi-Scale State Interaction Learning (AMIL) Module
    
    This module dynamically selects and weights Mamba layer interactions
    based on input complexity and semantic content, enabling more precise
    and context-aware interpretability.
    """
    
    def __init__(self, 
                 num_layers: int,
                 hidden_dim: int,
                 num_scales: int = 3,
                 complexity_threshold: float = 0.7):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.complexity_threshold = complexity_threshold
        
        # Complexity estimation network
        self.complexity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d((14, 14)),
            nn.Flatten(),
            nn.Linear(14 * 14, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Layer importance predictor
        self.layer_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_layers),
            nn.Softmax(dim=-1)
        )
        
        # Multi-scale interaction weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        
        # Dynamic interaction fusion
        self.interaction_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_scales)
        ])
        
    def estimate_input_complexity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate input complexity using gradient magnitude and spatial variance
        """
        # Spatial variance as complexity indicator
        spatial_var = torch.var(x.view(x.size(0), x.size(1), -1), dim=2).mean(dim=1)
        
        # Edge complexity using Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        
        edges_x = F.conv2d(x.mean(dim=1, keepdim=True), sobel_x, padding=1)
        edges_y = F.conv2d(x.mean(dim=1, keepdim=True), sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2).mean(dim=(1, 2, 3))
        
        # Combine complexity measures
        complexity = (spatial_var + edge_magnitude) / 2
        return complexity.unsqueeze(-1)
    
    def select_adaptive_layers(self, 
                              complexity: torch.Tensor,
                              layer_features: List[torch.Tensor]) -> Tuple[List[int], torch.Tensor]:
        """
        Dynamically select layers based on input complexity and learned importance
        """
        batch_size = complexity.size(0)
        
        # Get global feature representation for layer importance
        global_features = torch.stack([f.mean(dim=(2, 3)) for f in layer_features], dim=1)
        global_features = global_features.mean(dim=1)  # Average across layers
        
        # Predict layer importance
        importance_weights = self.layer_importance(global_features)
        
        # Adaptive layer selection based on complexity
        if complexity.mean() > self.complexity_threshold:
            # High complexity: use more layers
            num_selected = min(self.num_layers, max(3, int(self.num_layers * 0.8)))
        else:
            # Low complexity: use fewer layers
            num_selected = max(2, int(self.num_layers * 0.4))
        
        # Select top-k important layers
        _, selected_indices = torch.topk(importance_weights, num_selected, dim=-1)
        
        return selected_indices.tolist(), importance_weights
    
    def compute_multi_scale_interactions(self,
                                       selected_features: List[torch.Tensor],
                                       selected_indices: List[int]) -> torch.Tensor:
        """
        Compute multi-scale state interactions with learned fusion
        """
        if len(selected_features) < 2:
            return selected_features[0] if selected_features else torch.zeros(1)
        
        interactions = []
        
        for scale_idx in range(self.num_scales):
            scale_interactions = []
            
            # Different scales: local (adjacent), medium (skip-1), global (all)
            if scale_idx == 0:  # Local interactions
                for i in range(len(selected_features) - 1):
                    feat1, feat2 = selected_features[i], selected_features[i + 1]
                    interaction = torch.cat([feat1.flatten(1), feat2.flatten(1)], dim=1)
                    fused = self.interaction_fusion[scale_idx](interaction)
                    scale_interactions.append(fused)
                    
            elif scale_idx == 1:  # Medium-range interactions
                step = max(1, len(selected_features) // 3)
                for i in range(0, len(selected_features) - step, step):
                    feat1, feat2 = selected_features[i], selected_features[i + step]
                    interaction = torch.cat([feat1.flatten(1), feat2.flatten(1)], dim=1)
                    fused = self.interaction_fusion[scale_idx](interaction)
                    scale_interactions.append(fused)
                    
            else:  # Global interactions
                first_feat = selected_features[0].flatten(1)
                last_feat = selected_features[-1].flatten(1)
                interaction = torch.cat([first_feat, last_feat], dim=1)
                fused = self.interaction_fusion[scale_idx](interaction)
                scale_interactions.append(fused)
            
            if scale_interactions:
                scale_interaction = torch.stack(scale_interactions, dim=1).mean(dim=1)
                interactions.append(scale_interaction)
        
        if not interactions:
            return torch.zeros(selected_features[0].size(0), self.hidden_dim, 
                             device=selected_features[0].device)
        
        # Weighted fusion of multi-scale interactions
        interactions = torch.stack(interactions, dim=1)  # (B, scales, hidden_dim)
        scale_weights = F.softmax(self.scale_weights, dim=0)
        
        return torch.sum(interactions * scale_weights.view(1, -1, 1), dim=1)
    
    def forward(self, 
                input_image: torch.Tensor,
                layer_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with adaptive multi-scale interaction learning
        """
        # Estimate input complexity
        complexity = self.estimate_input_complexity(input_image)
        
        # Select adaptive layers
        selected_indices, importance_weights = self.select_adaptive_layers(
            complexity, layer_features)
        
        # Extract selected features
        selected_features = [layer_features[i] for i in selected_indices[0]]  # Batch-wise selection for simplicity
        
        # Compute multi-scale interactions
        multi_scale_interactions = self.compute_multi_scale_interactions(
            selected_features, selected_indices[0])
        
        return {
            'complexity': complexity,
            'selected_layers': selected_indices,
            'layer_importance': importance_weights,
            'multi_scale_interactions': multi_scale_interactions,
            'adaptive_features': selected_features
        }


class CurriculumInteractivityLearner(nn.Module):
    """
    Implements curriculum learning for gradually increasing interpretability complexity
    """
    
    def __init__(self, max_stages: int = 5):
        super().__init__()
        self.max_stages = max_stages
        self.current_stage = 0
        
    def update_curriculum_stage(self, epoch: int, total_epochs: int):
        """Update curriculum stage based on training progress"""
        progress = epoch / total_epochs
        self.current_stage = min(self.max_stages - 1, 
                                int(progress * self.max_stages))
    
    def get_curriculum_complexity(self) -> Dict[str, float]:
        """Get current curriculum complexity settings"""
        stage_configs = [
            {'layer_ratio': 0.2, 'interaction_complexity': 0.3},  # Stage 0: Simple
            {'layer_ratio': 0.4, 'interaction_complexity': 0.5},  # Stage 1: Basic
            {'layer_ratio': 0.6, 'interaction_complexity': 0.7},  # Stage 2: Intermediate
            {'layer_ratio': 0.8, 'interaction_complexity': 0.8},  # Stage 3: Advanced
            {'layer_ratio': 1.0, 'interaction_complexity': 1.0},  # Stage 4: Full
        ]
        
        return stage_configs[self.current_stage]


# Integration with existing Vision-Mamba-Mender framework
class EnhancedMambaInterpreter:
    """
    Enhanced interpreter that integrates AMIL with existing framework
    """
    
    def __init__(self, model, original_constraint):
        self.model = model
        self.original_constraint = original_constraint
        self.amil = AdaptiveMultiScaleInteractionLearner(
            num_layers=len(model.layers),
            hidden_dim=model.dims[-1]
        )
        self.curriculum = CurriculumInteractivityLearner()
        
    def enhanced_loss_computation(self, 
                                 outputs: torch.Tensor,
                                 labels: torch.Tensor,
                                 input_images: torch.Tensor,
                                 epoch: int,
                                 total_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced loss with adaptive multi-scale interactions
        """
        # Update curriculum
        self.curriculum.update_curriculum_stage(epoch, total_epochs)
        curriculum_config = self.curriculum.get_curriculum_complexity()
        
        # Extract layer features
        layer_features = []
        for layer in self.model.layers:
            # Extract intermediate features (implementation depends on model structure)
            pass  # Placeholder for actual feature extraction
        
        # Apply AMIL
        amil_results = self.amil(input_images, layer_features)
        
        # Original losses
        original_external_loss = self.original_constraint.loss_external(outputs, labels, None)
        original_internal_loss = self.original_constraint.loss_internal(outputs, labels)
        
        # Adaptive interaction loss
        interaction_complexity = curriculum_config['interaction_complexity']
        adaptive_loss = torch.norm(amil_results['multi_scale_interactions']) * interaction_complexity
        
        # Complexity regularization
        complexity_reg = torch.mean(amil_results['complexity'] ** 2) * 0.1
        
        total_loss = (original_external_loss + 
                     original_internal_loss + 
                     adaptive_loss + 
                     complexity_reg)
        
        return {
            'total_loss': total_loss,
            'adaptive_loss': adaptive_loss,
            'complexity_regularization': complexity_reg,
            'amil_results': amil_results
        }
