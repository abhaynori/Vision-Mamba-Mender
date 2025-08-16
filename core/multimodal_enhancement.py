import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from transformers import AutoTokenizer, AutoModel


class UnifiedMultiModalMambaFramework(nn.Module):
    """
    Novel Unified Multi-Modal Framework for Vision-Language Mamba Models
    
    Extends Vision-Mamba-Mender to handle multi-modal inputs (vision + text)
    with cross-modal state interaction analysis and repair.
    """
    
    def __init__(self,
                 vision_model,
                 text_encoder_name: str = "bert-base-uncased",
                 fusion_dim: int = 512,
                 num_fusion_layers: int = 3):
        super().__init__()
        
        self.vision_model = vision_model
        self.fusion_dim = fusion_dim
        self.num_fusion_layers = num_fusion_layers
        
        # Text encoder
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        
        # Cross-modal fusion components
        self.vision_projector = nn.Linear(vision_model.dims[-1], fusion_dim)
        self.text_projector = nn.Linear(self.text_encoder.config.hidden_size, fusion_dim)
        
        # Multi-modal Mamba fusion layers
        self.cross_modal_fusion = CrossModalMambaFusion(fusion_dim, num_fusion_layers)
        
        # Multi-modal interpretability components
        self.cross_modal_interpreter = CrossModalInterpreter(fusion_dim)
        self.modality_importance = ModalityImportanceAnalyzer(fusion_dim)
        
        # Multi-modal repair mechanism
        self.cross_modal_repair = CrossModalRepairMechanism(fusion_dim)
        
    def forward(self, 
                images: torch.Tensor,
                texts: List[str],
                return_interpretability: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-modal processing
        """
        batch_size = images.size(0)
        
        # Process vision modality
        vision_features = self.vision_model(images)
        vision_proj = self.vision_projector(vision_features)
        
        # Process text modality
        text_features = self._process_text(texts)
        text_proj = self.text_projector(text_features)
        
        # Cross-modal fusion
        fused_features, fusion_states = self.cross_modal_fusion(vision_proj, text_proj)
        
        results = {
            'fused_features': fused_features,
            'vision_features': vision_proj,
            'text_features': text_proj
        }
        
        if return_interpretability:
            # Multi-modal interpretability analysis
            interpretability = self.cross_modal_interpreter(
                vision_proj, text_proj, fusion_states)
            
            # Modality importance analysis
            importance = self.modality_importance(vision_proj, text_proj, fused_features)
            
            results.update({
                'interpretability': interpretability,
                'modality_importance': importance,
                'fusion_states': fusion_states
            })
        
        return results
    
    def _process_text(self, texts: List[str]) -> torch.Tensor:
        """
        Process text inputs through text encoder
        """
        # Tokenize texts
        tokenized = self.text_tokenizer(
            texts, padding=True, truncation=True, 
            return_tensors="pt", max_length=512)
        
        # Get text features
        with torch.no_grad():
            text_outputs = self.text_encoder(**tokenized)
            text_features = text_outputs.last_hidden_state.mean(dim=1)  # Global average pooling
        
        return text_features


class CrossModalMambaFusion(nn.Module):
    """
    Cross-modal fusion using Mamba-style selective state spaces
    """
    
    def __init__(self, fusion_dim: int, num_layers: int):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        self.num_layers = num_layers
        
        # Cross-modal attention mechanisms
        self.vision_to_text_attention = nn.ModuleList([
            CrossModalAttention(fusion_dim) for _ in range(num_layers)
        ])
        
        self.text_to_vision_attention = nn.ModuleList([
            CrossModalAttention(fusion_dim) for _ in range(num_layers)
        ])
        
        # Mamba-style state evolution
        self.state_evolution = nn.ModuleList([
            MambaStateEvolution(fusion_dim) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(fusion_dim) for _ in range(num_layers)
        ])
        
    def forward(self, 
                vision_features: torch.Tensor,
                text_features: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass through cross-modal fusion layers
        """
        current_vision = vision_features
        current_text = text_features
        fusion_states = []
        
        for layer_idx in range(self.num_layers):
            # Cross-modal attention
            attended_vision = self.vision_to_text_attention[layer_idx](
                current_vision, current_text)
            attended_text = self.text_to_vision_attention[layer_idx](
                current_text, current_vision)
            
            # Mamba-style state evolution
            evolved_vision, vision_state = self.state_evolution[layer_idx](
                current_vision, attended_vision)
            evolved_text, text_state = self.state_evolution[layer_idx](
                current_text, attended_text)
            
            # Layer normalization and residual connections
            current_vision = self.layer_norms[layer_idx](evolved_vision + current_vision)
            current_text = self.layer_norms[layer_idx](evolved_text + current_text)
            
            # Store fusion states for interpretability
            fusion_states.append({
                'layer': layer_idx,
                'vision_state': vision_state,
                'text_state': text_state,
                'cross_attention_vision': attended_vision,
                'cross_attention_text': attended_text
            })
        
        # Final fusion
        fused_features = (current_vision + current_text) / 2
        
        return fused_features, fusion_states


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism
    """
    
    def __init__(self, dim: int):
        super().__init__()
        
        self.dim = dim
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        
    def forward(self, 
                query_features: torch.Tensor,
                key_value_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal attention
        """
        batch_size = query_features.size(0)
        
        # Project features
        queries = self.query_proj(query_features)  # (B, dim)
        keys = self.key_proj(key_value_features)   # (B, dim)
        values = self.value_proj(key_value_features) # (B, dim)
        
        # Compute attention scores
        attention_scores = torch.bmm(
            queries.unsqueeze(1), keys.unsqueeze(2)
        ).squeeze() / (self.dim ** 0.5)  # (B,)
        
        attention_weights = F.softmax(attention_scores.unsqueeze(1), dim=1)  # (B, 1)
        
        # Apply attention
        attended_features = attention_weights * values  # (B, dim)
        
        return self.output_proj(attended_features)


class MambaStateEvolution(nn.Module):
    """
    Mamba-style selective state evolution for cross-modal features
    """
    
    def __init__(self, dim: int, state_dim: int = 16):
        super().__init__()
        
        self.dim = dim
        self.state_dim = state_dim
        
        # Selection mechanism
        self.selection_proj = nn.Linear(dim, state_dim)
        
        # State transition matrix
        self.transition_matrix = nn.Parameter(torch.randn(state_dim, state_dim))
        
        # Output projection
        self.output_proj = nn.Linear(state_dim, dim)
        
    def forward(self, 
                input_features: torch.Tensor,
                context_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve state based on input and context
        """
        batch_size = input_features.size(0)
        
        # Selection gate
        selection_gate = torch.sigmoid(self.selection_proj(input_features + context_features))
        
        # State evolution
        current_state = self.selection_proj(input_features)
        evolved_state = torch.matmul(current_state, self.transition_matrix)
        
        # Apply selection
        final_state = selection_gate * evolved_state + (1 - selection_gate) * current_state
        
        # Project back to feature dimension
        output_features = self.output_proj(final_state)
        
        return output_features, final_state


class CrossModalInterpreter(nn.Module):
    """
    Interpretability analysis for cross-modal interactions
    """
    
    def __init__(self, fusion_dim: int):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Interaction strength predictor
        self.interaction_strength = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Modality contribution analyzer
        self.contribution_analyzer = nn.Sequential(
            nn.Linear(fusion_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # Vision and text contributions
            nn.Softmax(dim=-1)
        )
        
    def forward(self,
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                fusion_states: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Analyze cross-modal interpretability
        """
        # Compute interaction strength
        combined_features = torch.cat([vision_features, text_features], dim=-1)
        interaction_strength = self.interaction_strength(combined_features)
        
        # Analyze modality contributions across layers
        layer_contributions = []
        for state in fusion_states:
            layer_vision = state['vision_state']
            layer_text = state['text_state']
            layer_fused = (layer_vision + layer_text) / 2
            
            layer_input = torch.cat([layer_vision, layer_text, layer_fused], dim=-1)
            contribution = self.contribution_analyzer(layer_input)
            layer_contributions.append(contribution)
        
        layer_contributions = torch.stack(layer_contributions, dim=1)  # (B, layers, 2)
        
        # Compute cross-modal alignment
        alignment_scores = []
        for state in fusion_states:
            alignment = F.cosine_similarity(
                state['vision_state'], state['text_state'], dim=-1)
            alignment_scores.append(alignment)
        
        alignment_scores = torch.stack(alignment_scores, dim=1)  # (B, layers)
        
        return {
            'interaction_strength': interaction_strength,
            'modality_contributions': layer_contributions,
            'cross_modal_alignment': alignment_scores,
            'mean_alignment': torch.mean(alignment_scores, dim=1)
        }


class ModalityImportanceAnalyzer(nn.Module):
    """
    Analyzes the importance of each modality for final predictions
    """
    
    def __init__(self, fusion_dim: int):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(fusion_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
        # Modality dropout for ablation analysis
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,
                vision_features: torch.Tensor,
                text_features: torch.Tensor,
                fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze modality importance
        """
        batch_size = vision_features.size(0)
        
        # Predict importance scores
        combined_input = torch.cat([vision_features, text_features, fused_features], dim=-1)
        importance_scores = self.importance_predictor(combined_input)
        
        # Ablation analysis
        vision_only_features = self.dropout(text_features) + vision_features
        text_only_features = self.dropout(vision_features) + text_features
        
        # Compute degradation when removing each modality
        vision_degradation = torch.norm(fused_features - vision_only_features, dim=-1)
        text_degradation = torch.norm(fused_features - text_only_features, dim=-1)
        
        # Normalize degradation scores
        total_degradation = vision_degradation + text_degradation + 1e-8
        vision_importance = vision_degradation / total_degradation
        text_importance = text_degradation / total_degradation
        
        return {
            'predicted_importance': importance_scores,
            'ablation_importance': torch.stack([vision_importance, text_importance], dim=-1),
            'vision_degradation': vision_degradation,
            'text_degradation': text_degradation
        }


class CrossModalRepairMechanism(nn.Module):
    """
    Repair mechanism for cross-modal inconsistencies
    """
    
    def __init__(self, fusion_dim: int):
        super().__init__()
        
        self.fusion_dim = fusion_dim
        
        # Inconsistency detector
        self.inconsistency_detector = nn.Sequential(
            nn.Linear(fusion_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Repair generators
        self.vision_repair = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh()
        )
        
        self.text_repair = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Tanh()
        )
        
    def detect_inconsistencies(self,
                              vision_features: torch.Tensor,
                              text_features: torch.Tensor) -> torch.Tensor:
        """
        Detect cross-modal inconsistencies
        """
        combined_features = torch.cat([vision_features, text_features], dim=-1)
        inconsistency_score = self.inconsistency_detector(combined_features)
        return inconsistency_score.squeeze()
    
    def repair_modalities(self,
                         vision_features: torch.Tensor,
                         text_features: torch.Tensor,
                         inconsistency_threshold: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repair modalities to reduce inconsistencies
        """
        inconsistency_scores = self.detect_inconsistencies(vision_features, text_features)
        
        # Only repair samples with high inconsistency
        repair_mask = inconsistency_scores > inconsistency_threshold
        
        if repair_mask.any():
            combined_features = torch.cat([vision_features, text_features], dim=-1)
            
            # Generate repairs
            vision_repair = self.vision_repair(combined_features)
            text_repair = self.text_repair(combined_features)
            
            # Apply repairs selectively
            repaired_vision = torch.where(
                repair_mask.unsqueeze(-1), 
                vision_features + vision_repair * 0.1,  # Small correction
                vision_features
            )
            
            repaired_text = torch.where(
                repair_mask.unsqueeze(-1),
                text_features + text_repair * 0.1,  # Small correction
                text_features
            )
            
            return repaired_vision, repaired_text
        
        return vision_features, text_features


class MultiModalMetrics:
    """
    Comprehensive metrics for multi-modal analysis
    """
    
    @staticmethod
    def compute_cross_modal_similarity(vision_features: torch.Tensor,
                                     text_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cross-modal similarity using multiple metrics
        """
        # Cosine similarity
        cosine_sim = F.cosine_similarity(vision_features, text_features, dim=-1)
        
        # Euclidean distance (normalized)
        euclidean_dist = torch.norm(vision_features - text_features, dim=-1)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combined similarity
        combined_sim = (cosine_sim + euclidean_sim) / 2
        
        return combined_sim
    
    @staticmethod
    def compute_modality_balance(vision_contributions: torch.Tensor,
                               text_contributions: torch.Tensor) -> torch.Tensor:
        """
        Compute balance between modalities
        """
        # Balance score (1.0 = perfect balance, 0.0 = one modality dominates)
        balance = 1.0 - torch.abs(vision_contributions - text_contributions)
        return balance.mean(dim=-1)
    
    @staticmethod
    def compute_fusion_quality(original_features: List[torch.Tensor],
                             fused_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute quality metrics for multi-modal fusion
        """
        vision_features, text_features = original_features
        
        # Information preservation
        vision_preservation = F.cosine_similarity(
            vision_features, fused_features, dim=-1).mean().item()
        text_preservation = F.cosine_similarity(
            text_features, fused_features, dim=-1).mean().item()
        
        # Synergy (how much fusion improves over individual modalities)
        vision_norm = torch.norm(vision_features, dim=-1).mean().item()
        text_norm = torch.norm(text_features, dim=-1).mean().item()
        fused_norm = torch.norm(fused_features, dim=-1).mean().item()
        
        synergy = fused_norm - max(vision_norm, text_norm)
        
        return {
            'vision_preservation': vision_preservation,
            'text_preservation': text_preservation,
            'average_preservation': (vision_preservation + text_preservation) / 2,
            'synergy_gain': synergy
        }


# Integration with existing framework
class EnhancedMultiModalMambaFramework:
    """
    Complete enhanced framework integrating multi-modal capabilities
    """
    
    def __init__(self, vision_model, original_framework):
        self.vision_model = vision_model
        self.original_framework = original_framework
        self.multimodal_framework = UnifiedMultiModalMambaFramework(vision_model)
        self.metrics = MultiModalMetrics()
        
    def comprehensive_multimodal_analysis(self,
                                        images: torch.Tensor,
                                        texts: List[str],
                                        labels: torch.Tensor) -> Dict:
        """
        Perform comprehensive multi-modal analysis and repair
        """
        # Multi-modal forward pass
        multimodal_results = self.multimodal_framework(
            images, texts, return_interpretability=True)
        
        # Extract components
        vision_features = multimodal_results['vision_features']
        text_features = multimodal_results['text_features']
        fused_features = multimodal_results['fused_features']
        interpretability = multimodal_results['interpretability']
        
        # Compute metrics
        cross_modal_similarity = self.metrics.compute_cross_modal_similarity(
            vision_features, text_features)
        
        modality_balance = self.metrics.compute_modality_balance(
            interpretability['modality_contributions'][:, :, 0],  # Vision contributions
            interpretability['modality_contributions'][:, :, 1]   # Text contributions
        )
        
        fusion_quality = self.metrics.compute_fusion_quality(
            [vision_features, text_features], fused_features)
        
        # Detect and repair inconsistencies
        repair_mechanism = self.multimodal_framework.cross_modal_repair
        inconsistency_scores = repair_mechanism.detect_inconsistencies(
            vision_features, text_features)
        
        repaired_vision, repaired_text = repair_mechanism.repair_modalities(
            vision_features, text_features)
        
        # Compute loss for training
        multimodal_loss = self._compute_multimodal_loss(
            multimodal_results, labels, inconsistency_scores)
        
        return {
            'multimodal_results': multimodal_results,
            'cross_modal_similarity': cross_modal_similarity,
            'modality_balance': modality_balance,
            'fusion_quality': fusion_quality,
            'inconsistency_scores': inconsistency_scores,
            'repaired_features': (repaired_vision, repaired_text),
            'multimodal_loss': multimodal_loss
        }
    
    def _compute_multimodal_loss(self,
                                multimodal_results: Dict,
                                labels: torch.Tensor,
                                inconsistency_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute comprehensive multi-modal loss
        """
        # Classification loss (if applicable)
        if hasattr(self.vision_model, 'classifier'):
            fused_features = multimodal_results['fused_features']
            classification_loss = F.cross_entropy(
                self.vision_model.classifier(fused_features), labels)
        else:
            classification_loss = torch.tensor(0.0)
        
        # Consistency loss
        consistency_loss = torch.mean(inconsistency_scores)
        
        # Alignment loss
        interpretability = multimodal_results['interpretability']
        alignment_loss = 1.0 - torch.mean(interpretability['mean_alignment'])
        
        # Modality balance loss
        contributions = interpretability['modality_contributions']
        balance_loss = torch.mean(torch.abs(
            contributions[:, :, 0] - contributions[:, :, 1]))
        
        # Combined loss
        total_loss = (classification_loss + 
                     0.1 * consistency_loss + 
                     0.1 * alignment_loss + 
                     0.05 * balance_loss)
        
        return total_loss
