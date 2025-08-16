import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from collections import defaultdict


class TemporalStateEvolutionTracker(nn.Module):
    """
    Novel Temporal State Evolution Tracking for Vision Mamba Models
    
    Tracks how Mamba states evolve through layers and identifies critical
    transition points where state dynamics change significantly.
    """
    
    def __init__(self, 
                 num_layers: int,
                 state_dim: int,
                 sequence_length: int,
                 evolution_memory: int = 5):
        super().__init__()
        
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.evolution_memory = evolution_memory
        
        # State transition predictor
        self.transition_predictor = StateTransitionPredictor(state_dim)
        
        # Attention flow tracker
        self.attention_tracker = AttentionFlowTracker(num_layers, state_dim)
        
        # Critical point detector
        self.critical_detector = CriticalTransitionDetector(state_dim)
        
        # State evolution memory
        self.state_memory = defaultdict(list)
        
    def track_evolution(self,
                       state_sequence: List[torch.Tensor],
                       layer_indices: List[int]) -> Dict[str, torch.Tensor]:
        """
        Track state evolution across layers with detailed analysis
        """
        evolution_data = {
            'states': state_sequence,
            'layer_indices': layer_indices,
            'transitions': [],
            'critical_points': [],
            'attention_flows': [],
            'evolution_metrics': {}
        }
        
        # Analyze state transitions
        for i in range(len(state_sequence) - 1):
            current_state = state_sequence[i]
            next_state = state_sequence[i + 1]
            
            # Compute transition dynamics
            transition = self.transition_predictor(current_state, next_state)
            evolution_data['transitions'].append(transition)
            
            # Detect critical points
            critical_score = self.critical_detector(current_state, next_state)
            evolution_data['critical_points'].append(critical_score)
            
            # Track attention flow
            attention_flow = self.attention_tracker.compute_flow(
                current_state, next_state, layer_indices[i], layer_indices[i + 1])
            evolution_data['attention_flows'].append(attention_flow)
        
        # Compute evolution metrics
        evolution_data['evolution_metrics'] = self._compute_evolution_metrics(evolution_data)
        
        # Update memory
        self._update_evolution_memory(evolution_data)
        
        return evolution_data
    
    def predict_future_states(self,
                             current_states: List[torch.Tensor],
                             num_future_steps: int = 3) -> List[torch.Tensor]:
        """
        Predict future state evolution based on learned patterns
        """
        predicted_states = []
        last_state = current_states[-1]
        
        for step in range(num_future_steps):
            # Use transition predictor to forecast next state
            if len(current_states) >= 2:
                transition_pattern = self.transition_predictor(
                    current_states[-2], current_states[-1])
                predicted_state = last_state + transition_pattern['delta']
            else:
                # If insufficient history, use learned patterns from memory
                predicted_state = self._predict_from_memory(last_state)
            
            predicted_states.append(predicted_state)
            current_states.append(predicted_state)
            last_state = predicted_state
        
        return predicted_states
    
    def identify_semantic_phases(self,
                                evolution_data: Dict) -> Dict[str, List[int]]:
        """
        Identify semantic processing phases based on state evolution patterns
        """
        critical_points = torch.stack(evolution_data['critical_points'])
        
        # Find phase boundaries using critical point analysis
        phase_boundaries = []
        threshold = torch.mean(critical_points) + torch.std(critical_points)
        
        for i, score in enumerate(critical_points):
            if score > threshold:
                phase_boundaries.append(i)
        
        # Define semantic phases
        phases = {
            'low_level_features': list(range(0, min(len(self.state_memory) // 3, 
                                           phase_boundaries[0] if phase_boundaries else 5))),
            'mid_level_features': [],
            'high_level_semantics': []
        }
        
        if len(phase_boundaries) >= 2:
            phases['mid_level_features'] = list(range(phase_boundaries[0], phase_boundaries[1]))
            phases['high_level_semantics'] = list(range(phase_boundaries[1], len(critical_points)))
        elif len(phase_boundaries) == 1:
            mid_point = (phase_boundaries[0] + len(critical_points)) // 2
            phases['mid_level_features'] = list(range(phase_boundaries[0], mid_point))
            phases['high_level_semantics'] = list(range(mid_point, len(critical_points)))
        
        return phases
    
    def _compute_evolution_metrics(self, evolution_data: Dict) -> Dict[str, float]:
        """
        Compute comprehensive metrics for state evolution analysis
        """
        transitions = evolution_data['transitions']
        critical_points = evolution_data['critical_points']
        
        if not transitions:
            return {}
        
        # Stability metric
        transition_magnitudes = [t['magnitude'].item() for t in transitions]
        stability = 1.0 / (1.0 + np.std(transition_magnitudes))
        
        # Complexity metric
        critical_scores = torch.stack(critical_points)
        complexity = torch.mean(critical_scores).item()
        
        # Smoothness metric
        if len(transition_magnitudes) > 1:
            smoothness = 1.0 - np.mean(np.abs(np.diff(transition_magnitudes)))
        else:
            smoothness = 1.0
        
        # Information flow metric
        attention_flows = evolution_data['attention_flows']
        if attention_flows:
            info_flow = np.mean([flow['information_transfer'].item() 
                               for flow in attention_flows])
        else:
            info_flow = 0.0
        
        return {
            'stability': stability,
            'complexity': complexity,
            'smoothness': max(0.0, smoothness),
            'information_flow': info_flow
        }
    
    def _update_evolution_memory(self, evolution_data: Dict):
        """
        Update evolution memory with new patterns
        """
        # Store recent evolution patterns
        for i, transition in enumerate(evolution_data['transitions']):
            layer_pair = (evolution_data['layer_indices'][i], 
                         evolution_data['layer_indices'][i + 1])
            
            if len(self.state_memory[layer_pair]) >= self.evolution_memory:
                self.state_memory[layer_pair].pop(0)
            
            self.state_memory[layer_pair].append(transition)
    
    def _predict_from_memory(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Predict next state using patterns from evolution memory
        """
        if not self.state_memory:
            return current_state + torch.randn_like(current_state) * 0.1
        
        # Find most similar historical pattern
        similarities = []
        for layer_pair, transitions in self.state_memory.items():
            for transition in transitions:
                sim = F.cosine_similarity(
                    current_state.flatten().unsqueeze(0),
                    transition['source_state'].flatten().unsqueeze(0)
                ).item()
                similarities.append((sim, transition))
        
        if similarities:
            # Use most similar transition pattern
            _, best_transition = max(similarities, key=lambda x: x[0])
            return current_state + best_transition['delta']
        
        return current_state


class StateTransitionPredictor(nn.Module):
    """
    Predicts state transitions and analyzes transition dynamics
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Transition dynamics network
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        
        # Transition magnitude predictor
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                source_state: torch.Tensor, 
                target_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze transition between two states
        """
        # Flatten states
        source_flat = source_state.view(source_state.size(0), -1)
        target_flat = target_state.view(target_state.size(0), -1)
        
        # Concatenate for transition analysis
        transition_input = torch.cat([source_flat, target_flat], dim=1)
        
        # Predict transition dynamics
        predicted_delta = self.dynamics_net(transition_input)
        actual_delta = target_flat - source_flat
        
        # Compute transition magnitude
        magnitude = self.magnitude_predictor(transition_input)
        
        # Analyze transition characteristics
        direction_similarity = F.cosine_similarity(predicted_delta, actual_delta, dim=1)
        
        return {
            'predicted_delta': predicted_delta,
            'actual_delta': actual_delta,
            'magnitude': magnitude.squeeze(),
            'direction_similarity': direction_similarity,
            'source_state': source_state,
            'target_state': target_state,
            'delta': actual_delta.view_as(source_state)
        }


class AttentionFlowTracker(nn.Module):
    """
    Tracks attention flow and information transfer between layers
    """
    
    def __init__(self, num_layers: int, state_dim: int):
        super().__init__()
        
        self.num_layers = num_layers
        self.state_dim = state_dim
        
        # Information bottleneck network
        self.info_bottleneck = nn.Sequential(
            nn.Linear(state_dim, state_dim // 4),
            nn.Tanh(),
            nn.Linear(state_dim // 4, state_dim)
        )
        
    def compute_flow(self,
                    source_state: torch.Tensor,
                    target_state: torch.Tensor,
                    source_layer: int,
                    target_layer: int) -> Dict[str, torch.Tensor]:
        """
        Compute attention flow between layers
        """
        # Information transfer analysis
        source_compressed = self.info_bottleneck(source_state.view(-1, self.state_dim))
        reconstruction_error = F.mse_loss(source_compressed, source_state.view(-1, self.state_dim))
        
        # Flow direction and strength
        flow_vector = target_state.view(-1, self.state_dim) - source_state.view(-1, self.state_dim)
        flow_strength = torch.norm(flow_vector, dim=1).mean()
        
        # Information preservation
        mutual_info = self._estimate_mutual_information(source_state, target_state)
        
        return {
            'flow_vector': flow_vector,
            'flow_strength': flow_strength,
            'information_transfer': mutual_info,
            'compression_loss': reconstruction_error,
            'layer_distance': abs(target_layer - source_layer)
        }
    
    def _estimate_mutual_information(self,
                                   x: torch.Tensor,
                                   y: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information between two state tensors
        """
        # Simplified MI estimation using correlation
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)
        
        # Compute correlation matrix
        if x_flat.size(1) == y_flat.size(1):
            correlation = torch.mm(x_flat.t(), y_flat) / x_flat.size(0)
            mi_estimate = torch.trace(torch.abs(correlation)) / min(x_flat.size(1), y_flat.size(1))
        else:
            # Handle dimension mismatch
            min_dim = min(x_flat.size(1), y_flat.size(1))
            x_proj = x_flat[:, :min_dim]
            y_proj = y_flat[:, :min_dim]
            correlation = torch.mm(x_proj.t(), y_proj) / x_flat.size(0)
            mi_estimate = torch.trace(torch.abs(correlation)) / min_dim
        
        return mi_estimate


class CriticalTransitionDetector(nn.Module):
    """
    Detects critical transition points in state evolution
    """
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Critical point classifier
        self.criticality_classifier = nn.Sequential(
            nn.Linear(state_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                source_state: torch.Tensor,
                target_state: torch.Tensor) -> torch.Tensor:
        """
        Compute criticality score for state transition
        """
        # Flatten states
        source_flat = source_state.view(source_state.size(0), -1)
        target_flat = target_state.view(target_state.size(0), -1)
        
        # Compute state difference magnitude
        diff_magnitude = torch.norm(target_flat - source_flat, dim=1)
        
        # Compute relative change
        source_magnitude = torch.norm(source_flat, dim=1) + 1e-8
        relative_change = diff_magnitude / source_magnitude
        
        # Use classifier to determine criticality
        transition_input = torch.cat([source_flat, target_flat], dim=1)
        criticality_score = self.criticality_classifier(transition_input).squeeze()
        
        # Combine multiple indicators
        final_score = (criticality_score + relative_change) / 2
        
        return final_score


class TemporalVisualizationTools:
    """
    Visualization tools for temporal state evolution analysis
    """
    
    @staticmethod
    def plot_evolution_trajectory(evolution_data: Dict, 
                                save_path: Optional[str] = None):
        """
        Plot state evolution trajectory across layers
        """
        states = evolution_data['states']
        layer_indices = evolution_data['layer_indices']
        
        # Compute state norms for visualization
        state_norms = [torch.norm(state.view(-1)).item() for state in states]
        
        plt.figure(figsize=(12, 8))
        
        # Plot state magnitude evolution
        plt.subplot(2, 2, 1)
        plt.plot(layer_indices, state_norms, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Layer Index')
        plt.ylabel('State Magnitude')
        plt.title('State Magnitude Evolution')
        plt.grid(True, alpha=0.3)
        
        # Plot critical points
        if evolution_data['critical_points']:
            critical_scores = [cp.item() for cp in evolution_data['critical_points']]
            plt.subplot(2, 2, 2)
            plt.plot(layer_indices[1:], critical_scores, 'r-o', linewidth=2, markersize=6)
            plt.xlabel('Layer Index')
            plt.ylabel('Criticality Score')
            plt.title('Critical Transition Points')
            plt.grid(True, alpha=0.3)
        
        # Plot information flow
        if evolution_data['attention_flows']:
            info_flows = [af['information_transfer'].item() 
                         for af in evolution_data['attention_flows']]
            plt.subplot(2, 2, 3)
            plt.plot(layer_indices[1:], info_flows, 'g-o', linewidth=2, markersize=6)
            plt.xlabel('Layer Index')
            plt.ylabel('Information Transfer')
            plt.title('Information Flow')
            plt.grid(True, alpha=0.3)
        
        # Plot evolution metrics
        metrics = evolution_data['evolution_metrics']
        if metrics:
            plt.subplot(2, 2, 4)
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            plt.bar(metric_names, metric_values, color=['blue', 'red', 'green', 'orange'])
            plt.xlabel('Metrics')
            plt.ylabel('Values')
            plt.title('Evolution Quality Metrics')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    @staticmethod
    def create_state_heatmap(states: List[torch.Tensor],
                           layer_indices: List[int],
                           save_path: Optional[str] = None):
        """
        Create heatmap visualization of state evolution
        """
        # Flatten and stack states
        flattened_states = []
        for state in states:
            flat_state = state.view(-1)
            # Take subset if too large
            if len(flat_state) > 100:
                indices = torch.linspace(0, len(flat_state)-1, 100).long()
                flat_state = flat_state[indices]
            flattened_states.append(flat_state.detach().cpu().numpy())
        
        state_matrix = np.array(flattened_states)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(state_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='State Value')
        plt.xlabel('State Dimension')
        plt.ylabel('Layer Index')
        plt.title('State Evolution Heatmap')
        plt.yticks(range(len(layer_indices)), layer_indices)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# Integration example
class EnhancedTemporalMambaAnalyzer:
    """
    Enhanced analyzer combining temporal evolution tracking with existing framework
    """
    
    def __init__(self, model, existing_framework):
        self.model = model
        self.existing_framework = existing_framework
        self.temporal_tracker = TemporalStateEvolutionTracker(
            num_layers=len(model.layers),
            state_dim=model.dims[-1],
            sequence_length=224  # Typical image sequence length
        )
        self.visualizer = TemporalVisualizationTools()
        
    def comprehensive_temporal_analysis(self,
                                      inputs: torch.Tensor,
                                      save_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive temporal evolution analysis
        """
        # Extract states from model layers
        states, layer_indices = self._extract_layer_states(inputs)
        
        # Track evolution
        evolution_data = self.temporal_tracker.track_evolution(states, layer_indices)
        
        # Identify semantic phases
        semantic_phases = self.temporal_tracker.identify_semantic_phases(evolution_data)
        
        # Predict future evolution
        future_states = self.temporal_tracker.predict_future_states(states)
        
        # Generate visualizations
        if save_dir:
            self.visualizer.plot_evolution_trajectory(
                evolution_data, f"{save_dir}/evolution_trajectory.png")
            self.visualizer.create_state_heatmap(
                states, layer_indices, f"{save_dir}/state_heatmap.png")
        
        return {
            'evolution_data': evolution_data,
            'semantic_phases': semantic_phases,
            'future_predictions': future_states,
            'temporal_metrics': evolution_data['evolution_metrics']
        }
    
    def _extract_layer_states(self, inputs: torch.Tensor) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Extract states from each model layer during forward pass
        """
        states = []
        layer_indices = []
        
        # This would be implemented based on actual model architecture
        # For demonstration, create dummy states
        batch_size = inputs.size(0)
        for i, layer in enumerate(self.model.layers):
            # Placeholder state extraction
            state = torch.randn(batch_size, self.model.dims[min(i, len(self.model.dims)-1)])
            states.append(state)
            layer_indices.append(i)
        
        return states, layer_indices
