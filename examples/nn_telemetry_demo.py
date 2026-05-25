"""
StructuralTime-Core: Neural Network Telemetry Simulation Demo
============================================================
This script simulates two classic deep learning phenomena:
  1. Grokking (Sudden generalization after overfitting)
  2. Mode Collapse (GAN failure)

These simulations demonstrate how the Structural Time framework (Level A/B)
can map raw training metrics to K-states, analyze Experienced Temporal Density T(K),
and group training phases into temporal regimes using Theory-Guided Clustering.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import package components
from structural_time_core import (
    NeuralNetworkTelemetryAdapter,
    BoundedTemporalDensityCalculator,
    HybridRegimeClustering
)

# Set styling for premium aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Segoe UI'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

# Color Palette (Deep Purple, Coral, Teal, Slate, Gold)
PALETTE = {
    'primary': '#673AB7',      # Deep Purple
    'secondary': '#FF5722',    # Coral
    'accent': '#009688',       # Teal
    'dark': '#37474F',         # Slate/Dark Grey
    'warning': '#FFC107',      # Amber/Gold
    'active': '#2196F3',       # Light Blue
    'turbulent': '#E91E63',    # Pink/Red
    'decayed': '#9C27B0',      # Purple
    'frozen': '#00BCD4',       # Cyan
    'critical': '#FF9800'      # Orange
}

def simulate_grokking_telemetry():
    """
    Simulates training telemetry for a network showing the 'grokking' phenomenon.
    Phase 1 (Epoch 0-50): Active learning. Train/val loss decrease. Weight norm grows.
    Phase 2 (Epoch 51-200): Overfitting (Decayed). Train loss ~0, Val loss high/flat. Weight norm climbs (structural complexity).
    Phase 3 (Epoch 201-220): Phase transition (Critical/Turbulent). Validation accuracy jumps, val loss collapses. High gradients.
    Phase 4 (Epoch 221-300): Stabilized generalization (Frozen/Active). Low losses, stable weight norm, low gradients.
    """
    epochs = 300
    telemetry = []
    
    for epoch in range(epochs):
        if epoch <= 50:
            # Phase 1: Active
            progress = epoch / 50.0
            train_loss = 2.0 - progress * 1.8 + np.random.normal(0, 0.05)
            val_loss = 2.0 - progress * 0.5 + np.random.normal(0, 0.05)
            val_accuracy = 0.1 + progress * 0.2 + np.random.normal(0, 0.02)
            weight_norm = 10.0 + progress * 20.0 + np.random.normal(0, 0.5)
            gradient_norm = 1.5 + np.random.normal(0, 0.1)
            decay_gamma = 0.1 + progress * 0.1  # Low initial decay
            
        elif epoch <= 200:
            # Phase 2: Overfitting / Generalization Lag (Decayed)
            progress = (epoch - 50) / 150.0
            train_loss = 0.2 - progress * 0.19 + np.random.normal(0, 0.005)
            val_loss = 1.5 + progress * 0.2 + np.random.normal(0, 0.05)
            val_accuracy = 0.3 + np.random.normal(0, 0.01)
            weight_norm = 30.0 + progress * 40.0 + np.random.normal(0, 0.5)  # Weights continue accumulating complexity
            gradient_norm = 0.2 + np.random.normal(0, 0.02)
            decay_gamma = 0.5 + progress * 0.3  # Rising structural decay/dissonance
            
        elif epoch <= 220:
            # Phase 3: Grokking Transition (Critical/Turbulent)
            progress = (epoch - 200) / 20.0
            train_loss = 0.01 + np.random.normal(0, 0.001)
            val_loss = 1.7 - progress * 1.6 + np.random.normal(0, 0.05)
            val_accuracy = 0.3 + progress * 0.68 + np.random.normal(0, 0.01)
            weight_norm = 70.0 - progress * 3.0 + np.random.normal(0, 0.2)  # Slight contraction/simplification
            gradient_norm = 0.2 + np.sin(progress * np.pi) * 4.0 + np.random.normal(0, 0.1)  # Gradient burst
            decay_gamma = 0.8 - progress * 0.7  # Sharp collapse in structural decay
            
        else:
            # Phase 4: Generalized Generalization
            progress = (epoch - 220) / 80.0
            train_loss = 0.005 + np.random.normal(0, 0.0005)
            val_loss = 0.05 - progress * 0.04 + np.random.normal(0, 0.002)
            val_accuracy = 0.98 + progress * 0.02
            weight_norm = 67.0 + np.random.normal(0, 0.1)
            gradient_norm = 0.02 + np.random.normal(0, 0.005)
            decay_gamma = 0.05 + np.random.normal(0, 0.005)
            
        telemetry.append({
            'epoch': epoch,
            'train_loss': max(0.001, train_loss),
            'val_loss': max(0.001, val_loss),
            'val_accuracy': min(1.0, max(0.0, val_accuracy)),
            'weight_norm': weight_norm,
            'gradient_norm': max(0.001, gradient_norm),
            'decay_gamma': decay_gamma
        })
        
    return telemetry

def simulate_mode_collapse_telemetry():
    """
    Simulates training telemetry for a GAN undergoing Mode Collapse.
    Phase 1 (Epoch 0-40): Active/Dynamic equilibrium. Generator and discriminator improve.
    Phase 2 (Epoch 41-60): Implosion (Turbulent). Dis loss collapses, gen loss explodes, diversity drops.
    Phase 3 (Epoch 61-100): Collapsed state (Frozen). Stagnation, diversity ~0, gradients die.
    """
    epochs = 100
    telemetry = []
    
    for epoch in range(epochs):
        if epoch <= 40:
            # Phase 1: Healthy Active competition
            progress = epoch / 40.0
            train_loss = 0.8 - progress * 0.2 + np.random.normal(0, 0.04)   # Gen loss
            val_loss = 0.7 - progress * 0.1 + np.random.normal(0, 0.04)     # Dis loss
            val_accuracy = 0.8 + progress * 0.1 + np.random.normal(0, 0.02) # Diversity index
            weight_norm = 20.0 + progress * 15.0 + np.random.normal(0, 0.3)
            gradient_norm = 1.0 + np.sin(progress * 10) * 0.2 + np.random.normal(0, 0.05)
            decay_gamma = 0.1 + progress * 0.05
            
        elif epoch <= 60:
            # Phase 2: Impending Collapse (Turbulent)
            progress = (epoch - 40) / 20.0
            train_loss = 0.6 + progress * 4.4 + np.random.normal(0, 0.1)    # Gen loss spikes
            val_loss = 0.6 - progress * 0.58 + np.random.normal(0, 0.01)    # Dis loss drops to zero
            val_accuracy = 0.9 - progress * 0.8 + np.random.normal(0, 0.02) # Diversity collapses
            weight_norm = 35.0 + progress * 45.0 + np.random.normal(0, 1.0) # Weight explosion
            gradient_norm = 1.0 + progress * 6.0 + np.random.normal(0, 0.3) # Severe gradient spikes
            decay_gamma = 0.15 + progress * 0.65  # Heavy dissipation
            
        else:
            # Phase 3: Collapsed/Frozen
            progress = (epoch - 60) / 40.0
            train_loss = 5.0 + np.random.normal(0, 0.05)
            val_loss = 0.01 + np.random.normal(0, 0.001)
            val_accuracy = 0.08 + np.random.normal(0, 0.01)
            weight_norm = 80.0 + np.random.normal(0, 0.2)
            gradient_norm = 0.01 + np.random.normal(0, 0.001) # Stagnant gradient
            decay_gamma = 0.8 + np.random.normal(0, 0.02)
            
        telemetry.append({
            'epoch': epoch,
            'train_loss': max(0.001, train_loss), # Generator Loss
            'val_loss': max(0.001, val_loss),     # Discriminator Loss
            'val_accuracy': min(1.0, max(0.0, val_accuracy)), # Diversity
            'weight_norm': weight_norm,
            'gradient_norm': max(0.001, gradient_norm),
            'decay_gamma': decay_gamma
        })
        
    return telemetry

def analyze_nn_case(telemetry_data, title, save_prefix):
    print(f"\nAnalyzing '{title}' telemetry using Core Structural Time...")
    
    # 1. Instantiate Adapter (Max expected weight norm 100, max gradient 10)
    adapter = NeuralNetworkTelemetryAdapter(max_weight_norm=100.0, max_grad_norm=10.0)
    density_calc = BoundedTemporalDensityCalculator(alpha=2.0)
    
    # Map raw data to K-vector history
    k_history = []
    mapped_features = []
    
    for step in telemetry_data:
        k_vec = adapter.map_to_K(step)
        k_history.append(k_vec)
        
    k_history = np.array(k_history)
    n_steps = len(k_history)
    
    trajectory_metrics = []
    
    for i in range(n_steps):
        # Complexity (Energy) - map complexity dimension
        E_K = k_history[i, 0] 
        
        # Compute numerical velocity (K_dot)
        if i == 0:
            dk_dt = np.zeros(3)
        else:
            dk_dt = (k_history[i] - k_history[i-1]) # dt = 1 step
            
        dK_dt_norm = np.linalg.norm(dk_dt)
        
        # Decay rate gamma from telemetry
        gamma = telemetry_data[i]['decay_gamma']
        
        # Bounded temporal density experienced by the network
        # Target state is mapped to the final converged state
        dist_to_equilibrium = np.linalg.norm(k_history[i] - k_history[-1])
        t_ops = density_calc.compute_T_ops(dK_dt_norm, dist_to_equilibrium)
        
        trajectory_metrics.append({
            'epoch': telemetry_data[i]['epoch'],
            'E_K': E_K,
            'dK_dt_norm': dK_dt_norm,
            'gamma': gamma,
            'T_ops': t_ops,
            'k_vector': k_history[i]
        })
        
        mapped_features.append([E_K, dK_dt_norm, gamma])
        
    mapped_features = np.array(mapped_features)
    
    # 2. Fit Hybrid Regime Clustering
    clustering = HybridRegimeClustering()
    regimes = clustering.fit_predict_regimes(mapped_features)
    
    # 3. Create Visualization Plots
    epochs = [m['epoch'] for m in trajectory_metrics]
    E_K_vals = [m['E_K'] for m in trajectory_metrics]
    velocities = [m['dK_dt_norm'] for m in trajectory_metrics]
    t_ops_densities = [m['T_ops'] for m in trajectory_metrics]
    
    # Plot 1: Neural Net Loss & Accuracy vs Experienced Time T(K)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    
    # Top Panel: Raw Metrics
    if 'grokking' in save_prefix:
        ax1.plot(epochs, [s['train_loss'] for s in telemetry_data], color=PALETTE['secondary'], label='Training Loss', alpha=0.8)
        ax1.plot(epochs, [s['val_loss'] for s in telemetry_data], color=PALETTE['primary'], label='Validation Loss', alpha=0.8)
        ax1.set_ylabel('Loss', color=PALETTE['dark'])
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, [s['val_accuracy'] for s in telemetry_data], color=PALETTE['accent'], label='Validation Accuracy (Generalization)', linestyle='--')
        ax1_twin.set_ylabel('Accuracy', color=PALETTE['accent'])
        ax1_twin.tick_params(axis='y', labelcolor=PALETTE['accent'])
        ax1_twin.legend(loc='upper right')
    else: # Mode collapse
        ax1.plot(epochs, [s['train_loss'] for s in telemetry_data], color=PALETTE['secondary'], label='Generator Loss', alpha=0.8)
        ax1.plot(epochs, [s['val_loss'] for s in telemetry_data], color=PALETTE['primary'], label='Discriminator Loss', alpha=0.8)
        ax1.set_ylabel('GAN Loss', color=PALETTE['dark'])
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, [s['val_accuracy'] for s in telemetry_data], color=PALETTE['accent'], label='Sample Diversity', linestyle='--')
        ax1_twin.set_ylabel('Diversity Index', color=PALETTE['accent'])
        ax1_twin.tick_params(axis='y', labelcolor=PALETTE['accent'])
        ax1_twin.legend(loc='upper right')
        
    ax1.set_title(f"Raw Network Telemetry: {title}", fontsize=13, fontweight='bold', color=PALETTE['dark'])
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Bottom Panel: Structural Time Variables
    ax2.plot(epochs, E_K_vals, color=PALETTE['dark'], label=r'Systemic Energy $E_K$ (Complexity)', linewidth=2)
    ax2.plot(epochs, velocities, color=PALETTE['warning'], label=r'State Velocity $\|\dot{K}\|$ (Gradients)', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('State Metrics', color=PALETTE['dark'])
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, t_ops_densities, color=PALETTE['turbulent'], label=r'Temporal Density $T(K)$', linestyle='-.', linewidth=2)
    ax2_twin.set_ylabel(r'Experienced Time Density $T(K)$', color=PALETTE['turbulent'])
    ax2_twin.tick_params(axis='y', labelcolor=PALETTE['turbulent'])
    
    # Highlight Regimes along the timeline
    regime_colors = {
        'Active': (0.13, 0.59, 0.95, 0.1),     # Light Blue
        'Turbulent': (0.91, 0.12, 0.39, 0.1),  # Pink/Red
        'Decayed': (0.61, 0.15, 0.69, 0.1),    # Purple
        'Frozen': (0.0, 0.74, 0.83, 0.1),      # Cyan
        'Critical': (1.0, 0.6, 0.0, 0.1)       # Orange
    }
    
    # Detect transitions to plot shaded bands
    current_regime = regimes[0]
    start_epoch = epochs[0]
    for idx in range(1, len(epochs)):
        if regimes[idx] != current_regime or idx == len(epochs) - 1:
            color = regime_colors.get(current_regime, (0, 0, 0, 0))
            ax2.axvspan(start_epoch, epochs[idx], color=color, alpha=0.3)
            mid_epoch = (start_epoch + epochs[idx]) / 2.0
            ax2.text(mid_epoch, 0.05, current_regime, fontsize=9, fontweight='bold',
                     ha='center', va='bottom', color=np.array(color[:3])*0.8)
            current_regime = regimes[idx]
            start_epoch = epochs[idx]
            
    ax2.set_title("Mapped Structural Time Dynamics & Temporal Regimes", fontsize=13, fontweight='bold', color=PALETTE['dark'])
    ax2.set_xlabel('Epochs')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Merge legends
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"examples/{save_prefix}.png", dpi=150)
    plt.close()
    print(f"  Saved telemetry and regime diagram to: examples/{save_prefix}.png")
    
    return mapped_features, regimes

def run_nn_telemetry_demo():
    print("=== Starting StructuralTime-Core AI Use Case Simulation ===")
    os.makedirs("examples", exist_ok=True)
    
    # 1. Run Grokking Simulation
    grokking_data = simulate_grokking_telemetry()
    g_features, g_regimes = analyze_nn_case(grokking_data, "Grokking (Sudden Generalization)", "nn_grokking")
    
    # 2. Run Mode Collapse Simulation
    collapse_data = simulate_mode_collapse_telemetry()
    c_features, c_regimes = analyze_nn_case(collapse_data, "GAN Mode Collapse", "nn_mode_collapse")
    
    # 3. Build a combined 3D Regime Space Plot for deep learning telemetry
    print("\nCreating Combined 3D Deep Learning Regime Space Plot...")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors_map = {
        "Active": PALETTE['active'],
        "Turbulent": PALETTE['turbulent'],
        "Decayed": PALETTE['decayed'],
        "Frozen": PALETTE['frozen'],
        "Critical": PALETTE['critical']
    }
    
    # Combine datasets
    combined_features = np.vstack([g_features, c_features])
    combined_regimes = g_regimes + c_regimes
    
    for regime, color in colors_map.items():
        indices = [i for i, label in enumerate(combined_regimes) if label == regime]
        if indices:
            ax.scatter(combined_features[indices, 0], 
                       combined_features[indices, 1], 
                       combined_features[indices, 2], 
                       c=color, label=regime, s=40, alpha=0.6, edgecolors='none')
                       
    ax.set_xlabel('Systemic Energy ($E_K$ / Complexity)', fontweight='bold')
    ax.set_ylabel('Velocity ($dK/dt$ / Gradients)', fontweight='bold')
    ax.set_zlabel(r'Decay Rate ($\gamma$ / Dissipation)', fontweight='bold')
    ax.set_title('Theory-Guided Regime Clustering in AI Telemetry Space', fontsize=14, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    
    # Set nice camera angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig("examples/nn_clustering_3d.png", dpi=150)
    plt.close()
    print("  Saved 3D AI telemetry regime space plot to: examples/nn_clustering_3d.png")
    
    # Save simulation data for reference
    simulation_log = {
        'grokking': {
            'epochs': [s['epoch'] for s in grokking_data],
            'train_loss': [round(s['train_loss'], 4) for s in grokking_data],
            'val_loss': [round(s['val_loss'], 4) for s in grokking_data],
            'val_accuracy': [round(s['val_accuracy'], 4) for s in grokking_data],
            'regimes': g_regimes
        },
        'mode_collapse': {
            'epochs': [s['epoch'] for s in collapse_data],
            'train_loss': [round(s['train_loss'], 4) for s in collapse_data],
            'val_loss': [round(s['val_loss'], 4) for s in collapse_data],
            'val_accuracy': [round(s['val_accuracy'], 4) for s in collapse_data],
            'regimes': c_regimes
        }
    }
    
    with open("examples/nn_telemetry_data.json", "w", encoding="utf-8") as f:
        json.dump(simulation_log, f, indent=2)
    print("  Exported AI telemetry log to: examples/nn_telemetry_data.json")
    print("\n=== AI Telemetry Simulation Completed Successfully! ===")

if __name__ == '__main__':
    run_nn_telemetry_demo()
