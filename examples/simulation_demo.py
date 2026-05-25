"""
StructuralTime-Core v0.1: Simulation & Visualization Demo
==========================================================
This script demonstrates the end-to-end capabilities of the library:
  1. Setting up a Quartic Potential well and finding equilibria (Level B).
  2. Simulating structural trajectory changes using the RK4 Integrator.
  3. Calculating Bounded Temporal Density T(K) along the flow.
  4. Clustering trajectory states into the 5 Regimes using gamma.
  5. Generating and saving matplotlib visualization graphs.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import Core modules
from structural_time_core import (
    QuarticPotentialSolver,
    GradientFlowIntegrator,
    BoundedTemporalDensityCalculator,
    HybridRegimeClustering,
    VisualizationAPI
)

def run_demo():
    print("=== Starting StructuralTime-Core MVP Demo ===")
    
    # Ensure examples directory exists
    os.makedirs("examples", exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. Potential Well Setup
    # -------------------------------------------------------------------------
    # F(K) = K^4 - 2*K^2 (Double-well potential: minima at -1 and 1, max at 0)
    potential = QuarticPotentialSolver(a=1.0, b=0.0, c=-2.0, d=0.0)
    equilibria = potential.find_equilibria()
    
    print("\n[1] Potential Solver Equilibria:")
    for eq in equilibria:
        print(f"  - K = {eq['K']:5.2f} | Type: {eq['type']:8s} | F(K) = {eq['F']:5.2f}")
        
    # Plot Potential curve
    K_vals = np.linspace(-2.0, 2.0, 200)
    F_vals = [potential.compute_F(k) for k in K_vals]
    
    plt.figure(figsize=(8, 5))
    plt.plot(K_vals, F_vals, 'b-', label='Free Energy Potential F(K)')
    
    # Mark equilibria
    for eq in equilibria:
        color = 'go' if eq['type'] == 'stable' else 'ro'
        plt.plot(eq['K'], eq['F'], color, markersize=10, 
                 label=f"{eq['type'].capitalize()} Eq ({eq['K']:.1f})")
                 
    plt.title("Level B: Quartic Potential Energy Well (Double-Well)")
    plt.xlabel("Structural Operator State (K)")
    plt.ylabel("Potential Energy F(K)")
    plt.grid(True)
    plt.legend()
    plt.savefig("examples/potential_well.png", dpi=150)
    plt.close()
    print("  Saved potential well diagram to: examples/potential_well.png")
    
    # -------------------------------------------------------------------------
    # 2. Gradient Flow Simulation (RK4)
    # -------------------------------------------------------------------------
    # Start near unstable peak (K=0.1) and simulate flow towards stable well (K=1.0)
    integrator = GradientFlowIntegrator(gamma=0.05, dt=0.05)
    density_calc = BoundedTemporalDensityCalculator(alpha=1.5)
    
    K_state = 0.1
    K_eq = 1.0  # Stable well target
    
    history = []
    print("\n[2] Running Gradient Flow simulation (RK4)...")
    
    for t_step in range(120):
        # Compute derivative of potential
        dF_dK = potential.compute_dF_dK(K_state)
        
        # Save state before update
        prev_K = K_state
        
        # RK4 step
        K_state = integrator.step_rk4(K_state, potential.compute_dF_dK, u=0.0, noise_scale=0.02)
        
        # Velocity and distance to target
        velocity = (K_state - prev_K) / integrator.dt
        dist_to_eq = abs(K_state - K_eq)
        
        # Calculate experienced temporal density T(K)
        t_ops = density_calc.compute_T_ops(velocity, dist_to_eq)
        
        history.append({
            'step': t_step,
            'K': K_state,
            'velocity': velocity,
            'T_ops': t_ops
        })
        
    # Plot Trajectory & Temporal Density
    steps = [h['step'] for h in history]
    k_trajectory = [h['K'] for h in history]
    t_ops_density = [h['T_ops'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(9, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Simulation Step')
    ax1.set_ylabel('Operator State (K)', color=color)
    ax1.plot(steps, k_trajectory, color=color, linewidth=2, label='K-State trajectory')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(K_eq, color='g', linestyle='--', alpha=0.7, label='Equilibrium (K_eq=1.0)')
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Temporal Density T(K)', color=color)
    ax2.plot(steps, t_ops_density, color=color, linestyle='-.', linewidth=2, label='T(K) Experienced Time')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Level B: State Flow Trajectory and Bounded Temporal Density T(K)")
    fig.tight_layout()  
    plt.savefig("examples/trajectory.png", dpi=150)
    plt.close()
    print("  Saved K-state flow trajectory plot to: examples/trajectory.png")
    
    # -------------------------------------------------------------------------
    # 3. Theory-Guided Regime Clustering (incorporating gamma)
    # -------------------------------------------------------------------------
    print("\n[3] Fitting and predicting Temporal Regimes...")
    
    # Generate synthetic training dataset with clear features: [E_K, dK_dt, gamma]
    np.random.seed(42)
    n_per_regime = 50
    
    # Active: Moderate energy, moderate-high velocity, low decay
    active_data = np.random.normal(loc=[0.3, 0.4, 0.1], scale=0.05, size=(n_per_regime, 3))
    
    # Turbulent: High energy, high velocity, low decay
    turbulent_data = np.random.normal(loc=[0.8, 0.8, 0.15], scale=0.05, size=(n_per_regime, 3))
    
    # Decayed: Low energy, low velocity, high structural decay (gamma)
    decayed_data = np.random.normal(loc=[0.2, 0.05, 0.7], scale=0.05, size=(n_per_regime, 3))
    
    # Frozen: Low energy, low velocity, low structural decay (gamma)
    frozen_data = np.random.normal(loc=[0.1, 0.02, 0.1], scale=0.03, size=(n_per_regime, 3))
    
    # Critical: High/Intermediate energy, low-medium velocity, low-medium decay
    critical_data = np.random.normal(loc=[0.5, 0.2, 0.25], scale=0.05, size=(n_per_regime, 3))
    
    dataset = np.vstack([active_data, turbulent_data, decayed_data, frozen_data, critical_data])
    
    clustering = HybridRegimeClustering()
    regime_labels = clustering.fit_predict_regimes(dataset)
    
    # Plotting 3D Phase Space of Regimes
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    colors_map = {
        "Active": "blue",
        "Turbulent": "red",
        "Decayed": "purple",
        "Frozen": "cyan",
        "Critical": "orange"
    }
    
    for regime, color in colors_map.items():
        indices = [i for i, label in enumerate(regime_labels) if label == regime]
        if indices:
            ax.scatter(dataset[indices, 0], dataset[indices, 1], dataset[indices, 2], 
                       c=color, label=regime, s=30, alpha=0.7)
                       
    ax.set_xlabel('Systemic Energy (E_K)')
    ax.set_ylabel('Velocity (dK/dt)')
    ax.set_zlabel('Decay Rate (gamma)')
    plt.title('Theory-Guided Regime Clustering (KMeans + gamma Separation)')
    plt.legend()
    plt.savefig("examples/regime_clustering.png", dpi=150)
    plt.close()
    print("  Saved regime clustering 3D phase space to: examples/regime_clustering.png")
    
    # -------------------------------------------------------------------------
    # 4. Exporting Data
    # -------------------------------------------------------------------------
    # Export simulation log to JSON
    json_data = {
        'metadata': {
            'potential': 'K^4 - 2*K^2',
            'equilibria': [{'K': eq['K'], 'type': eq['type']} for eq in equilibria]
        },
        'trajectory': [{'step': h['step'], 'K': round(h['K'], 4), 'T_ops': round(h['T_ops'], 4)} for h in history]
    }
    
    success = VisualizationAPI.export_to_json(json_data, "examples/simulation_data.json")
    if success:
        print("  Exported formatted simulation log to: examples/simulation_data.json")
        
    print("\n=== Demo completed successfully! ===")

if __name__ == '__main__':
    run_demo()
