"""
Robustness testing for the GNN model.

This script tests the GNN's resilience against noisy environments by simulating:
1. Feature Noise: Adding Gaussian noise to the node features.
2. Structural Noise: Randomly dropping edges from the graph.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.pipeline import default_data_path, ExperimentConfig, run_experiment

def test_feature_noise(data_path: Path, max_samples: int):
    print("\n--- Testing Feature Noise Resilience ---")
    noise_levels = [0.0, 0.1, 0.5, 1.0, 2.0]
    results = []
    
    for noise in noise_levels:
        config = ExperimentConfig(
            data_path=data_path,
            graph_mode="flows",
            max_samples=max_samples,
            noise_level_features=noise,
            epochs=50
        )
        print(f"\nRunning with feature noise level {noise}...")
        res = run_experiment(config)
        results.append({
            "Feature Noise Level": noise, 
            "F1 Score": res.test_f1, 
            "Accuracy": res.test_accuracy
        })
        
    df = pd.DataFrame(results)
    print("\nFeature Noise Results:")
    print(df.to_string(index=False))

def test_structural_noise(data_path: Path, max_samples: int):
    print("\n--- Testing Structural Noise Resilience (Edge Dropout) ---")
    noise_levels = [0.0, 0.1, 0.3, 0.5, 0.8]
    results = []
    
    for noise in noise_levels:
        config = ExperimentConfig(
            data_path=data_path,
            graph_mode="flows",
            max_samples=max_samples,
            noise_level_edges=noise,
            epochs=50
        )
        print(f"\nRunning with edge dropout probability {noise}...")
        res = run_experiment(config)
        results.append({
            "Edge Dropout Prob": noise, 
            "F1 Score": res.test_f1, 
            "Accuracy": res.test_accuracy
        })
        
    df = pd.DataFrame(results)
    print("\nStructural Noise Results:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GNN Robustness")
    parser.add_argument("--data", type=str, default=str(default_data_path()), help="Path to CSV data.")
    parser.add_argument("--max-samples", type=int, default=15000, help="Max flow rows for testing.")
    args = parser.parse_args()
    
    data_p = Path(args.data)
    if not data_p.is_file():
        print(f"Dataset not found: {data_p}")
        raise SystemExit(1)
        
    test_feature_noise(data_p, max_samples=args.max_samples)
    test_structural_noise(data_p, max_samples=args.max_samples)
