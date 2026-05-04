"""
Run standard machine learning baseline models (Random Forest, Logistic Regression)
on the network flow data to establish a performance benchmark for the GNN.
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path so we can import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data_loader import TrafficDataLoader, DataLoaderConfig
from src.pipeline import default_data_path

def run_baselines(data_path: Path, label_col: str, max_samples: int = 25000, seed: int = 42):
    print(f"Loading data from {data_path}...")
    loader = TrafficDataLoader(DataLoaderConfig(label_column=label_col))
    
    # We use prepare_flow_dataset to get flat (X, y) matrices suitable for standard ML
    try:
        X, y, feature_columns, label_mapping = loader.prepare_flow_dataset(
            data_path,
            positive_class_name="DDoS",
            max_samples=max_samples,
            random_state=seed,
        )
    except ValueError as e:
        print(f"Data loading failed: {e}")
        return
        
    print(f"Data loaded. Shape: {X.shape}, Classes: {len(label_mapping)}")
    
    # Split data into train and test sets (GNN uses 15% for test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    
    # Scale features (important for Logistic Regression and MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to benchmark
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "MLP (Simple)": MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=seed)
    }
    
    results = []
    # Evaluate each model
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        if name in ["Logistic Regression", "MLP (Simple)"]:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
        acc = accuracy_score(y_test, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": p,
            "Recall": r,
            "F1-Score": f1
        })
        
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall   : {r:.4f}")
        print(f"F1-Score : {f1:.4f}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Baselines")
    parser.add_argument("--data", type=str, default=str(default_data_path()), help="Path to CSV data.")
    parser.add_argument("--label-col", type=str, default="Label", help="Label column name.")
    parser.add_argument("--max-samples", type=int, default=25000, help="Max flow rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    
    data_p = Path(args.data)
    if not data_p.is_file():
        print(f"Dataset not found: {data_p}")
        raise SystemExit(1)
        
    run_baselines(data_p, label_col=args.label_col, max_samples=args.max_samples, seed=args.seed)
