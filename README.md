# Robust Graphs for Coordinated Cloud Attack Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.4%2B-green.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Project Overview
This repository contains a high-precision Graph Neural Network (GNN) framework designed to detect **coordinated cloud network attacks** (such as multi-source Botnet DDoS, microservice flooding, and lateral movement).

Traditional packet-level firewalls and flat Machine Learning models analyze traffic flows in isolation, making them vulnerable to distributed, low-volume coordinated attacks. This system treats the cloud infrastructure as a dynamic mathematical **Graph**, leveraging multi-head Graph Attention Networks (`GATv2Conv`) with dynamic edge weighting and topological centrality features to catch coordinated attack clusters with **99.99% accuracy**.

---

## ⚡ Performance Comparison Table

| Model Architecture | Accuracy | Precision | Recall | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.9981 | 0.9981 | 0.9981 | 0.9981 | Locked Baseline |
| **Random Forest** | 0.9997 | 0.9997 | 0.9997 | 0.9997 | Locked Baseline |
| **MLP (Simple)** | 0.9997 | 0.9997 | 0.9997 | 0.9997 | Locked Baseline |
| **Proposed GNN (Before Tuning)** | 0.9936 | 0.9936 | 0.9936 | 0.9936 | Baseline GCN |
| **Proposed GNN (After Tuning / Proposed)** | **0.9999** | **0.9999** | **0.9999** | **0.9999** | **SOTA Proposed (Multi-Head GATv2)** |

---

## 📂 Modular Architecture

The repository enforces clean modular separation across attack simulation, GNN architecture, training optimization, evaluation, and user interfaces:

*   **`coordinated_attack.py`**: Simulates multi-source cloud attack traffic (botnet burst injection) and extracts high-precision topological features (degree centrality, clustering coefficient, PageRank, bidirectional flow volume, dynamic edge weighting).
*   **`gnn_model.py`**: Advanced GNN architectures featuring multi-head `GATv2Conv` (dynamic attention) and `GraphSAGE` with Jumping Knowledge (`JK="cat"`), Layer Normalization, residual skip connections, and tuned dropout ($0.18$).
*   **`train_tune.py`**: Training loop with Focal Loss ($\gamma=2.0$, class weighting, label smoothing=$0.001$), AdamW optimizer, and Cosine Annealing learning rate scheduler.
*   **`evaluate.py`**: Benchmark comparison suite that formats baseline comparison tables and saves side-by-side Confusion Matrix and ROC curve plots.
*   **`streamlit_app.py`**: Interactive Web Dashboard powered by Streamlit for live training, parameter tuning, and dynamic baseline comparison.
*   **`main.py`**: Command-line entry point for headless batch training and CSV evaluation.
*   **`src/`**: Modular sub-components for data loading (`data_loader.py`), graph construction (`graph_builder.py`), and model utilities (`utils.py`).

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/lewiskirega/Gnn-cyber-detection.git
cd Gnn-cyber-detection
```

### Step 2: Create and Activate Virtual Environment
```bash
# For Linux / macOS:
python3 -m venv gnn-env
source gnn-env/bin/activate

# For Windows (Command Prompt):
python -m venv gnn-env
gnn-env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run the System

### 1. Interactive Web Dashboard (Streamlit UI)
Launch the interactive web application to visualize graph learning, live loss curves, and baseline comparison tables:

```bash
# Recommended command:
streamlit run streamlit_app.py

# Or referencing the virtual environment binary directly:
./gnn-env/bin/streamlit run streamlit_app.py
```
*(Navigates automatically to `http://localhost:8501`)*

### 2. Train & Tune Proposed GNN Architecture
Train the multi-head `GATv2` GNN with Focal Loss, Cosine Annealing, and topological feature enrichment:

```bash
python train_tune.py
```
*Outputs trained model checkpoint to `outputs/proposed_gnn_tuned.pth`.*

### 3. Generate Comparative Evaluation Plots & Tables
Generate the comparison markdown table, side-by-side Confusion Matrices, and ROC Curves comparing Before vs. After Tuning:

```bash
python evaluate.py
```
*Generates high-resolution PNG figures in the `outputs/` directory.*

### 4. Test Coordinated Attack Simulation & Feature Extraction
Run the attack simulator and topological feature engineer independently:

```bash
python coordinated_attack.py
```

### 5. Headless Command Line Interface (Batch Run)
Run batch training on dataset CSVs via `main.py`:

```bash
python main.py --data data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv --graph-mode auto --epochs 100
```

---

## 📊 Evaluation Artifacts

Generated evaluation figures are automatically saved under `outputs/`:
*   `outputs/confusion_matrix_comparison.png` — Side-by-side Confusion Matrix comparison (Before vs. After Tuning).
*   `outputs/roc_curve_comparison.png` — Comparative ROC Curves (AUC = 1.0000).
*   `outputs/evaluation_summary.json` — Structured metric summary.

---

## 📝 License
This project is open-source under the [MIT License](LICENSE).
